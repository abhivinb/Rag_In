"""Application HTTP routes."""

import logging
import secrets

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.api.dependencies import (
    get_database_session,
    get_conversation_repository,
    get_document_indexing_service,
    get_rag_service,
)
from app.api.chat_models import ChatRequest, ChatResponse
from app.api.models import DocumentUploadResponse
from app.conversation.access import conversation_token, valid_conversation_token
from app.conversation.models import ConversationMessage
from app.conversation.repository import ConversationRepository
from app.core.config import get_settings
from app.database.session import check_database_connection
from app.ingestion.errors import IngestionError
from app.ingestion.indexing import DocumentIndexingService
from app.ingestion.validators import MAX_FILE_SIZE_BYTES
from app.rag.models import RAGRequest, RAGResponse
from app.rag.phase9 import Phase9RAGService
from app.security.dependencies import require_api_key

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", tags=["system"])
async def health_check(request: Request) -> dict[str, str]:
    """Return service health after checking the live database connection."""
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        try:
            await check_database_connection(engine)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is unavailable.",
            ) from error
    return {"status": "healthy"}


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_database_session),
    indexing_service: DocumentIndexingService = Depends(get_document_indexing_service),
) -> DocumentUploadResponse:
    """Validate, embed, and persist one PDF, DOCX, or TXT upload."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_FILE_SIZE_BYTES + 1_048_576:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="The uploaded request exceeds the 50 MB limit.",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header.",
            ) from None
    try:
        result = await indexing_service.index(session, file.filename, file.file)
        return DocumentUploadResponse(
            document_id=result.document_id,
            file_name=result.file_name,
            file_type=result.file_type,
            chunk_count=result.chunk_count,
        )
    except IngestionError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document indexing failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The document could not be indexed.",
        ) from error


@router.post(
    "/ask",
    response_model=RAGResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ask_question(
    request: RAGRequest,
    session: AsyncSession = Depends(get_database_session),
    rag_service: Phase9RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """Answer one question through the real Phase 9 RAG pipeline."""
    try:
        return await rag_service.answer(session, request)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The RAG request could not be completed.",
        ) from error


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_database_session),
    rag_service: Phase9RAGService = Depends(get_rag_service),
    conversations: ConversationRepository = Depends(get_conversation_repository),
) -> ChatResponse:
    """Run one chat turn while preserving client and optional DB history."""
    settings = get_settings()
    conversation_id = request.conversation_id or str(uuid4())
    if settings.conversation_memory_enabled and request.conversation_id:
        if not valid_conversation_token(
            conversation_id, request.conversation_token, settings
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid conversation token is required.",
            )
    response_conversation_token = (
        conversation_token(conversation_id, settings)
        if settings.conversation_memory_enabled
        else secrets.token_urlsafe(32)
    )
    history = list(request.messages)
    if settings.conversation_memory_enabled and request.conversation_id:
        stored = await conversations.get_messages(
            session, conversation_id, settings.conversation_max_messages
        )
        if stored:
            history = stored

    try:
        response = await rag_service.answer(session, RAGRequest(query=request.query))
        assistant_message = ConversationMessage(role="assistant", content=response.answer)
        updated_history = (history + [
            ConversationMessage(role="user", content=request.query),
            assistant_message,
        ])[-settings.conversation_max_messages :]
        if settings.conversation_memory_enabled:
            await conversations.append(
                session,
                conversation_id,
                [
                    ConversationMessage(role="user", content=request.query),
                    assistant_message,
                ],
            )
        return ChatResponse(
            conversation_id=conversation_id,
            conversation_token=response_conversation_token,
            answer=response.answer,
            sources=response.sources,
            messages=updated_history,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The chat request could not be completed.",
        ) from error
