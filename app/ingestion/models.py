"""Typed models exchanged by the ingestion pipeline."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExtractedSection(BaseModel):
    """Text extracted from one logical source section."""

    text: str
    page_number: int | None = Field(default=None, ge=1)


class DocumentMetadata(BaseModel):
    """Useful source metadata retained with a normalized document."""

    model_config = ConfigDict(extra="forbid")

    page_number: int | None = Field(default=None, ge=1)
    page_numbers: list[int] = Field(default_factory=list)
    file_size: int = Field(ge=0)
    content_type: str
    created_at: datetime
    source_type: str


class Document(BaseModel):
    """Normalized document ready for downstream chunking."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source: str
    file_name: str
    file_type: str
    text: str
    metadata: DocumentMetadata

    @classmethod
    def from_path(
        cls,
        *,
        document_id: str,
        source: Path,
        file_type: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> "Document":
        """Build a document while keeping filename handling in one place."""
        return cls(
            document_id=document_id,
            source=str(source),
            file_name=source.name,
            file_type=file_type,
            text=text,
            metadata=metadata,
        )