"""Recursive chunking strategy tests."""

from datetime import UTC, datetime

from app.ingestion.chunking.models import ChunkingConfig
from app.ingestion.chunking.recursive import RecursiveChunkingStrategy
from app.ingestion.models import Document, DocumentMetadata


def make_document(text: str, page_numbers: list[int] | None = None) -> Document:
    return Document(
        document_id="document-id",
        source="data/raw/example.pdf",
        file_name="example.pdf",
        file_type="pdf",
        text=text,
        metadata=DocumentMetadata(
            file_size=len(text),
            content_type="application/pdf",
            created_at=datetime.now(UTC),
            source_type="file",
            page_numbers=page_numbers or [],
        ),
    )


def test_small_document_produces_one_chunk() -> None:
    document = make_document("A small document.")

    chunks = RecursiveChunkingStrategy(
        ChunkingConfig(chunk_size=100, chunk_overlap=10, minimum_chunk_size=1)
    ).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.text


def test_empty_document_produces_no_chunks() -> None:
    assert RecursiveChunkingStrategy().chunk(make_document(" \n\t ")) == []


def test_large_document_is_ordered_and_content_is_covered() -> None:
    text = "0123456789" * 30
    chunks = RecursiveChunkingStrategy(
        ChunkingConfig(chunk_size=50, chunk_overlap=10, minimum_chunk_size=5)
    ).chunk(make_document(text))

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].content == text[: len(chunks[0].content)]
    assert chunks[-1].metadata.chunk_end == len(text)
    covered = [False] * len(text)
    for chunk in chunks:
        for position in range(chunk.metadata.chunk_start, chunk.metadata.chunk_end):
            covered[position] = True
    assert all(covered)


def test_configured_overlap_is_present_without_empty_chunks() -> None:
    text = "abcdefghij" * 10
    chunks = RecursiveChunkingStrategy(
        ChunkingConfig(chunk_size=30, chunk_overlap=5, minimum_chunk_size=1)
    ).chunk(make_document(text))

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.content[-5:] == current.content[:5]
        assert current.content


def test_paragraph_boundary_is_preferred() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = RecursiveChunkingStrategy(
        ChunkingConfig(chunk_size=35, chunk_overlap=0, minimum_chunk_size=1)
    ).chunk(make_document(text))

    assert chunks[0].content == "First paragraph.\n\n"


def test_page_metadata_and_document_metadata_are_preserved() -> None:
    chunks = RecursiveChunkingStrategy(
        ChunkingConfig(chunk_size=10, chunk_overlap=1, minimum_chunk_size=1)
    ).chunk(make_document("A document with pages.", [1, 2]))

    assert chunks
    assert all(chunk.document_id == "document-id" for chunk in chunks)
    assert all(chunk.metadata.file_name == "example.pdf" for chunk in chunks)
    assert all(chunk.metadata.page_numbers == [1, 2] for chunk in chunks)
    assert all(chunk.metadata.page_number is None for chunk in chunks)


def test_same_document_produces_deterministic_chunks_and_ids() -> None:
    document = make_document("A deterministic document." * 5)
    strategy = RecursiveChunkingStrategy(
        ChunkingConfig(chunk_size=30, chunk_overlap=4, minimum_chunk_size=1)
    )

    first = strategy.chunk(document)
    second = strategy.chunk(document)

    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]