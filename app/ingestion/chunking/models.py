"""Typed models for chunking configuration and output."""

from pydantic import BaseModel, ConfigDict, Field


class ChunkingConfig(BaseModel):
    """Configuration for deterministic recursive chunking."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = 1000
    chunk_overlap: int = 150
    minimum_chunk_size: int = 50


class ChunkMetadata(BaseModel):
    """Document and position metadata carried by a chunk."""

    model_config = ConfigDict(extra="forbid")

    source: str
    file_name: str
    file_type: str
    source_type: str
    page_number: int | None = Field(default=None, ge=1)
    page_numbers: list[int] = Field(default_factory=list)
    chunk_start: int = Field(ge=0)
    chunk_end: int = Field(ge=0)


class DocumentChunk(BaseModel):
    """Normalized chunk ready for future embedding and vector storage."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    content: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    metadata: ChunkMetadata