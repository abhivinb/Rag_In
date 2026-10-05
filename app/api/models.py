"""HTTP response models for the API boundary."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    """Result of successfully indexing one uploaded document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    file_name: str
    file_type: str
    chunk_count: int = Field(ge=0)
    status: str = "indexed"
