"""Document format loaders."""

from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx_loader import DOCXLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import TextLoader

__all__ = ["DOCXLoader", "DocumentLoader", "PDFLoader", "TextLoader"]