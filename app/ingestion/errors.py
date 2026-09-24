"""Errors raised by the document ingestion pipeline."""


class IngestionError(Exception):
    """Base class for ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a file format is not supported."""


class DocumentValidationError(IngestionError):
    """Raised when a source file does not pass validation."""


class DocumentExtractionError(IngestionError):
    """Raised when a supported document cannot be extracted."""