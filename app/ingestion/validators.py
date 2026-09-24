"""Validation and supported-format detection for ingestion sources."""

import zipfile
from pathlib import Path

from app.ingestion.errors import DocumentValidationError, UnsupportedFileTypeError

SUPPORTED_FILE_TYPES = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


def detect_file_type(file_path: Path) -> str:
    """Return the supported type for a path suffix or raise a clear error."""
    file_type = SUPPORTED_FILE_TYPES.get(file_path.suffix.lower())
    if file_type is None:
        raise UnsupportedFileTypeError("Only PDF, DOCX, and TXT files are supported.")
    return file_type


def validate_file(file_path: str | Path) -> tuple[Path, str]:
    """Validate existence, readability, size, encoding, and basic file signature."""
    path = Path(file_path)
    file_type = detect_file_type(path)

    if not path.is_file():
        raise DocumentValidationError("The source file does not exist.")
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            raise DocumentValidationError("The source file is empty.")
        if file_size > MAX_FILE_SIZE_BYTES:
            raise DocumentValidationError("The source file exceeds the 50 MB limit.")
        with path.open("rb") as source:
            header = source.read(8)
        if file_type == "pdf" and not header.startswith(b"%PDF-"):
            raise DocumentValidationError("The file is not a valid PDF document.")
        if file_type == "docx" and not _is_docx_archive(path):
            raise DocumentValidationError("The file is not a valid DOCX document.")
        if file_type == "txt":
            path.read_text(encoding="utf-8")
    except DocumentValidationError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as error:
        raise DocumentValidationError("The source file is not readable.") from error

    return path, file_type


def _is_docx_archive(path: Path) -> bool:
    """Check the minimal OOXML markers without extracting the archive."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        return "[Content_Types].xml" in names and "word/document.xml" in names