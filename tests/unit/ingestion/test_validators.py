"""Ingestion validation tests."""

from pathlib import Path

import pytest

from app.ingestion.errors import DocumentValidationError, UnsupportedFileTypeError
from app.ingestion.validators import detect_file_type, validate_file


def test_validator_accepts_supported_extensions(tmp_path: Path) -> None:
    for extension, file_type in ((".pdf", "pdf"), (".docx", "docx"), (".txt", "txt")):
        source = tmp_path / f"document{extension}"
        if extension == ".pdf":
            source.write_bytes(b"%PDF-1.7\ncontent")
        elif extension == ".docx":
            pytest.importorskip("docx").Document().save(source)
        else:
            source.write_text("text", encoding="utf-8")
        assert validate_file(source)[1] == file_type


def test_validator_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "document.csv"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        detect_file_type(source)


def test_validator_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentValidationError):
        validate_file(tmp_path / "missing.txt")


def test_validator_rejects_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")

    with pytest.raises(DocumentValidationError):
        validate_file(source)