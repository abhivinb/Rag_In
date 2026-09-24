"""TXT loader tests."""

from pathlib import Path

import pytest

from app.ingestion.errors import DocumentExtractionError
from app.ingestion.loaders.text_loader import TextLoader


def test_text_loader_reads_utf8(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("Café\nRésumé", encoding="utf-8")

    assert TextLoader().load(source)[0].text == "Café\nRésumé"


def test_text_loader_handles_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")

    assert TextLoader().load(source)[0].text == ""


def test_text_loader_reports_unreadable_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.txt"

    with pytest.raises(DocumentExtractionError):
        TextLoader().load(source)