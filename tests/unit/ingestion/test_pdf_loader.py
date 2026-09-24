"""PDF loader tests."""

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from app.ingestion.errors import DocumentExtractionError
from app.ingestion.loaders.pdf_loader import PDFLoader


def _write_pdf(path: Path, pages: list[str]) -> None:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = NameObject("/F1")
        font_definition = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({font: font_definition})}
        )
        stream = StreamObject()
        stream._data = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        page[NameObject("/Contents")] = stream
    with path.open("wb") as output:
        writer.write(output)


def test_pdf_loader_extracts_single_page(tmp_path: Path) -> None:
    source = tmp_path / "single.pdf"
    _write_pdf(source, ["Hello PDF"])

    sections = PDFLoader().load(source)

    assert len(sections) == 1
    assert "Hello PDF" in sections[0].text
    assert sections[0].page_number == 1


def test_pdf_loader_extracts_multiple_pages_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "multi.pdf"
    _write_pdf(source, ["First page", "Second page"])

    sections = PDFLoader().load(source)

    assert [section.page_number for section in sections] == [1, 2]
    assert [section.text.strip() for section in sections] == ["First page", "Second page"]


def test_pdf_loader_handles_empty_page(tmp_path: Path) -> None:
    source = tmp_path / "empty-page.pdf"
    _write_pdf(source, [""])

    assert PDFLoader().load(source)[0].text == ""


def test_pdf_loader_rejects_invalid_pdf(tmp_path: Path) -> None:
    source = tmp_path / "invalid.pdf"
    source.write_bytes(b"not a PDF")

    with pytest.raises(DocumentExtractionError):
        PDFLoader().load(source)