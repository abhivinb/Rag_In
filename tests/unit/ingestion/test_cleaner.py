"""Text cleaner tests."""

from app.ingestion.cleaner import clean_text


def test_cleaner_normalizes_whitespace_and_line_breaks() -> None:
    assert clean_text("  First   line\r\n\r\n\n  Second\tline  ") == "First line\n\nSecond line"


def test_cleaner_preserves_meaningful_text_and_punctuation() -> None:
    text = "Hello, World!\n\nThis is a paragraph."

    assert clean_text(text) == text