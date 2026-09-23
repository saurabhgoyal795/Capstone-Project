"""Tests for complaint_processor.ingestion."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

from complaint_processor.ingestion import IngestionError, extract_text, load_documents

EXTS = (".txt", ".pdf", ".docx")
ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_sample_data", ROOT / "scripts" / "generate_sample_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def test_extract_txt(tmp_path: Path) -> None:
    p = gen.write_txt(tmp_path / "a.txt", "Customer: Jane Doe\nIssue: late delivery")
    assert "Jane Doe" in extract_text(p)


def test_extract_txt_latin1_fallback(tmp_path: Path) -> None:
    p = tmp_path / "legacy.txt"
    p.write_bytes("Caf\xe9 complaint".encode("latin-1"))
    assert extract_text(p) == "Café complaint"


def test_extract_pdf(tmp_path: Path) -> None:
    p = gen.write_pdf(tmp_path / "c.pdf", "Test Complaint", [
        ("Customer Details", [("Customer Name", "Jane Doe"), ("Email", "jane@example.com")]),
        ("Description", "The router stopped working."),
    ])
    text = extract_text(p)
    assert "Jane Doe" in text and "jane@example.com" in text and "router stopped" in text


def test_extract_docx_includes_tables(tmp_path: Path) -> None:
    p = gen.write_docx(tmp_path / "c.docx", "Test Complaint", [
        ("Order Details", [("Order Number", "ORD-123")]),
        ("Description", "Parcel never arrived."),
    ])
    text = extract_text(p)
    assert "ORD-123" in text and "Order Number" in text and "Parcel never arrived." in text


def test_extract_unsupported_and_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(gen.write_txt(tmp_path / "x.csv", "a,b"))
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text(empty)


def test_load_documents_skips_and_captures_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    gen.write_txt(tmp_path / "b.txt", "Second   doc\n\n\n\nwith   spacing")
    gen.write_pdf(tmp_path / "a.pdf", "PDF Doc", [("Body", "First doc text")])
    gen.write_txt(tmp_path / "notes.csv", "not,supported")
    gen.write_txt(tmp_path / ".hidden.txt", "hidden")
    gen.write_corrupted_pdf(tmp_path / "broken.pdf")

    with caplog.at_level(logging.INFO, logger="complaint_processor.ingestion"):
        docs, errors = load_documents(tmp_path, EXTS, max_chars=10_000)

    assert [d.file_name for d in docs] == ["a.pdf", "b.txt"]  # sorted, hidden/csv skipped
    assert docs[1].text == "Second doc\n\nwith spacing"  # whitespace normalised
    assert docs[1].doc_id == "b" and docs[1].file_type == "txt"
    assert docs[1].char_count == len(docs[1].text)
    assert len(errors) == 1 and isinstance(errors[0], IngestionError)
    assert errors[0].file_name == "broken.pdf" and errors[0].error
    assert any("Skipping notes.csv" in r.message for r in caplog.records)


def test_load_documents_truncates(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    gen.write_txt(tmp_path / "long.txt", "x" * 500)
    with caplog.at_level(logging.WARNING, logger="complaint_processor.ingestion"):
        docs, _ = load_documents(tmp_path, EXTS, max_chars=100)
    assert docs[0].char_count == 100
    assert any("Truncating" in r.message for r in caplog.records)


def test_load_documents_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope", EXTS, max_chars=100)
