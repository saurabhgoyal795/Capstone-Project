"""Tests for pipeline orchestration and output writers (no real LLM calls)."""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from complaint_processor import chains
from complaint_processor.config import Settings
from complaint_processor.pipeline import CaseProcessingPipeline
from complaint_processor.schemas import (
    CaseStatus,
    CaseSummary,
    ComplaintCategory,
    ComplaintExtraction,
    CustomerEmail,
    LoadedDocument,
    ProcessingStatus,
    YesNo,
)
from complaint_processor.writers import REPORT_COLUMNS, OutputWriter


@dataclass
class FakeIngestionError:
    """Duck-typed stand-in for ingestion.IngestionError."""

    file_name: str
    path: str
    error: str


def make_doc(doc_id: str, text: str = "My router has been down for 3 days. I want a refund.") -> LoadedDocument:
    return LoadedDocument(
        doc_id=doc_id, file_name=f"{doc_id}.txt", file_type="txt",
        path=Path(f"/tmp/{doc_id}.txt"), text=text, char_count=len(text),
    )


def fake_extraction(llm, document_text: str) -> ComplaintExtraction:
    return ComplaintExtraction(
        customer_name="Asha Rao", email="ASHA@example.com", phone_number="+91 98765 43210",
        product_or_service="Home Broadband", complaint_category=ComplaintCategory.TECHNICAL_ISSUE,
        issue_description="Internet down for three days.", resolution_provided=None,
        is_complaint=YesNo.YES, escalation_required=YesNo.YES, supporting_document_available=YesNo.NO,
        overall_case_status=CaseStatus.OPEN,
    )


def fake_email(llm, document_text, extraction) -> CustomerEmail:
    return CustomerEmail(subject="We're on it: your broadband outage", body="Dear Asha,\n\nSorry...\n\nRegards,\nSupport")


def fake_summary(llm, document_text, extraction) -> CaseSummary:
    return CaseSummary(
        case_overview="Outage complaint.", key_issue="No internet.", action_taken="None yet.",
        current_status="Open", recommended_next_action="Dispatch technician.",
    )


def boom(*args, **kwargs):
    raise RuntimeError("LLM exploded")


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(chains, "extract_case", fake_extraction)
    monkeypatch.setattr(chains, "generate_customer_email", fake_email)
    monkeypatch.setattr(chains, "generate_case_summary", fake_summary)
    return monkeypatch


@pytest.fixture
def pipeline() -> CaseProcessingPipeline:
    return CaseProcessingPipeline(llm=object(), settings=Settings(max_workers=3))


def test_process_document_success(patched, pipeline):
    result = pipeline.process_document(make_doc("complaint_001"))
    assert result.status == ProcessingStatus.SUCCESS
    assert result.extraction.email == "asha@example.com"
    assert result.email.subject.startswith("We're on it")
    assert result.summary.recommended_next_action == "Dispatch technician."
    assert result.errors == []
    assert result.duration_seconds >= 0


def test_extraction_failure_marks_failed_and_skips_downstream(patched, pipeline):
    calls = []
    patched.setattr(chains, "extract_case", boom)
    patched.setattr(chains, "generate_customer_email", lambda *a: calls.append("email"))
    result = pipeline.process_document(make_doc("complaint_002"))
    assert result.status == ProcessingStatus.FAILED
    assert result.extraction is None and result.email is None and result.summary is None
    assert "extraction failed" in result.errors[0] and "LLM exploded" in result.errors[0]
    assert calls == []


def test_summary_failure_marks_partial(patched, pipeline):
    patched.setattr(chains, "generate_case_summary", boom)
    result = pipeline.process_document(make_doc("complaint_003"))
    assert result.status == ProcessingStatus.PARTIAL
    assert result.email is not None and result.summary is None
    assert len(result.errors) == 1 and "case_summary failed" in result.errors[0]


def test_email_and_summary_run_in_parallel(patched, pipeline):
    barrier = threading.Barrier(2, timeout=5)

    def slow_email(*a):
        barrier.wait()  # deadlocks (times out) unless summary runs concurrently
        return fake_email(*a)

    def slow_summary(*a):
        barrier.wait()
        return fake_summary(*a)

    patched.setattr(chains, "generate_customer_email", slow_email)
    patched.setattr(chains, "generate_case_summary", slow_summary)
    result = pipeline.process_document(make_doc("complaint_004"))
    assert result.status == ProcessingStatus.SUCCESS


def test_run_batch_preserves_order_and_isolates_failures(patched, pipeline):
    def jittery_extract(llm, text):
        time.sleep(random.uniform(0, 0.05))
        if "BAD" in text:
            raise ValueError("unparseable")
        return fake_extraction(llm, text)

    patched.setattr(chains, "extract_case", jittery_extract)
    docs = [make_doc(f"complaint_{i:03d}", "BAD doc" if i == 3 else "fine doc") for i in range(1, 7)]
    results = pipeline.run_batch(docs, max_workers=4)
    assert [r.doc_id for r in results] == [d.doc_id for d in docs]
    assert results[2].status == ProcessingStatus.FAILED
    assert sum(r.status == ProcessingStatus.SUCCESS for r in results) == 5


def test_run_batch_empty(pipeline):
    assert pipeline.run_batch([], max_workers=2) == []


def test_writers_create_expected_files_and_report(patched, pipeline, tmp_path):
    patched.setattr(chains, "generate_case_summary",
                    lambda llm, text, ext: boom() if "partial" in text else fake_summary(llm, text, ext))
    docs = [make_doc("complaint_001"), make_doc("complaint_002", "partial doc")]
    results = pipeline.run_batch(docs, max_workers=2)
    ingestion_errors = [FakeIngestionError("broken.pdf", "/data/broken.pdf", "PDF has no extractable text")]

    out = tmp_path / "output"
    writer = OutputWriter(out)
    (out / "structured_data").mkdir(parents=True)
    stale = out / "structured_data" / "old.json"
    stale.write_text("{}")
    writer.prepare()
    assert not stale.exists()

    writer.write_results(results)
    report = writer.write_report(results, ingestion_errors)
    summary_path = writer.write_run_summary(results, ingestion_errors, total_seconds=1.23,
                                            provider="fake", model="fake-model", max_workers=2)

    data = json.loads((out / "structured_data" / "complaint_001.json").read_text())
    assert data["doc_id"] == "complaint_001" and data["status"] == "success"
    assert data["extraction"]["complaint_category"] == "Technical Issue"

    email_txt = (out / "customer_emails" / "complaint_001_email.txt").read_text()
    assert email_txt.startswith("Subject: We're on it") and "\n\nDear Asha" in email_txt

    md = (out / "case_summaries" / "complaint_001_summary.md").read_text()
    assert md.startswith("# Case Summary: complaint_001") and "## Recommended Next Action" in md
    assert not (out / "case_summaries" / "complaint_002_summary.md").exists()  # partial: summary failed
    assert (out / "customer_emails" / "complaint_002_email.txt").exists()

    df = pd.read_csv(report, keep_default_na=False)
    assert list(df.columns) == REPORT_COLUMNS
    assert len(df) == 3
    by_file = df.set_index("file_name")
    assert by_file.loc["complaint_001.txt", "status"] == "success"
    assert by_file.loc["complaint_001.txt", "escalation_required"] == "Yes"
    assert by_file.loc["complaint_002.txt", "status"] == "partial"
    assert "case_summary failed" in by_file.loc["complaint_002.txt", "errors"]
    assert by_file.loc["broken.pdf", "status"] == "failed"
    assert "PDF has no extractable text" in by_file.loc["broken.pdf", "errors"]

    run = json.loads(summary_path.read_text())
    assert run["total_files"] == 3 and run["success"] == 1 and run["partial"] == 1 and run["failed"] == 1
    assert run["provider"] == "fake" and run["model"] == "fake-model"
