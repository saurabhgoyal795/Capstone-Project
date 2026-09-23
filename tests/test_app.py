"""Tests for the Streamlit web UI (app.py). No network / real LLM calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from complaint_processor import chains, llm as llm_module  # noqa: E402
from complaint_processor.config import Settings  # noqa: E402
from complaint_processor.schemas import ProcessingStatus  # noqa: E402
from test_pipeline import fake_email, fake_extraction, fake_summary  # noqa: E402

APP_PATH = str(ROOT / "app.py")


def _no_llm(*args, **kwargs):
    raise AssertionError("the LLM must not be initialised in this test")


@pytest.fixture
def offline(monkeypatch):
    """Fake the three AI chains and forbid real LLM construction."""
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_module, "get_llm", _no_llm)
    monkeypatch.setattr(app, "get_llm", _no_llm)
    monkeypatch.setattr(chains, "extract_case", fake_extraction)
    monkeypatch.setattr(chains, "generate_customer_email", fake_email)
    monkeypatch.setattr(chains, "generate_case_summary", fake_summary)
    return monkeypatch


def test_app_initial_render_has_no_exceptions(offline):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "AI Complaint & Case Processor"
    assert [b.label for b in at.button] == ["Process documents"]
    # Sample mode selects every sample file (including the corrupted one) by default.
    assert "corrupted_007.pdf" in at.multiselect[0].value
    assert not at.metric  # nothing processed yet

    at.radio[0].set_value("Upload your own").run()
    assert not at.exception


def test_app_process_click_renders_results(offline):
    from streamlit.testing.v1 import AppTest

    offline.setattr(llm_module, "get_llm", lambda settings: object())
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.multiselect[0].set_value(["complaint_003.txt", "corrupted_007.pdf"]).run()
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total files"] == "2" and metrics["Success"] == "1" and metrics["Failed"] == "1"
    assert any("corrupted_007.pdf" in e.value for e in at.error)
    assert at.session_state["run_count"] == 1


def test_run_processing_isolates_corrupted_file(offline, tmp_path):
    files = [ROOT / "data" / "complaint_003.txt", ROOT / "data" / "corrupted_007.pdf"]
    result = app.run_processing(files, 2, settings=Settings(llm_provider="ollama"), llm=object(),
                                work_dir=tmp_path)

    assert result.total == 2
    assert result.count(ProcessingStatus.SUCCESS) == 1 and result.failed == 1
    assert result.output_dir.is_relative_to(tmp_path)
    df = result.report_df()
    assert list(df["file_name"]) == ["complaint_003.txt", "corrupted_007.pdf"]
    assert df.loc[1, "errors"].startswith("ingestion failed")
    assert result.report_bytes().startswith(b"doc_id,")
    assert len(result.zip_bytes()) > 0
    # Inputs are copied, never written into data/.
    assert (result.output_dir.parent / "input" / "complaint_003.txt").exists()


def test_validate_uploads_limits():
    class Up:
        def __init__(self, name, size):
            self.name, self.size = name, size

    ups = [Up("a.txt", 10), Up("b.exe", 10), Up("c.pdf", app.MAX_UPLOAD_BYTES + 1), Up("d.docx", 0),
           Up("e.pdf", 5), Up("f.txt", 5)]
    accepted, errors = app.validate_uploads(ups)
    assert [u.name for u in accepted] == ["a.txt", "e.pdf"]
    assert any("Too many files" in e for e in errors)
    assert any("b.exe" in e for e in errors) and any("c.pdf" in e for e in errors)
    assert any("d.docx" in e for e in errors)
