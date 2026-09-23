"""Streamlit web UI for the AI Customer Complaint & Case Processing System.

Run with::

    streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

The processing logic lives in :func:`run_processing` (a plain function, no Streamlit
calls) so it can be tested independently of the UI.
"""

from __future__ import annotations

import io
import logging
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from complaint_processor.config import PROJECT_ROOT, Settings, get_settings  # noqa: E402
from complaint_processor.ingestion import IngestionError, extract_text, load_documents  # noqa: E402
from complaint_processor.llm import get_llm  # noqa: E402
from complaint_processor.logging_setup import setup_logging  # noqa: E402
from complaint_processor.pipeline import CaseProcessingPipeline  # noqa: E402
from complaint_processor.schemas import DocumentResult, ProcessingStatus  # noqa: E402
from complaint_processor.writers import OutputWriter  # noqa: E402

logger = logging.getLogger("complaint_processor.app")

GITHUB_URL = "https://github.com/saurabhgoyal795/Capstone-Project"
SAMPLE_DIR = ROOT / "data"
ALLOWED_TYPES = ("txt", "pdf", "docx")
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB per file
MAX_RUNS_PER_SESSION = 10
MAX_CONCURRENT_RUNS = 2
SEMAPHORE_WAIT_SECONDS = 20


# =========================================================================== #
# Non-UI processing logic (testable without Streamlit)
# =========================================================================== #
@dataclass
class RunResult:
    """Everything one processing run produced."""

    results: list[DocumentResult]
    ingestion_errors: list[IngestionError]
    output_dir: Path
    report_path: Path
    total_seconds: float
    provider: str = ""
    model: str = ""
    workers: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results) + len(self.ingestion_errors)

    def count(self, status: ProcessingStatus) -> int:
        return sum(1 for r in self.results if r.status == status)

    @property
    def failed(self) -> int:
        return self.count(ProcessingStatus.FAILED) + len(self.ingestion_errors)

    def report_df(self) -> pd.DataFrame:
        """The consolidated report exactly as written to ``final_report.csv``."""
        return pd.read_csv(self.report_path, keep_default_na=False)

    def report_bytes(self) -> bytes:
        return self.report_path.read_bytes()

    def zip_bytes(self) -> bytes:
        """A zip archive of the whole output folder."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.output_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(Path("output") / path.relative_to(self.output_dir)))
        return buffer.getvalue()


def _unique_name(name: str, taken: set[str]) -> str:
    candidate, stem, suffix, i = name, Path(name).stem, Path(name).suffix, 1
    while candidate.lower() in taken:
        candidate = f"{stem}_{i}{suffix}"
        i += 1
    taken.add(candidate.lower())
    return candidate


def run_processing(
    files: Sequence[Path],
    workers: int,
    *,
    settings: Optional[Settings] = None,
    llm: Any = None,
    work_dir: Optional[Path] = None,
) -> RunResult:
    """Copy ``files`` into a fresh run folder, then ingest, process and write outputs.

    Args:
        files: Input documents (txt/pdf/docx). They are copied, never modified.
        workers: Number of documents processed concurrently.
        settings: App settings (defaults to environment-based ``Settings``).
        llm: Chat model to use; built with ``get_llm(settings)`` when omitted.
        work_dir: Parent folder for the run; a new temp dir is created when omitted.

    Returns:
        A :class:`RunResult`. Raises only on LLM initialisation / infrastructure errors.
    """
    settings = settings or get_settings()
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(work_dir) if work_dir else None))
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    input_dir.mkdir(parents=True)

    taken: set[str] = set()
    for src in files:
        src = Path(src)
        shutil.copy2(src, input_dir / _unique_name(src.name, taken))

    start = time.perf_counter()
    documents, ingestion_errors = load_documents(
        input_dir, settings.supported_extensions, settings.max_document_chars
    )
    if llm is None:
        llm = get_llm(settings)

    writer = OutputWriter(output_dir)
    writer.prepare()
    results = CaseProcessingPipeline(llm, settings).run_batch(documents, max_workers=workers)
    writer.write_results(results)
    total_seconds = time.perf_counter() - start
    report_path = writer.write_report(results, ingestion_errors)
    writer.write_run_summary(
        results, ingestion_errors, total_seconds=total_seconds,
        provider=settings.llm_provider, model=settings.resolved_model, max_workers=workers,
        extra={"source": "streamlit"},
    )
    logger.info("Web run finished: %d file(s) in %.1fs (%s)", len(files), total_seconds, run_dir)
    return RunResult(
        results=results, ingestion_errors=list(ingestion_errors), output_dir=output_dir,
        report_path=report_path, total_seconds=total_seconds, provider=settings.llm_provider,
        model=settings.resolved_model, workers=workers,
    )


def validate_uploads(uploads: Sequence[Any]) -> tuple[list[Any], list[str]]:
    """Return (accepted uploads, error messages) applying count/type/size limits."""
    errors: list[str] = []
    accepted: list[Any] = []
    if len(uploads) > MAX_UPLOAD_FILES:
        errors.append(f"Too many files: {len(uploads)} uploaded, max {MAX_UPLOAD_FILES}. "
                      f"Only the first {MAX_UPLOAD_FILES} will be used.")
        uploads = list(uploads)[:MAX_UPLOAD_FILES]
    for up in uploads:
        ext = Path(up.name).suffix.lower().lstrip(".")
        if ext not in ALLOWED_TYPES:
            errors.append(f"{up.name}: unsupported type '.{ext}' (allowed: {', '.join(ALLOWED_TYPES)}).")
        elif up.size > MAX_UPLOAD_BYTES:
            errors.append(f"{up.name}: {up.size / 1024 / 1024:.1f} MB exceeds the 2 MB limit.")
        elif up.size == 0:
            errors.append(f"{up.name}: file is empty.")
        else:
            accepted.append(up)
    return accepted, errors


def list_sample_files() -> list[Path]:
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLE_DIR.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower().lstrip(".") in ALLOWED_TYPES)


def human_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024:.1f} KB" if num_bytes < 1024 * 1024 else f"{num_bytes / 1024 / 1024:.2f} MB"


# =========================================================================== #
# Streamlit UI
# =========================================================================== #
def main() -> None:  # pragma: no cover - exercised via AppTest / manual runs
    import streamlit as st

    st.set_page_config(page_title="AI Complaint & Case Processor", page_icon="📨", layout="wide")

    @st.cache_resource
    def _init_logging() -> Path:
        return setup_logging(get_settings().log_level, PROJECT_ROOT / "logs")

    @st.cache_resource
    def _run_semaphore() -> threading.Semaphore:
        return threading.Semaphore(MAX_CONCURRENT_RUNS)

    @st.cache_resource(show_spinner=False)
    def _cached_llm() -> Any:
        return get_llm(get_settings())

    @st.cache_data(show_spinner=False, max_entries=64)
    def _preview_text(path_str: str, mtime: float) -> tuple[bool, str]:
        try:
            return True, extract_text(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    _init_logging()
    settings = get_settings()
    state = st.session_state
    state.setdefault("run_count", 0)
    state.setdefault("run_result", None)
    if "upload_dir" not in state:
        state["upload_dir"] = tempfile.mkdtemp(prefix="uploads_")
    if "work_dir" not in state:
        state["work_dir"] = tempfile.mkdtemp(prefix="runs_")

    # ------------------------------------------------------------- header
    st.title("AI Complaint & Case Processor")
    st.markdown(
        "Upload customer complaint documents (TXT, PDF or DOCX) and a GenAI workflow processes each one "
        "in three steps: **1. Extraction** pulls structured case fields (customer, category, escalation, "
        "status ...) from the raw text; then, **in parallel**, **2a. Customer Email** drafts an empathetic "
        "reply and **2b. Internal Case Summary** writes a concise brief for the support team. Documents are "
        "processed concurrently, bad files are isolated instead of stopping the batch, and everything is "
        f"collected into a consolidated CSV report. Source code: [GitHub]({GITHUB_URL})."
    )

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Configuration")
        st.markdown(f"**Provider:** `{settings.llm_provider}`  \n**Model:** `{settings.resolved_model}`")
        if settings.llm_provider == "bedrock":
            st.caption(f"AWS region: {settings.aws_region}")
        st.caption("Provider and model are fixed by the server configuration.")
        workers = st.slider("Parallel workers", min_value=1, max_value=4,
                            value=max(1, min(settings.max_workers, 4)),
                            help="Number of documents processed concurrently.")
        st.divider()
        st.info("The sample set includes **corrupted_007.pdf**, an intentionally broken file. "
                "It demonstrates error handling: it is reported as *failed* while the rest of the "
                "batch completes normally.")
        st.caption(f"Runs used this session: {state['run_count']} / {MAX_RUNS_PER_SESSION}")

    # -------------------------------------------------------------- input
    st.subheader("1. Choose documents")
    mode = st.radio("Input source", ["Use sample documents", "Upload your own"],
                    horizontal=True, label_visibility="collapsed")
    selected: list[Path] = []

    if mode == "Use sample documents":
        samples = list_sample_files()
        if not samples:
            st.warning("No sample documents found in data/.")
        else:
            st.dataframe(
                pd.DataFrame([{"File": p.name, "Type": p.suffix.lstrip(".").upper(),
                               "Size": human_size(p.stat().st_size)} for p in samples]),
                hide_index=True,
            )
            names = [p.name for p in samples]
            chosen = st.multiselect("Documents to process", names, default=names)
            selected = [SAMPLE_DIR / n for n in chosen]
            with st.expander("Preview extracted text"):
                if selected:
                    pick = st.selectbox("Document", [p.name for p in selected])
                    path = SAMPLE_DIR / pick
                    ok, text = _preview_text(str(path), path.stat().st_mtime)
                    if ok:
                        st.text_area("Extracted text", text, height=300, disabled=True)
                    else:
                        st.error(f"Could not extract text: {text}")
                else:
                    st.caption("Select at least one document to preview.")
    else:
        uploads = st.file_uploader(
            f"Upload up to {MAX_UPLOAD_FILES} files (TXT, PDF, DOCX; max 2 MB each)",
            type=list(ALLOWED_TYPES), accept_multiple_files=True,
        ) or []
        accepted, upload_errors = validate_uploads(uploads)
        for msg in upload_errors:
            st.error(msg)
        upload_dir = Path(state["upload_dir"])
        taken: set[str] = set()
        for up in accepted:
            safe = _unique_name(Path(up.name).name.replace("\x00", "")[:120] or "upload", taken)
            dest = upload_dir / safe
            dest.write_bytes(up.getbuffer())
            selected.append(dest)
        if selected:
            st.success(f"{len(selected)} file(s) ready: " + ", ".join(p.name for p in selected))

    # ------------------------------------------------------------ process
    st.subheader("2. Process")
    runs_left = MAX_RUNS_PER_SESSION - state["run_count"]
    clicked = st.button("Process documents", type="primary",
                        disabled=not selected or runs_left <= 0)
    if runs_left <= 0:
        st.warning(f"Session limit reached ({MAX_RUNS_PER_SESSION} runs). Reload later to start a new session.")

    if clicked and selected and runs_left > 0:
        semaphore = _run_semaphore()
        if not semaphore.acquire(timeout=SEMAPHORE_WAIT_SECONDS):
            st.warning("The demo server is busy with other runs. Please try again in a moment.")
        else:
            try:
                state["run_count"] += 1
                # Drop the previous run's temp folder to keep disk usage bounded.
                previous: Optional[RunResult] = state.get("run_result")
                if previous is not None:
                    shutil.rmtree(previous.output_dir.parent, ignore_errors=True)
                    state["run_result"] = None
                with st.status(f"Processing {len(selected)} document(s) with {workers} worker(s)...",
                               expanded=True) as status:
                    st.write("Initialising model...")
                    llm = _cached_llm()
                    st.write("Extracting text, then running extraction → email + summary for each document...")
                    result = run_processing(selected, workers, settings=settings, llm=llm,
                                            work_dir=Path(state["work_dir"]))
                    status.update(label=f"Done in {result.total_seconds:.1f}s", state="complete",
                                  expanded=False)
                state["run_result"] = result
            except Exception as exc:  # noqa: BLE001 - never show a traceback to public users
                logger.exception("Web run failed")
                st.error(f"Processing failed: the AI service could not complete the request "
                         f"({type(exc).__name__}). Please try again shortly.")
            finally:
                semaphore.release()

    # ------------------------------------------------------------ results
    result: Optional[RunResult] = state.get("run_result")
    if result is None:
        return

    st.subheader("3. Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total files", result.total)
    c2.metric("Success", result.count(ProcessingStatus.SUCCESS))
    c3.metric("Partial", result.count(ProcessingStatus.PARTIAL))
    c4.metric("Failed", result.failed)
    c5.metric("Total time", f"{result.total_seconds:.1f}s")

    st.markdown("**Consolidated report** (`final_report.csv`)")
    st.dataframe(result.report_df(), hide_index=True)

    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Download final_report.csv", result.report_bytes(), file_name="final_report.csv",
                       mime="text/csv")
    d2.download_button("Download all outputs (.zip)", result.zip_bytes(), file_name="case_outputs.zip",
                       mime="application/zip")

    for err in result.ingestion_errors:
        st.error(f"**{err.file_name}**: could not be read. {err.error}")

    status_icon = {ProcessingStatus.SUCCESS: "✅", ProcessingStatus.PARTIAL: "⚠️", ProcessingStatus.FAILED: "❌"}
    for res in result.results:
        with st.expander(f"{status_icon[res.status]} {res.file_name} ({res.status.value}, "
                         f"{res.duration_seconds:.1f}s)"):
            for msg in res.errors:
                st.error(msg)
            tab_data, tab_email, tab_summary = st.tabs(["Structured data", "Customer email", "Case summary"])
            with tab_data:
                if res.extraction is None:
                    st.info("No structured data (extraction failed).")
                else:
                    data = res.extraction.model_dump(mode="json")
                    left, right = st.columns(2)
                    for i, (key, value) in enumerate(data.items()):
                        (left if i % 2 == 0 else right).markdown(
                            f"**{key.replace('_', ' ').title()}**  \n{value if value not in (None, '') else '—'}")
                    st.json(data, expanded=False)
            with tab_email:
                if res.email is None:
                    st.info("No customer email generated.")
                else:
                    st.markdown(f"**Subject:** {res.email.subject}")
                    st.text_area("Body", res.email.body, height=320, disabled=True,
                                 key=f"email_{res.doc_id}")
            with tab_summary:
                if res.summary is None:
                    st.info("No case summary generated.")
                else:
                    s = res.summary
                    st.markdown(
                        f"#### Case Overview\n{s.case_overview}\n\n#### Key Issue\n{s.key_issue}\n\n"
                        f"#### Action Taken\n{s.action_taken}\n\n#### Current Status\n{s.current_status}\n\n"
                        f"#### Recommended Next Action\n{s.recommended_next_action}"
                    )


if __name__ == "__main__":
    main()
