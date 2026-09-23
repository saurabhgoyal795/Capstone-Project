"""Output writers: per-document JSON / email / summary files, CSV report and run summary."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

import pandas as pd

from complaint_processor.schemas import CaseSummary, DocumentResult, ProcessingStatus

logger = logging.getLogger(__name__)

STRUCTURED_DIR = "structured_data"
EMAILS_DIR = "customer_emails"
SUMMARIES_DIR = "case_summaries"
REPORT_FILE = "final_report.csv"
RUN_SUMMARY_FILE = "run_summary.json"

REPORT_COLUMNS = [
    "doc_id",
    "file_name",
    "status",
    "customer_name",
    "email",
    "phone_number",
    "complaint_category",
    "is_complaint",
    "escalation_required",
    "supporting_document_available",
    "overall_case_status",
    "product_or_service",
    "issue_description",
    "resolution_provided",
    "email_subject",
    "duration_seconds",
    "errors",
]


class IngestionErrorLike(Protocol):
    """Duck-typed view of ``ingestion.IngestionError``."""

    file_name: str
    path: Path
    error: str


class OutputWriter:
    """Writes all pipeline artefacts under a single output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.structured_dir = self.output_dir / STRUCTURED_DIR
        self.emails_dir = self.output_dir / EMAILS_DIR
        self.summaries_dir = self.output_dir / SUMMARIES_DIR

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        """Create the output sub-directories and clear files left by a previous run.

        Only files inside the three managed sub-directories are removed.
        """
        for directory in (self.structured_dir, self.emails_dir, self.summaries_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
        logger.debug("Prepared output directory %s", self.output_dir)

    # --------------------------------------------------------- per document
    def write_result(self, result: DocumentResult) -> list[Path]:
        """Write the JSON, email and summary files for one result. Returns written paths."""
        written: list[Path] = []

        json_path = self.structured_dir / f"{result.doc_id}.json"
        json_path.write_text(
            result.model_dump_json(indent=2, exclude={"email", "summary"}), encoding="utf-8"
        )
        written.append(json_path)

        if result.email is not None:
            email_path = self.emails_dir / f"{result.doc_id}_email.txt"
            email_path.write_text(
                f"Subject: {result.email.subject.strip()}\n\n{result.email.body.strip()}\n", encoding="utf-8"
            )
            written.append(email_path)

        if result.summary is not None:
            summary_path = self.summaries_dir / f"{result.doc_id}_summary.md"
            summary_path.write_text(self._render_summary_md(result, result.summary), encoding="utf-8")
            written.append(summary_path)

        return written

    def write_results(self, results: Iterable[DocumentResult]) -> None:
        """Write files for every result, logging (not raising) on I/O errors."""
        for result in results:
            try:
                self.write_result(result)
            except OSError as exc:
                logger.error("Failed to write outputs for %s: %s", result.doc_id, exc)
                result.errors.append(f"write failed: {exc}")

    @staticmethod
    def _render_summary_md(result: DocumentResult, summary: CaseSummary) -> str:
        ext = result.extraction
        lines = [f"# Case Summary: {result.doc_id}", ""]
        if ext is not None:
            lines += [
                "| Field | Value |",
                "|---|---|",
                f"| Source file | {result.file_name} |",
                f"| Customer | {ext.customer_name or 'N/A'} |",
                f"| Category | {ext.complaint_category.value} |",
                f"| Product / Service | {ext.product_or_service or 'N/A'} |",
                f"| Escalation required | {ext.escalation_required.value} |",
                f"| Case status | {ext.overall_case_status.value} |",
                "",
            ]
        sections = [
            ("Case Overview", summary.case_overview),
            ("Key Issue", summary.key_issue),
            ("Action Taken", summary.action_taken),
            ("Current Status", summary.current_status),
            ("Recommended Next Action", summary.recommended_next_action),
        ]
        for heading, body in sections:
            lines += [f"## {heading}", "", body.strip(), ""]
        return "\n".join(lines)

    # ---------------------------------------------------------------- report
    @staticmethod
    def _result_row(result: DocumentResult) -> dict[str, Any]:
        ext = result.extraction
        row: dict[str, Any] = {col: "" for col in REPORT_COLUMNS}
        row.update(
            doc_id=result.doc_id,
            file_name=result.file_name,
            status=result.status.value,
            email_subject=result.email.subject if result.email else "",
            duration_seconds=round(result.duration_seconds, 2),
            errors=" | ".join(result.errors),
        )
        if ext is not None:
            row.update(
                customer_name=ext.customer_name or "",
                email=ext.email or "",
                phone_number=ext.phone_number or "",
                complaint_category=ext.complaint_category.value,
                is_complaint=ext.is_complaint.value,
                escalation_required=ext.escalation_required.value,
                supporting_document_available=ext.supporting_document_available.value,
                overall_case_status=ext.overall_case_status.value,
                product_or_service=ext.product_or_service or "",
                issue_description=ext.issue_description,
                resolution_provided=ext.resolution_provided or "",
            )
        return row

    @staticmethod
    def _ingestion_error_row(err: IngestionErrorLike) -> dict[str, Any]:
        row: dict[str, Any] = {col: "" for col in REPORT_COLUMNS}
        row.update(
            doc_id=Path(err.file_name).stem,
            file_name=err.file_name,
            status=ProcessingStatus.FAILED.value,
            duration_seconds=0.0,
            errors=f"ingestion failed: {err.error}",
        )
        return row

    def write_report(
        self,
        results: Sequence[DocumentResult],
        ingestion_errors: Sequence[IngestionErrorLike] = (),
    ) -> Path:
        """Write the consolidated CSV (one row per input file, sorted by file name)."""
        rows = [self._result_row(r) for r in results] + [self._ingestion_error_row(e) for e in ingestion_errors]
        df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
        if not df.empty:
            df = df.sort_values("file_name", kind="stable").reset_index(drop=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / REPORT_FILE
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Wrote consolidated report: %s (%d rows)", path, len(df))
        return path

    def write_run_summary(
        self,
        results: Sequence[DocumentResult],
        ingestion_errors: Sequence[IngestionErrorLike] = (),
        total_seconds: float = 0.0,
        provider: str = "",
        model: str = "",
        max_workers: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write ``run_summary.json`` with counts, timing and model information."""
        counts = {s.value: sum(1 for r in results if r.status == s) for s in ProcessingStatus}
        summary: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": provider,
            "model": model,
            "max_workers": max_workers,
            "total_files": len(results) + len(ingestion_errors),
            "documents_processed": len(results),
            "ingestion_failures": len(ingestion_errors),
            "success": counts[ProcessingStatus.SUCCESS.value],
            "partial": counts[ProcessingStatus.PARTIAL.value],
            "failed": counts[ProcessingStatus.FAILED.value] + len(ingestion_errors),
            "total_seconds": round(total_seconds, 2),
            "avg_seconds_per_document": round(
                sum(r.duration_seconds for r in results) / len(results), 2
            ) if results else 0.0,
            "ingestion_errors": [{"file_name": e.file_name, "error": e.error} for e in ingestion_errors],
        }
        if extra:
            summary.update(extra)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / RUN_SUMMARY_FILE
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote run summary: %s", path)
        return path
