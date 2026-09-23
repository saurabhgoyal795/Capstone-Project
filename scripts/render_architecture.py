"""Render the system architecture diagram to ``docs/architecture.png``.

Usage::

    .venv/bin/python scripts/render_architecture.py [output_path]

Requires matplotlib (listed in requirements.txt). The diagram is drawn with plain
matplotlib patches so it has no other dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Palette ------------------------------------------------------------------
INK = "#1F2933"
MUTED = "#52606D"
EDGE = "#9AA5B1"
C_INPUT = ("#E3F2FD", "#1565C0")
C_INGEST = ("#E8F5E9", "#2E7D32")
C_WORKER = ("#FAFAFA", "#7B8794")
C_STEP1 = ("#FFF3E0", "#E65100")
C_STEP2 = ("#F3E5F5", "#6A1B9A")
C_OUTPUT = ("#E0F7FA", "#00838F")
C_SIDE = ("#F5F7FA", "#616E7C")
C_FAIL = ("#FFEBEE", "#C62828")


def box(ax, x, y, w, h, title, lines=(), colors=C_SIDE, title_size=11.5, line_size=9.0,
        dashed=False, radius=0.12, title_pad=0.28, align="center"):
    """Draw a rounded box with a bold title and optional body lines (top-aligned)."""
    face, edge = colors
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.6, edgecolor=edge, facecolor=face,
        linestyle=(0, (5, 3)) if dashed else "solid", zorder=2,
    ))
    tx = x + w / 2 if align == "center" else x + 0.15
    ax.text(tx, y + h - title_pad, title, ha=align, va="center", fontsize=title_size,
            fontweight="bold", color=edge, zorder=3)
    for i, line in enumerate(lines):
        ax.text(tx, y + h - title_pad - 0.36 - i * 0.29, line, ha=align, va="center",
                fontsize=line_size, color=INK, zorder=3)


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.8, dashed=False, rad=0.0, label=None,
          label_xy=None, label_color=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16, linewidth=lw, color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=2, shrinkB=2,
    ))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.16)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.5, color=label_color or color,
                style="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def render(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title --------------------------------------------------------------
    ax.text(7, 8.68, "AI Customer Complaint & Case Processing System", ha="center", va="center",
            fontsize=17, fontweight="bold", color=INK)
    ax.text(7, 8.33, "LangChain + Pydantic batch pipeline  |  architecture overview",
            ha="center", va="center", fontsize=10.5, color=MUTED)

    # Side boxes (top row) -------------------------------------------------
    box(ax, 0.35, 6.75, 3.3, 1.2, "config.py  /  .env",
        ["Settings dataclass (python-dotenv)", "provider, model, workers, paths, limits"], C_SIDE)
    box(ax, 5.35, 6.75, 3.9, 1.2, "LLM factory  (llm.py)",
        ["get_llm(): OpenAI | Gemini | Ollama", "temperature 0.1, timeout, retries"], C_SIDE)
    box(ax, 10.65, 6.75, 3.1, 1.2, "logging_setup.py",
        ["console + rotating logs/app.log", "per-thread, per-step timings"], C_SIDE)

    # Input ---------------------------------------------------------------
    box(ax, 0.35, 3.1, 2.0, 2.6, "data/",
        ["complaint_*.pdf", "complaint_*.txt", "complaint_*.docx", "", "7 sample files", "(1 corrupted)"],
        C_INPUT)

    # Ingestion -----------------------------------------------------------
    box(ax, 2.85, 2.75, 2.3, 3.2, "Ingestion",
        ["ingestion.py", "", "pypdf  (.pdf)", "python-docx  (.docx)", "utf-8 / latin-1  (.txt)",
         "normalise whitespace", "truncate to", "MAX_DOCUMENT_CHARS"], C_INGEST)
    box(ax, 2.75, 0.6, 2.5, 1.5, "IngestionError",
        ["file skipped, batch continues;", "reported as FAILED in", "final_report.csv + run_summary"],
        C_FAIL, title_size=10.5, line_size=8.2)
    arrow(ax, (4.0, 2.73), (4.0, 2.12), color=C_FAIL[1], dashed=True)
    ax.text(4.12, 2.43, "unreadable file", ha="left", va="center", fontsize=8, color=C_FAIL[1], style="italic")

    arrow(ax, (2.37, 4.4), (2.83, 4.4))

    # Worker container ----------------------------------------------------
    box(ax, 5.8, 0.55, 4.9, 5.8, "Per-document worker  (x N threads)", [], C_WORKER, dashed=True, title_size=11)
    ax.text(8.25, 5.76, "ThreadPoolExecutor - MAX_WORKERS docs in parallel (default 3)",
            ha="center", va="center", fontsize=8.3, color=MUTED, style="italic")
    arrow(ax, (5.17, 4.4), (6.08, 4.4))
    ax.text(5.47, 4.72, "Loaded-\nDocument", ha="center", va="center", fontsize=7.5, color=MUTED, style="italic")

    # Step 1
    box(ax, 6.1, 3.55, 4.4, 1.95, "Step 1  -  Structured Extraction",
        ["Pydantic ComplaintExtraction (11 fields, enums)",
         "native: llm.with_structured_output(...)",
         "fallback: format instructions -> JSON repair",
         "-> validate -> 1 retry with error feedback"], C_STEP1, title_size=11, line_size=8.7)

    # Step 2
    box(ax, 6.1, 0.8, 4.4, 2.25, "Step 2  -  RunnableParallel", [], C_STEP2, title_size=11)
    box(ax, 6.3, 0.95, 1.95, 1.55, "Customer Email",
        ["-> CustomerEmail", "subject + body", "+ guardrail checks"],
        C_STEP2, title_size=9.5, line_size=8.0, title_pad=0.22, radius=0.08)
    box(ax, 8.35, 0.95, 1.95, 1.55, "Case Summary",
        ["-> CaseSummary", "5 internal sections", "(incl. next action)"],
        C_STEP2, title_size=9.5, line_size=8.0, title_pad=0.22, radius=0.08)
    arrow(ax, (8.3, 3.53), (8.3, 3.07), label="extraction JSON + text", label_xy=(9.35, 3.3))

    # Failure branch from step 1
    ax.text(6.15, 3.3, "fail -> FAILED\n(skip step 2)", ha="left", va="center", fontsize=8,
            color=C_FAIL[1], style="italic")

    # Output writer -------------------------------------------------------
    box(ax, 11.45, 4.35, 2.3, 1.75, "OutputWriter",
        ["writers.py", "SUCCESS / PARTIAL /", "FAILED per document"], C_OUTPUT)
    arrow(ax, (10.72, 5.0), (11.43, 5.0))
    ax.text(11.07, 5.32, "Document-\nResult", ha="center", va="center", fontsize=7.5, color=MUTED, style="italic")

    outputs = [
        ("structured_data/*.json", 3.55),
        ("customer_emails/*.txt", 2.95),
        ("case_summaries/*.md", 2.35),
        ("final_report.csv", 1.75),
        ("run_summary.json", 1.15),
    ]
    ax.text(11.5, 4.05, "output/", ha="left", va="center", fontsize=11, fontweight="bold",
            color=C_OUTPUT[1])
    for label, y in outputs:
        box(ax, 11.45, y - 0.22, 2.3, 0.44, label, [], C_OUTPUT, title_size=8.6, title_pad=0.22,
            radius=0.06)
    arrow(ax, (13.0, 4.33), (13.0, 3.82))

    # Side-box connectors (dashed) ---------------------------------------
    arrow(ax, (3.67, 7.35), (5.33, 7.35), color=EDGE, dashed=True)
    arrow(ax, (7.3, 6.73), (7.3, 6.38), color=EDGE, dashed=True)
    ax.text(7.45, 6.55, "injected BaseChatModel", ha="left", va="center", fontsize=8, color=MUTED, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "docs" / "architecture.png"
    print(f"wrote {render(target)}")
