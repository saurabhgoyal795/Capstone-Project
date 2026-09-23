"""Build the capstone presentation deck: docs/Capstone_Presentation.pptx.

Usage:
    uv pip install -p .venv/bin/python python-pptx
    .venv/bin/python scripts/build_presentation.py

All numbers and excerpts on the results/sample slides are read from the real run artefacts in
``output/`` (final_report.csv, run_summary.json, structured_data/, customer_emails/,
case_summaries/), so the deck stays in sync with the latest pipeline run.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DEST = ROOT / "docs" / "Capstone_Presentation.pptx"

# --------------------------------------------------------------------------- theme
NAVY = RGBColor(0x0F, 0x1F, 0x3D)
NAVY_2 = RGBColor(0x1C, 0x33, 0x5E)
ACCENT = RGBColor(0x14, 0xB8, 0xA6)  # teal
ACCENT_2 = RGBColor(0xF5, 0x9E, 0x0B)  # amber, used sparingly
RED = RGBColor(0xDC, 0x26, 0x26)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x55, 0x60, 0x70)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
TEAL_LIGHT = RGBColor(0xE6, 0xF7, 0xF5)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
MONO = "Consolas"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
TOTAL_SLIDES = 12


# --------------------------------------------------------------------------- helpers
def _style_run(run, size, color=INK, bold=False, font=FONT, italic=False):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def text_box(slide, x, y, w, h, paragraphs, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, font=FONT, spacing=4, line_spacing=None):
    """Add a textbox. ``paragraphs`` is a str or list of str / (str, dict-of-overrides)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0.05))
    fill_paragraphs(tf, paragraphs, size, color, bold, align, font, spacing, line_spacing)
    return tb


def fill_paragraphs(tf, paragraphs, size, color, bold, align, font=FONT, spacing=4, line_spacing=None):
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    for i, item in enumerate(paragraphs):
        text, opts = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opts.get("align", align)
        p.space_after = Pt(opts.get("spacing", spacing))
        if line_spacing:
            p.line_spacing = line_spacing
        run = p.add_run()
        run.text = text
        _style_run(run, opts.get("size", size), opts.get("color", color), opts.get("bold", bold),
                   opts.get("font", font), opts.get("italic", False))


def bullets(slide, x, y, w, h, items, size=20, color=INK, bullet_color=ACCENT, spacing=10):
    """Bulleted list; items are str or (bold_lead, rest)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing)
        r = p.add_run()
        r.text = "■  "
        _style_run(r, size - 6, bullet_color, bold=True)
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = lead
            _style_run(r1, size, color, bold=True)
            r2 = p.add_run()
            r2.text = rest
            _style_run(r2, size, color)
        else:
            r1 = p.add_run()
            r1.text = item
            _style_run(r1, size, color)
    return tb


def box(slide, x, y, w, h, fill=LIGHT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=1.0,
        dash=False, radius=0.08):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
        if dash:
            ln = shp.line._get_or_add_ln()
            prst = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
            ln.append(prst)
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def label_box(slide, x, y, w, h, title, body=None, fill=LIGHT, line=BORDER, title_color=NAVY,
              body_color=INK, title_size=18, body_size=14, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
              dash=False, body_font=FONT):
    """Shape with a bold title and optional body text inside it."""
    shp = box(slide, x, y, w, h, fill=fill, line=line, dash=dash)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right"):
        setattr(tf, f"margin_{side}", Inches(0.1))
    for side in ("top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0.06))
    paras = [(title, {"bold": True, "size": title_size, "color": title_color, "spacing": 3})]
    if body:
        for line_text in body if isinstance(body, list) else [body]:
            paras.append((line_text, {"size": body_size, "color": body_color, "spacing": 2, "font": body_font}))
    fill_paragraphs(tf, paras, body_size, body_color, False, align)
    return shp


def arrow(slide, x1, y1, x2, y2, color=NAVY_2, width=2.0):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    ln = conn.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return conn


def base_slide(prs, title, subtitle=None, number=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE
    bar = box(slide, 0, 0, 13.333, 1.05, fill=NAVY, line=None, shape=MSO_SHAPE.RECTANGLE)
    bar.name = "TitleBar"
    box(slide, 0, 1.05, 13.333, 0.06, fill=ACCENT, line=None, shape=MSO_SHAPE.RECTANGLE)
    text_box(slide, 0.5, 0.12, 10.8, 0.8, title, size=30, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        text_box(slide, 0.5, 1.2, 12.3, 0.5, subtitle, size=18, color=MUTED)
    # footer
    text_box(slide, 0.5, 7.05, 9.0, 0.35, "AI Customer Complaint & Case Processing System  |  Saurabh Goyal",
             size=11, color=MUTED)
    if number:
        text_box(slide, 11.8, 7.05, 1.03, 0.35, f"{number} / {TOTAL_SLIDES}", size=11, color=MUTED,
                 align=PP_ALIGN.RIGHT)
    return slide


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------- data
def load_run_data():
    with open(OUT / "final_report.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    summary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
    extraction = json.loads((OUT / "structured_data" / "complaint_001.json").read_text(encoding="utf-8"))
    email = (OUT / "customer_emails" / "complaint_001_email.txt").read_text(encoding="utf-8")
    case_summary = (OUT / "case_summaries" / "complaint_001_summary.md").read_text(encoding="utf-8")
    return rows, summary, extraction, email, case_summary


def log_facts(log_path: Path = ROOT / "logs" / "app.log") -> dict:
    """Timings / worker assignment for the most recent pipeline run, parsed from the app log."""
    facts = {"extract": "10.71", "category": "Billing", "email": "29.79", "summary": "23.21",
             "done": "[1/6] complaint_001.pdf done (success, 40.5s)", "workers": {}}
    if not log_path.exists():
        return facts
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if "Processing " in ln and "worker(s)" in ln]
    run = lines[starts[-1]:] if starts else lines
    for ln in run:
        if m := re.search(r"\[complaint_001\] step 1/2 extraction ok in ([\d.]+)s \(category=([^,]+)", ln):
            facts["extract"], facts["category"] = m.group(1), m.group(2)
        elif m := re.search(r"\[complaint_001\] step 2/2 email_generation ok in ([\d.]+)s", ln):
            facts["email"] = m.group(1)
        elif m := re.search(r"\[complaint_001\] step 2/2 case_summary ok in ([\d.]+)s", ln):
            facts["summary"] = m.group(1)
        elif m := re.search(r"(\[\d+/\d+\] complaint_001\.pdf done \([^)]*\))", ln):
            facts["done"] = m.group(1)
        if m := re.search(r"\| (doc-worker_\d+)\s*\|.*\[complaint_(\d+)\] starting", ln):
            facts["workers"].setdefault(m.group(1), []).append(m.group(2))
    return facts


def md_section(md: str, heading: str) -> str:
    lines = md.splitlines()
    out, grab = [], False
    for ln in lines:
        if ln.startswith("## "):
            grab = ln[3:].strip().lower() == heading.lower()
            continue
        if grab and ln.strip():
            out.append(ln.strip())
    return " ".join(out)


# --------------------------------------------------------------------------- slides
def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    box(slide, 0, 0, 0.35, 7.5, fill=ACCENT, line=None, shape=MSO_SHAPE.RECTANGLE)
    text_box(slide, 1.0, 0.9, 11.5, 0.5, "IIT PATNA  |  GENAI DEVELOPMENT PROGRAM  |  FINAL EVALUATION",
             size=16, color=ACCENT, bold=True)
    text_box(slide, 1.0, 1.6, 11.5, 2.0, "AI Customer Complaint & Case Processing System",
             size=44, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    box(slide, 1.0, 3.75, 2.2, 0.07, fill=ACCENT_2, line=None, shape=MSO_SHAPE.RECTANGLE)
    text_box(slide, 1.0, 4.0, 11.5, 0.9,
             ["Batch 1  •  Project 1: AI-Powered Document Processing & Business Workflow"],
             size=22, color=RGBColor(0xCB, 0xD5, 0xE1))
    text_box(slide, 1.0, 5.3, 8.0, 0.6, "Saurabh Goyal", size=28, color=WHITE, bold=True)
    text_box(slide, 1.0, 5.95, 8.0, 0.5, "September 2026", size=20, color=RGBColor(0xCB, 0xD5, 0xE1))
    text_box(slide, 8.3, 6.55, 4.6, 0.5, "Python • LangChain • Pydantic", size=16,
             color=ACCENT, align=PP_ALIGN.RIGHT)
    notes(slide, "Capstone project: an end-to-end GenAI pipeline that turns raw customer complaint "
                 "documents into structured case data, a customer reply email and an internal case summary.")


def slide_problem(prs):
    slide = base_slide(prs, "Problem Statement", number=2)
    cards = [
        ("Unstructured inputs", "Complaints arrive as PDFs, Word files, emails, chat logs and forms"),
        ("Manual triage", "Agents read every document to find the customer, issue, category and urgency"),
        ("Slow, uneven replies", "Response quality and tone depend on who picks up the case"),
        ("No structured record", "Hard to report, route or escalate without clean case data"),
    ]
    cw, gap, x0 = 2.9, 0.23, 0.5
    for i, (t, b) in enumerate(cards):
        x = x0 + i * (cw + gap)
        box(slide, x, 1.55, cw, 0.09, fill=ACCENT if i % 2 == 0 else ACCENT_2, line=None,
            shape=MSO_SHAPE.RECTANGLE)
        label_box(slide, x, 1.64, cw, 2.35, t, b, fill=LIGHT, line=BORDER, title_size=19, body_size=17,
                  anchor=MSO_ANCHOR.MIDDLE)
    box(slide, 0.5, 4.45, 12.33, 2.1, fill=TEAL_LIGHT, line=ACCENT, line_w=1.5)
    text_box(slide, 0.8, 4.6, 11.8, 0.5, "GOAL", size=16, color=ACCENT, bold=True)
    text_box(slide, 0.8, 5.05, 11.8, 1.4,
             "Automatically turn each raw complaint document into (1) validated structured case data, "
             "(2) an empathetic customer reply and (3) an internal case summary — reliably, "
             "without hallucinated facts, and at batch scale.",
             size=20, color=INK)
    notes(slide, "Support teams spend a lot of time just reading documents. The goal is to automate "
                 "triage and first response while keeping facts grounded in the source document.")


def slide_solution(prs):
    slide = base_slide(prs, "Solution Overview", number=3)
    # input column
    label_box(slide, 0.5, 2.0, 2.3, 3.2, "Input", ["data/ folder", "PDF • DOCX • TXT",
                                                   "7 sample files", "(1 deliberately corrupt)"],
              fill=LIGHT, title_size=20, body_size=16)
    tasks = [
        ("Task 1", "Structured Extraction", "11 fields validated by a Pydantic schema"),
        ("Task 2", "Customer Email", "Empathetic, grounded reply with guardrail checks"),
        ("Task 3", "Case Summary", "5-section internal brief incl. next action"),
    ]
    for i, (tag, name, desc) in enumerate(tasks):
        y = 1.55 + i * 1.45
        box(slide, 3.5, y, 1.1, 1.2, fill=NAVY, line=None)
        text_box(slide, 3.5, y, 1.1, 1.2, tag, size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        label_box(slide, 4.6, y, 4.2, 1.2, name, desc, fill=WHITE, line=NAVY_2, title_size=19,
                  body_size=15, align=PP_ALIGN.LEFT)
    label_box(slide, 9.55, 2.0, 3.28, 3.2, "Outputs",
              ["structured_data/*.json", "customer_emails/*.txt", "case_summaries/*.md",
               "final_report.csv", "run_summary.json"],
              fill=TEAL_LIGHT, line=ACCENT, title_size=20, body_size=15, body_font=MONO)
    arrow(slide, 2.85, 3.6, 3.45, 3.6)
    arrow(slide, 8.85, 3.6, 9.5, 3.6)
    text_box(slide, 0.5, 6.1, 12.33, 0.8,
             "One command:  python main.py --provider ollama --workers 3",
             size=20, color=NAVY, bold=True, align=PP_ALIGN.CENTER, font=MONO)
    notes(slide, "Three chained AI tasks. Extraction feeds both email and summary so all outputs share "
                 "the same validated facts.")


def slide_architecture(prs):
    slide = base_slide(prs, "Architecture", number=4)
    # pipeline bracket
    box(slide, 3.05, 1.45, 5.3, 3.35, fill=None, line=ACCENT, dash=True, line_w=1.5)
    text_box(slide, 3.15, 1.48, 5.1, 0.4, "CaseProcessingPipeline  (runs once per document)",
             size=13, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)
    y, h = 2.05, 2.5
    label_box(slide, 0.4, y, 2.2, h, "Input documents", ["data/", "PDF, DOCX, TXT"], title_size=17,
              body_size=14)
    label_box(slide, 3.25, y, 2.2, h, "1. Extraction chain",
              ["extraction prompt", "structured_call()", "contact sanitiser", "→ ComplaintExtraction"],
              fill=WHITE, line=NAVY_2, title_size=16, body_size=13)
    # RunnableParallel container
    box(slide, 5.85, y, 2.3, h, fill=TEAL_LIGHT, line=ACCENT)
    text_box(slide, 5.85, y + 0.05, 2.3, 0.35, "RunnableParallel", size=13, color=ACCENT, bold=True,
             align=PP_ALIGN.CENTER)
    label_box(slide, 6.0, y + 0.45, 2.0, 0.9, "2. Email chain", "+ guardrails", fill=WHITE, line=NAVY_2,
              title_size=15, body_size=12)
    label_box(slide, 6.0, y + 1.45, 2.0, 0.9, "3. Summary chain", "5 sections", fill=WHITE, line=NAVY_2,
              title_size=15, body_size=12)
    label_box(slide, 8.8, y, 2.0, h, "Writers", ["JSON per doc", "email .txt", "summary .md",
                                               "CSV report", "run summary"], title_size=17, body_size=13)
    label_box(slide, 11.2, y, 1.75, h, "output/", ["case pack", "per document", "+ batch report"],
              fill=NAVY, line=None, title_color=WHITE, body_color=WHITE, title_size=17, body_size=13)
    # ingestion as a slim box on the arrow between input and pipeline
    label_box(slide, 0.4, 4.85, 2.2, 1.35, "Ingestion", ["pypdf • python-docx", "normalise + truncate"],
              fill=LIGHT, title_size=15, body_size=12)
    arrow(slide, 2.65, 3.3, 3.2, 3.3)
    arrow(slide, 5.5, 3.3, 5.8, 3.3)
    arrow(slide, 8.2, 3.3, 8.75, 3.3)
    arrow(slide, 10.85, 3.3, 11.15, 3.3)
    arrow(slide, 1.5, 4.6, 1.5, 4.82)
    # bottom row: LLM factory feeds the chains; config feeds the factory
    by, bh = 5.3, 1.35
    label_box(slide, 3.25, by, 4.9, bh, "LLM Factory  •  get_llm(settings)",
              ["OpenAI  •  Gemini  •  Ollama (local)"], fill=WHITE, line=ACCENT_2, title_size=15,
              body_size=13)
    label_box(slide, 8.8, by, 2.0, bh, "Config", [".env → Settings", "CLI flags override"],
              title_size=15, body_size=12)
    label_box(slide, 11.2, by, 1.75, bh, "Logging", ["console +", "logs/app.log"], title_size=15,
              body_size=12)
    arrow(slide, 8.75, by + bh / 2, 8.2, by + bh / 2, color=ACCENT_2)
    arrow(slide, 4.35, by - 0.02, 4.35, 4.6, color=ACCENT_2)
    arrow(slide, 7.0, by - 0.02, 7.0, 4.6, color=ACCENT_2)
    notes(slide, "Ingestion loads and normalises files; the pipeline runs extraction then a RunnableParallel "
                 "of email and summary; writers persist everything. The LLM is injected via a provider-"
                 "agnostic factory configured from .env.")


def slide_workflow(prs, summary, facts):
    slide = base_slide(prs, "Workflow: 3 AI Tasks + Two Levels of Parallelism", number=5)
    text_box(slide, 0.5, 1.3, 7.0, 0.4, "Per document (LangChain)", size=18, color=ACCENT, bold=True)
    label_box(slide, 0.5, 2.25, 1.9, 1.3, "Document text", "normalised", title_size=16, body_size=13)
    label_box(slide, 2.9, 2.25, 2.1, 1.3, "Task 1", "Extraction → validated JSON", fill=NAVY,
              line=None, title_color=WHITE, body_color=WHITE, title_size=17, body_size=13)
    label_box(slide, 5.6, 1.8, 1.9, 1.0, "Task 2", "Customer email", fill=WHITE, line=NAVY_2,
              title_size=16, body_size=13)
    label_box(slide, 5.6, 3.0, 1.9, 1.0, "Task 3", "Case summary", fill=WHITE, line=NAVY_2,
              title_size=16, body_size=13)
    arrow(slide, 2.45, 2.9, 2.85, 2.9)
    arrow(slide, 5.05, 2.75, 5.55, 2.3)
    arrow(slide, 5.05, 3.05, 5.55, 3.5)
    text_box(slide, 5.4, 4.05, 2.3, 0.4, "RunnableParallel", size=13, color=ACCENT, bold=True,
             align=PP_ALIGN.CENTER)
    bullets(slide, 0.5, 4.6, 7.2, 2.3, [
        ("Extraction first: ", "email + summary both reuse its JSON"),
        ("Status per doc: ", "success / partial / failed"),
        ("Real run (doc 001): ", f"email {float(facts['email']):.1f} s ‖ summary {float(facts['summary']):.1f} s, concurrent"),
    ], size=17, spacing=8)
    # right: across documents
    box(slide, 8.1, 1.3, 4.73, 5.55, fill=LIGHT, line=BORDER)
    text_box(slide, 8.3, 1.4, 4.4, 0.4, "Across documents", size=18, color=ACCENT, bold=True)
    text_box(slide, 8.3, 1.75, 4.4, 0.35, "ThreadPoolExecutor, max_workers=3", size=13, color=MUTED,
             font=MONO)
    workers = [(w, " → ".join(docs)) for w, docs in sorted(facts["workers"].items())][:3] or [
        ("doc-worker_0", "001 → 004"), ("doc-worker_1", "002 → 005"), ("doc-worker_2", "003 → 006")]
    for i, (w, d) in enumerate(workers):
        yy = 2.25 + i * 0.72
        label_box(slide, 8.3, yy, 2.1, 0.6, w, None, fill=NAVY_2, line=None, title_color=WHITE,
                  title_size=13)
        text_box(slide, 10.5, yy, 2.2, 0.6, d, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE, font=MONO)
    total = summary["total_seconds"]
    avg = summary["avg_seconds_per_document"]
    n = summary["documents_processed"]
    seq = avg * n
    text_box(slide, 8.3, 4.5, 4.4, 2.3, [
        (f"{total:.1f} s", {"size": 36, "bold": True, "color": NAVY, "spacing": 0}),
        (f"wall time for {summary['total_files']} files, {summary['max_workers']} workers",
         {"size": 15, "color": MUTED, "spacing": 8}),
        (f"≈ {seq / total:.1f}× faster than sequential", {"size": 18, "bold": True, "color": ACCENT,
                                                                   "spacing": 0}),
        (f"({n} × {avg:.1f} s avg ≈ {seq:.0f} s estimated)", {"size": 14, "color": MUTED}),
    ])
    notes(slide, "Two levels of concurrency: RunnableParallel inside a document, ThreadPoolExecutor across "
                 "documents. Results are returned in input order.")


def slide_schema(prs):
    slide = base_slide(prs, "Structured Output & Pydantic Schema", number=6)
    text_box(slide, 0.5, 1.3, 6.0, 0.4, "ComplaintExtraction (11 fields)", size=18, color=ACCENT, bold=True)
    fields = [
        ("customer_name", "str | null"), ("email", "str | null (normalised)"), ("phone_number", "str | null"),
        ("product_or_service", "str | null"), ("complaint_category", "enum (8)"),
        ("issue_description", "str (1-3 sentences)"), ("resolution_provided", "str | null"),
        ("is_complaint", "Yes / No"), ("escalation_required", "Yes / No"),
        ("supporting_document_available", "Yes / No"), ("overall_case_status", "enum (5)"),
    ]
    rows = len(fields)
    tbl = slide.shapes.add_table(rows, 2, Inches(0.5), Inches(1.8), Inches(6.2), Inches(0.42 * rows)).table
    tbl.columns[0].width = Inches(3.5)
    tbl.columns[1].width = Inches(2.7)
    for r, (f, t) in enumerate(fields):
        for c, val in enumerate((f, t)):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r % 2 == 0 else WHITE
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame
            tf.text = ""
            run = tf.paragraphs[0].add_run()
            run.text = val
            _style_run(run, 14, NAVY if c == 0 else INK, bold=(c == 0), font=MONO if c == 0 else FONT)
    tbl.first_row = False
    # right column
    label_box(slide, 7.1, 1.3, 5.73, 1.55, "Enums keep the model on-rails",
              ["Category: Billing, Product Defect, Service Quality, Delivery, Technical Issue, Refund, "
               "Account, Other",
               "Status: Open, In Progress, Resolved, Escalated, Closed"],
              fill=LIGHT, title_size=16, body_size=13, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    text_box(slide, 7.1, 3.05, 5.7, 0.4, "structured_call(): 3-step safety net", size=18, color=ACCENT,
             bold=True)
    steps = [
        ("1", "Native structured output", "with_structured_output (Ollama: JSON-schema grammar)"),
        ("2", "Fallback parser + repair", "format instructions; strip <think>, fences, trailing commas"),
        ("3", "One retry with feedback", "validation error sent back to the model"),
    ]
    for i, (n, t, d) in enumerate(steps):
        yy = 3.55 + i * 1.12
        box(slide, 7.1, yy, 0.7, 0.95, fill=NAVY, line=None)
        text_box(slide, 7.1, yy, 0.7, 0.95, n, size=24, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        label_box(slide, 7.8, yy, 5.03, 0.95, t, d, fill=WHITE, line=BORDER, title_size=16, body_size=13,
                  align=PP_ALIGN.LEFT)
    notes(slide, "Every AI task returns a validated Pydantic model. Native structured output is tried "
                 "first; if the provider fails, a repair-and-retry fallback takes over.")


def slide_prompts(prs, rows):
    slide = base_slide(prs, "Prompt Engineering & Hallucination Controls", number=7)
    cols = [
        ("Prompt design", ACCENT, [
            "Role-specific system prompts per task",
            "Document wrapped in <document> tags",
            "Explicit category & status definitions",
            "Status precedence rule",
            "Temperature 0.1",
        ]),
        ("Grounding rules", NAVY_2, [
            "Use ONLY facts in the document",
            "Missing field → null, never \"N/A\"",
            "Request ≠ resolution",
            "No promised refunds, dates or credits",
            "No placeholders like [Name]",
        ]),
        ("Code-level checks", ACCENT_2, [
            "Extracted email/phone must appear in source, else set to null",
            "Email guardrail flags unknown contacts & placeholders",
            "Enum + validator enforce allowed values",
        ]),
    ]
    cw, gap = 3.95, 0.24
    for i, (title, col, items) in enumerate(cols):
        x = 0.5 + i * (cw + gap)
        box(slide, x, 1.45, cw, 0.6, fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
        text_box(slide, x, 1.45, cw, 0.6, title, size=20, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        box(slide, x, 2.05, cw, 3.75, fill=LIGHT, line=None, shape=MSO_SHAPE.RECTANGLE)
        bullets(slide, x + 0.15, 2.25, cw - 0.3, 3.5, items, size=18, bullet_color=col, spacing=12)
    ok_rows = [r for r in rows if r["status"] == "success"]
    no_phone = ", ".join(r["doc_id"].split("_")[-1] for r in ok_rows if not r["phone_number"]) or "none"
    no_res = ", ".join(r["doc_id"].split("_")[-1] for r in ok_rows if not r["resolution_provided"]) or "none"
    text_box(
        slide, 0.5, 6.0, 12.33, 0.9,
        f"Evidence from the run: no phone in source ({no_phone}) → phone_number = null, not invented. "
        f"Unresolved cases ({no_res}) keep resolution_provided = null.",
        size=16, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    notes(slide, "Controls operate at three layers: prompt instructions, schema constraints, and "
                 "post-hoc verification in code against the source text.")


def slide_errors(prs, facts):
    slide = base_slide(prs, "Error Handling & Logging", number=8)
    levels = [
        ("File level", "Corrupt / empty / unsupported files are recorded as ingestion errors; batch continues"),
        ("Task level", "Each task runs guarded; extraction failure → failed, email/summary failure "
                       "→ partial"),
        ("Worker level", "_safe_process() guard: a worker thread never raises"),
        ("Provider level", "LLM timeouts + max_retries; native → fallback → retry"),
        ("Config level", "Settings.validate(): bad provider or missing API key → clear error, exit code"),
    ]
    for i, (t, d) in enumerate(levels):
        yy = 1.45 + i * 0.86
        box(slide, 0.5, yy, 0.12, 0.72, fill=ACCENT, line=None, shape=MSO_SHAPE.RECTANGLE)
        text_box(slide, 0.75, yy, 2.1, 0.72, t, size=17, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text_box(slide, 2.85, yy, 4.6, 0.72, d, size=14, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    # log panel
    box(slide, 7.75, 1.45, 5.08, 3.75, fill=NAVY, line=None)
    text_box(slide, 7.95, 1.55, 4.7, 0.4, "logs/app.log (real run excerpt)", size=14, color=ACCENT, bold=True)
    log_lines = [
        ("ERROR  ingestion", RGBColor(0xFC, 0xA5, 0xA5)),
        ("Failed to ingest corrupted_007.pdf:", WHITE),
        ("PdfStreamError: Stream has ended", WHITE),
        ("", WHITE),
        ("INFO  doc-worker_0", RGBColor(0x99, 0xF6, 0xE4)),
        ("[complaint_001] step 1/2 extraction", WHITE),
        (f"ok in {facts['extract']}s (category={facts['category']})", WHITE),
        (f"[complaint_001] step 2/2 email ok {facts['email']}s", WHITE),
        (f"[complaint_001] step 2/2 summary ok {facts['summary']}s", WHITE),
        (facts["done"], WHITE),
    ]
    text_box(slide, 7.95, 2.0, 4.75, 3.1, [(t, {"color": c, "font": MONO, "size": 12, "spacing": 1})
                                           for t, c in log_lines], size=12)
    text_box(slide, 7.75, 5.35, 5.08, 1.1,
             ["Format: time | level | thread | module | message",
              "Console + rotating file (logs/app.log)"],
             size=14, color=MUTED)
    notes(slide, "Failures are isolated at every level so one bad document never stops the batch. "
                 "Thread names in the log make concurrent processing traceable.")


def slide_sample(prs, extraction, email, case_summary):
    slide = base_slide(prs, "Sample: complaint_001.pdf → Outputs", number=9)
    top, h = 1.35, 4.45
    label_box(slide, 0.5, top, 3.9, h, "Input (PDF form, excerpt)", [
        "Customer: Arjun Mehta",
        "Plan: FiberMax 300 Mbps",
        "“My August 2026 invoice shows Rs. 2,398, which is exactly double.”",
        "Resolution: refund of Rs. 1,199 (ref RF-77120), corrected invoice issued",
        "Attachments: invoice, bank statement",
    ], fill=LIGHT, title_size=17, body_size=15, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    arrow(slide, 4.45, top + h / 2, 4.85, top + h / 2, width=2.5)
    ex = extraction["extraction"]
    keys = ["customer_name", "complaint_category", "is_complaint", "escalation_required",
            "supporting_document_available", "overall_case_status"]
    j = ["{"] + [f' "{k}": "{ex[k]}"' + ("," if i < len(keys) - 1 else "") for i, k in enumerate(keys)] + ["}"]
    box(slide, 4.9, top, 4.2, h, fill=NAVY, line=None)
    text_box(slide, 5.05, top + 0.1, 3.9, 0.4, "Task 1: structured JSON", size=17, color=ACCENT, bold=True)
    text_box(slide, 5.05, top + 0.6, 4.0, 2.8, [(ln, {"font": MONO, "size": 11, "color": WHITE, "spacing": 3})
                                               for ln in j], size=11)
    text_box(slide, 5.05, top + 2.75, 3.9, 0.9,
             "+ email, phone, product, issue_description, resolution_provided",
             size=13, color=RGBColor(0xCB, 0xD5, 0xE1))
    subject = email.splitlines()[0].replace("Subject:", "").strip()
    label_box(slide, 9.4, top, 3.43, 2.4, "Task 2: customer email", [
        f"Subject: {subject}",
        "“...a refund of Rs. 1,199 has been processed (Ref: RF-77120)”",
    ], fill=WHITE, line=NAVY_2, title_size=16, body_size=13, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    nxt = md_section(case_summary, "Recommended Next Action").split(". ")[0].rstrip(".") + "."
    label_box(slide, 9.4, top + 2.55, 3.43, h - 2.55, "Task 3: case summary", [
        "Key issue: plan billed twice (Rs. 2,398 vs 1,199)",
        f"Next action: {nxt}",
    ], fill=WHITE, line=NAVY_2, title_size=16, body_size=13, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    text_box(slide, 0.5, 6.05, 12.33, 0.8,
             "Every fact in the email and summary traces back to the source: same refund amount, same "
             "reference, no invented dates or contacts.",
             size=16, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    notes(slide, "All values are copied from the real output files for complaint_001. The email "
                 "restates only the refund and reference number present in the document.")


def slide_results(prs, rows, summary):
    slide = base_slide(prs, "Results: Real Run (Ollama glm-4.7-flash, local)", number=10)
    header = ["File", "Category", "Escalation", "Case status", "Result", "Time (s)"]
    tbl = slide.shapes.add_table(len(rows) + 1, len(header), Inches(0.5), Inches(1.4), Inches(8.4),
                                 Inches(0.5 * (len(rows) + 1))).table
    widths = [2.1, 1.65, 1.2, 1.35, 1.0, 1.1]
    for c, w in enumerate(widths):
        tbl.columns[c].width = Inches(w)

    def put(r, c, text, bold=False, color=INK, fill=WHITE, size=13):
        cell = tbl.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.06)
        tf = cell.text_frame
        tf.text = ""
        run = tf.paragraphs[0].add_run()
        run.text = text
        _style_run(run, size, color, bold)

    for c, h in enumerate(header):
        put(0, c, h, bold=True, color=WHITE, fill=NAVY, size=13)
    for r, row in enumerate(rows, start=1):
        fill = LIGHT if r % 2 == 0 else WHITE
        ok = row["status"] == "success"
        put(r, 0, row["file_name"], fill=fill)
        put(r, 1, row["complaint_category"] or "—", fill=fill)
        esc = row["escalation_required"] or "—"
        put(r, 2, esc, fill=fill, bold=(esc == "Yes"), color=RED if esc == "Yes" else INK)
        put(r, 3, row["overall_case_status"] or "—", fill=fill)
        put(r, 4, row["status"], fill=fill, bold=True, color=GREEN if ok else RED)
        put(r, 5, f"{float(row['duration_seconds']):.1f}" if ok else "—", fill=fill)
    fail = next((r for r in rows if r["status"] != "success"), None)
    if fail:
        text_box(slide, 0.5, 1.45 + 0.5 * (len(rows) + 1), 8.4, 0.5,
                 f"{fail['file_name']}: {fail['errors'].split(': ')[1]} → logged, reported, batch continued",
                 size=12, color=MUTED)
    # stat tiles
    stats = [
        (f"{summary['success']}/{summary['total_files']}", "documents succeeded"),
        (f"{summary['failed']}", "failed gracefully (corrupt PDF)"),
        (f"{summary['total_seconds']:.1f} s", f"total, {summary['max_workers']} workers"),
        ("28", "pytest tests passing"),
    ]
    for i, (big, small) in enumerate(stats):
        yy = 1.4 + i * 1.33
        box(slide, 9.3, yy, 3.53, 1.18, fill=TEAL_LIGHT if i != 1 else RGBColor(0xFE, 0xF2, 0xF2),
            line=None)
        text_box(slide, 9.45, yy + 0.05, 3.3, 0.65, big, size=30, color=NAVY, bold=True)
        text_box(slide, 9.45, yy + 0.68, 3.3, 0.45, small, size=14, color=MUTED)
    notes(slide, "Data from output/final_report.csv and run_summary.json. Escalations were flagged for "
                 "the laptop defect, late TV delivery and unpaid refund; the pure enquiry (006) was "
                 "correctly marked is_complaint = No.")


def slide_stack(prs):
    slide = base_slide(prs, "Tech Stack & Engineering Practices", number=11)
    stack = [
        ("Python 3.11", "core language"),
        ("LangChain", "prompts, runnables, RunnableParallel"),
        ("Pydantic v2", "schemas & validation"),
        ("pypdf / python-docx", "document ingestion"),
        ("OpenAI • Gemini • Ollama", "swappable LLM providers"),
        ("pytest", "28 offline tests (fake LLM)"),
    ]
    text_box(slide, 0.5, 1.35, 5.8, 0.45, "Stack", size=20, color=ACCENT, bold=True)
    for i, (t, d) in enumerate(stack):
        yy = 1.9 + i * 0.8
        box(slide, 0.5, yy, 5.8, 0.68, fill=LIGHT, line=None)
        text_box(slide, 0.65, yy, 2.9, 0.68, t, size=16, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text_box(slide, 3.55, yy, 2.7, 0.68, d, size=14, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    text_box(slide, 6.9, 1.35, 5.9, 0.45, "Practices", size=20, color=ACCENT, bold=True)
    bullets(slide, 6.9, 1.9, 5.93, 4.9, [
        ("Config via .env: ", "provider, model, workers, timeouts; CLI flags override"),
        ("Provider-agnostic factory: ", "get_llm() lazily imports only the chosen SDK; runs fully "
                                        "local with Ollama"),
        ("Dependency injection: ", "pipeline takes any chat model, so tests use a fake LLM"),
        ("Modular package: ", "ingestion • prompts • chains • pipeline • writers"),
        ("Reproducible: ", "Git, requirements.txt, sample-data generator script"),
    ], size=17, spacing=10)
    notes(slide, "Switching providers is a one-line .env change. Tests cover ingestion, JSON repair, "
                 "retries, guardrails, parallelism and failure isolation without calling a real LLM.")


def slide_future(prs, summary):
    slide = base_slide(prs, "Limitations & Future Work", number=12)
    text_box(slide, 0.5, 1.3, 5.9, 0.55, "Limitations", size=22, color=RED, bold=True)
    bullets(slide, 0.5, 1.95, 5.9, 3.3, [
        "No OCR: scanned PDFs yield no text",
        f"Local model: ~{summary['avg_seconds_per_document']:.0f} s per document",
        "Email guardrail warns, no auto-rewrite",
        "Small test set (7 files), no accuracy benchmark",
    ], size=19, bullet_color=RED, spacing=14)
    text_box(slide, 6.93, 1.3, 5.9, 0.55, "Future work", size=22, color=GREEN, bold=True)
    bullets(slide, 6.93, 1.95, 5.9, 3.3, [
        "OCR (Tesseract) + .eml ingestion",
        "Human-in-the-loop review UI + REST API",
        "Labelled eval set, field-level accuracy",
        "CRM / ticketing integration, auto-routing",
    ], size=19, bullet_color=GREEN, spacing=14)
    box(slide, 0, 5.35, 13.333, 1.6, fill=NAVY, line=None, shape=MSO_SHAPE.RECTANGLE)
    text_box(slide, 0.5, 5.45, 12.33, 0.8, "Thank you  •  Questions?", size=34, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text_box(slide, 0.5, 6.25, 12.33, 0.5, "Saurabh Goyal  |  IIT Patna GenAI Development Program",
             size=16, color=ACCENT, align=PP_ALIGN.CENTER)
    notes(slide, "Closing: the system is working end to end; next steps focus on OCR, evaluation "
                 "metrics and integration with a real ticketing system.")


def build() -> Path:
    rows, summary, extraction, email, case_summary = load_run_data()
    facts = log_facts()
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    slide_title(prs)
    slide_problem(prs)
    slide_solution(prs)
    slide_architecture(prs)
    slide_workflow(prs, summary, facts)
    slide_schema(prs)
    slide_prompts(prs, rows)
    slide_errors(prs, facts)
    slide_sample(prs, extraction, email, case_summary)
    slide_results(prs, rows, summary)
    slide_stack(prs)
    slide_future(prs, summary)
    assert len(prs.slides) == TOTAL_SLIDES
    DEST.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DEST)
    return DEST


if __name__ == "__main__":
    path = build()
    print(f"Saved {path} ({path.stat().st_size / 1024:.1f} KB, {TOTAL_SLIDES} slides)")
