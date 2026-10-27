from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core.oracle_engine import OracleAnswer
from utils.market_research import MarketResearchResult


def workspace_root() -> Path:
    # self_healing_bi_agent/<this file> -> parent is self_healing_bi_agent -> parent is workspace root
    return Path(__file__).resolve().parents[2]


def _ensure_dirs() -> tuple[Path, Path, Path]:
    root = workspace_root()
    tmp_dir = root / "tmp" / "pdfs"
    pdf_out = root / "output" / "pdf"
    md_out = root / "output" / "reports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_out.mkdir(parents=True, exist_ok=True)
    md_out.mkdir(parents=True, exist_ok=True)
    return tmp_dir, pdf_out, md_out


@dataclass(frozen=True)
class ReportArtifacts:
    markdown_path: Path
    pdf_path: Optional[Path]
    chart_paths: list[Path]


def _format_accuracy(answer: OracleAnswer) -> str:
    if not answer.forecast:
        return "N/A"
    r2 = answer.forecast.accuracy_r2
    mape = answer.forecast.accuracy_mape
    if r2 is not None:
        return f"R²={r2:.2f}" + (f", MAPE={mape:.1%}" if mape is not None else "")
    if mape is not None:
        return f"MAPE={mape:.1%}"
    return "N/A"


def render_markdown_report(
    answer: OracleAnswer,
    *,
    df_snapshot: pd.DataFrame,
    market: Optional[MarketResearchResult] = None,
) -> str:
    date_str = answer.asked_at.strftime("%Y-%m-%d %H:%M")
    accuracy = _format_accuracy(answer)
    steps_md = "\n".join([f"- {s}" for s in answer.strategic_next_steps])
    ver_md = "\n".join([f"- {n}" for n in answer.verification_notes]) if answer.verification_notes else "- None"

    schema = ", ".join(map(str, df_snapshot.columns.tolist()))
    try:
        desc = df_snapshot.describe(numeric_only=True).round(4).to_markdown()
    except Exception:
        desc = df_snapshot.describe(numeric_only=True).round(4).to_string()

    market_md = ""
    if market and market.summary_markdown:
        market_md = "\n\n" + market.summary_markdown + "\n"

    return f"""# Executive Strategy Report

**Generated:** {date_str}

## Executive Summary
- **Question:** {answer.question}
- **Data Trust Score:** {answer.trust_score:.0%}
- **Diagnosis:** {answer.diagnosis}

## Descriptive Insight
{answer.descriptive_insight}

## Predictive Outlook (Next 3 Months)
- **Model:** {answer.forecast.model_name if answer.forecast else "N/A"}
- **Accuracy:** {accuracy}

## Strategic Next Steps
{steps_md}

## Chain-of-Verification
{ver_md}

## Data Snapshot
- **Schema:** {schema}

### df.describe()
{desc}
{market_md}
"""


def export_plotly_figure(fig, out_path: Path) -> Path:
    """
    Export Plotly figure to PNG (requires kaleido).
    """
    try:
        fig.write_image(str(out_path), scale=2)  # kaleido
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"Plotly image export failed ({e}). Install `kaleido` to enable chart embedding in PDFs."
        ) from e


def build_pdf(
    *,
    pdf_path: Path,
    title: str,
    subtitle: str,
    body_lines: list[str],
    image_paths: list[Path],
) -> None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.units import inch
    except Exception as e:
        raise RuntimeError(f"PDF dependencies missing: {e}. Install `reportlab`.") from e

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER, title=title)
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    for line in body_lines:
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 0.08 * inch))

    story.append(Spacer(1, 0.15 * inch))

    max_w = 7.2 * inch
    for p in image_paths:
        if p.exists():
            story.append(Image(str(p), width=max_w, height=3.8 * inch, kind="proportional"))
            story.append(Spacer(1, 0.15 * inch))

    doc.build(story)


def generate_executive_report(
    answer: OracleAnswer,
    *,
    df_snapshot: pd.DataFrame,
    proof_fig=None,
    forecast_fig=None,
    market: Optional[MarketResearchResult] = None,
) -> ReportArtifacts:
    tmp_dir, pdf_out, md_out = _ensure_dirs()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = md_out / f"executive_report_{ts}.md"
    pdf_path = pdf_out / f"executive_report_{ts}.pdf"

    md = render_markdown_report(answer, df_snapshot=df_snapshot, market=market)
    md_path.write_text(md, encoding="utf-8")

    chart_paths: list[Path] = []
    for fig, name in ((proof_fig, "proof"), (forecast_fig, "forecast")):
        if fig is None:
            continue
        png_path = tmp_dir / f"{name}_{ts}.png"
        try:
            chart_paths.append(export_plotly_figure(fig, png_path))
        except Exception:
            # Keep the report generation alive even if image export is unavailable.
            continue

    subtitle = f"Generated {answer.asked_at.strftime('%Y-%m-%d %H:%M')}"
    body = [
        f"<b>Question:</b> {answer.question}",
        f"<b>Data Trust Score:</b> {answer.trust_score:.0%} ({answer.diagnosis})",
        f"<b>Descriptive:</b> {answer.descriptive_insight}",
        "<b>Strategic Next Steps:</b>",
        "<br/>".join([f"- {s}" for s in answer.strategic_next_steps]),
    ]
    if market and market.summary_markdown:
        body.append("<b>Market Context:</b>")
        body.append(market.summary_markdown.replace("\n", "<br/>"))

    pdf_final: Optional[Path] = None
    try:
        build_pdf(
            pdf_path=pdf_path,
            title="Executive Strategy Report",
            subtitle=subtitle,
            body_lines=body,
            image_paths=chart_paths,
        )
        pdf_final = pdf_path
    except Exception:
        pdf_final = None

    return ReportArtifacts(markdown_path=md_path, pdf_path=pdf_final, chart_paths=chart_paths)
