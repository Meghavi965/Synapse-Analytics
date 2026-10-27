from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional


@dataclass(frozen=True)
class MarketResearchResult:
    enabled: bool
    summary_markdown: str
    sources: list[str]
    benchmark_candidates: list[str]


def _firecrawl_enabled() -> bool:
    return bool(os.getenv("FIRECRAWL_API_KEY"))


def fetch_market_context(query: str) -> MarketResearchResult:
    """
    Optional 2026 market context via Firecrawl (if configured).

    Notes:
    - This is intentionally best-effort. If Firecrawl isn't configured,
      returns a disabled result with instructions.
    """
    if "market context" not in query.lower():
        return MarketResearchResult(enabled=False, summary_markdown="", sources=[], benchmark_candidates=[])

    if not _firecrawl_enabled():
        return MarketResearchResult(
            enabled=False,
            summary_markdown=(
                "Market Context requested, but Firecrawl is not configured. "
                "Set `FIRECRAWL_API_KEY` and install `firecrawl-py` to enable."
            ),
            sources=[],
            benchmark_candidates=[],
        )

    try:
        # Prefer official SDK if available.
        # firecrawl-py API tends to expose FirecrawlApp(api_key=...).
        from firecrawl import FirecrawlApp  # type: ignore

        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        results = app.search(query, limit=5)  # type: ignore[attr-defined]
        sources: list[str] = []
        candidates: list[str] = []
        bullets: list[str] = []
        for r in (results.get("data") if isinstance(results, dict) else results) or []:
            url = r.get("url") if isinstance(r, dict) else None
            title = r.get("title") if isinstance(r, dict) else None
            snippet = ""
            if isinstance(r, dict):
                snippet = str(
                    r.get("snippet")
                    or r.get("description")
                    or r.get("markdown")
                    or r.get("content")
                    or ""
                )
            if url:
                sources.append(url)
                bullets.append(f"- {title or url}")
            for m in re.findall(r"\b\d+(?:\.\d+)?\s*%\b", f"{title or ''} {snippet}"):
                if m not in candidates:
                    candidates.append(m)

        md = "### Market Context (Firecrawl)\n" + ("\n".join(bullets) if bullets else "- No results.")
        if candidates:
            md += "\n\n**Benchmark candidates (auto-extracted):** " + ", ".join(candidates[:8])
        return MarketResearchResult(enabled=True, summary_markdown=md, sources=sources, benchmark_candidates=candidates)
    except Exception as e:
        return MarketResearchResult(
            enabled=False,
            summary_markdown=f"Market Context requested but Firecrawl lookup failed: {e}",
            sources=[],
            benchmark_candidates=[],
        )
