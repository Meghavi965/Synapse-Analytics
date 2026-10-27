from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# A basic UA helps avoid some blocks on anonymous requests.
_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}

_TROY_OUNCE_TO_GRAM = 31.1034768


class LiveMarketDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetalsDataConfig:
    history_range: str = "2y"
    interval: str = "1d"
    gold_unit: str = "10g"  # "1g" | "10g" | "oz"
    silver_unit: str = "1kg"  # "1g" | "1kg" | "oz"
    tz: str = "Asia/Kolkata"


def _yahoo_chart_url(ticker: str, *, history_range: str, interval: str) -> str:
    base = _YAHOO_CHART_URL.format(ticker=ticker)
    params = {
        "range": history_range,
        "interval": interval,
        "includePrePost": "false",
        "events": "div|split",
        "lang": "en-US",
        "region": "US",
    }
    return base + "?" + urllib.parse.urlencode(params)


def fetch_yahoo_history(ticker: str, *, history_range: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Minimal Yahoo Finance chart fetch (no external deps).

    Returns:
      DataFrame columns: ["date", "close"]
    """
    url = _yahoo_chart_url(ticker, history_range=history_range, interval=interval)
    try:
        req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except Exception as e:
        raise LiveMarketDataError(f"Failed to fetch Yahoo history for {ticker}: {e}") from e

    chart = payload.get("chart") or {}
    if chart.get("error"):
        desc = (chart.get("error") or {}).get("description") or "Unknown Yahoo chart error"
        raise LiveMarketDataError(f"Yahoo chart error for {ticker}: {desc}")

    results = chart.get("result") or []
    if not results:
        raise LiveMarketDataError(f"Yahoo chart returned no result for {ticker}.")

    r0 = results[0] or {}
    ts = r0.get("timestamp") or []
    quote = ((r0.get("indicators") or {}).get("quote") or [{}])[0] or {}
    close = quote.get("close") or []

    n = min(len(ts), len(close))
    ts = ts[:n]
    close = close[:n]

    dates = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True)
    try:
        dates = dates.dt.tz_convert("Asia/Kolkata")
    except Exception:
        pass
    dates = dates.dt.tz_localize(None)

    df = pd.DataFrame({"date": dates, "close": pd.to_numeric(pd.Series(close), errors="coerce")})
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise LiveMarketDataError(f"Yahoo chart returned no usable rows for {ticker}.")
    return df


def _to_inr_per_unit(usd_per_oz: pd.Series, usd_inr: pd.Series, *, unit: str) -> pd.Series:
    inr_per_oz = usd_per_oz * usd_inr
    unit = str(unit).lower().strip()
    if unit in {"oz", "ounce"}:
        return inr_per_oz
    if unit in {"1g", "g", "gram"}:
        return inr_per_oz / _TROY_OUNCE_TO_GRAM
    if unit in {"10g"}:
        return (inr_per_oz / _TROY_OUNCE_TO_GRAM) * 10.0
    if unit in {"1kg", "kg", "kilogram"}:
        return (inr_per_oz / _TROY_OUNCE_TO_GRAM) * 1000.0
    # Default: INR per gram.
    return inr_per_oz / _TROY_OUNCE_TO_GRAM


def fetch_metals_india_prices(config: Optional[MetalsDataConfig] = None) -> pd.DataFrame:
    """
    Fetch "live" gold + silver series and convert to INR units.

    NOTE: This is an approximation of India spot pricing:
    - Gold/Silver are sourced from Yahoo Finance futures (GC=F / SI=F) in USD.
    - Converted using USD/INR (USDINR=X).
    - Local premiums/taxes are not included.

    Returns long-form DataFrame columns:
      ["date", "asset", "price_inr", "unit", "source"]
    """
    cfg = config or MetalsDataConfig()

    gold = fetch_yahoo_history("GC=F", history_range=cfg.history_range, interval=cfg.interval).rename(
        columns={"close": "gold_usd_oz"}
    )
    silver = fetch_yahoo_history("SI=F", history_range=cfg.history_range, interval=cfg.interval).rename(
        columns={"close": "silver_usd_oz"}
    )
    fx = fetch_yahoo_history("USDINR=X", history_range=cfg.history_range, interval=cfg.interval).rename(
        columns={"close": "usd_inr"}
    )

    merged = gold.merge(silver, on="date", how="outer").merge(fx, on="date", how="outer").sort_values("date")
    merged["usd_inr"] = merged["usd_inr"].ffill()
    merged = merged.dropna(subset=["usd_inr"])

    out_frames: list[pd.DataFrame] = []
    if "gold_usd_oz" in merged.columns:
        g = merged.dropna(subset=["gold_usd_oz"]).copy()
        g["asset"] = "gold"
        g["unit"] = cfg.gold_unit
        g["price_inr"] = _to_inr_per_unit(g["gold_usd_oz"], g["usd_inr"], unit=cfg.gold_unit)
        out_frames.append(g[["date", "asset", "price_inr", "unit"]])

    if "silver_usd_oz" in merged.columns:
        s = merged.dropna(subset=["silver_usd_oz"]).copy()
        s["asset"] = "silver"
        s["unit"] = cfg.silver_unit
        s["price_inr"] = _to_inr_per_unit(s["silver_usd_oz"], s["usd_inr"], unit=cfg.silver_unit)
        out_frames.append(s[["date", "asset", "price_inr", "unit"]])

    if not out_frames:
        raise LiveMarketDataError("No metals series returned after conversion.")

    out = pd.concat(out_frames, ignore_index=True).dropna(subset=["date", "price_inr"]).sort_values(["asset", "date"])
    out["source"] = "Yahoo Finance (GC=F/SI=F) * USDINR=X (converted to INR)"
    out["asof"] = datetime.now()
    return out.reset_index(drop=True)

