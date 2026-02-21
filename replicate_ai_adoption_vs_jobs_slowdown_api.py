#!/usr/bin/env python3
"""
Replication kit: AI adoption vs employment growth relative to trend (BLS CES + Ramp AI Index)

What this script does
---------------------
1) Downloads / loads the Ramp AI Index "AI adoption by NAICS sector" dataset.
   - Preferred: scrape and auto-discover the downloadable dataset from:
       https://ramp.com/data/ai-index
   - Fallback: user can download the CSV manually and pass --ramp_csv.

2) Pulls payroll employment (CES, thousands of employees) by sector using the BLS Public Data API v2.

3) Constructs:
   X (AI adoption): latest available month (percent of businesses paying for AI tools), by NAICS sector.
   Y (jobs slowdown / acceleration): 
       [annualized log-linear employment growth from 2024-01 through end_date]
     - [annualized log-linear employment growth trend from 2017-01 through 2023-12]
     expressed in percentage points (pp).

4) Outputs:
   - CSV of sector-level plot inputs
   - Dot scatter PNG
   - Bubble scatter PNG (bubble size ~ employment level)

Usage (recommended)
-------------------
python replicate_ai_adoption_vs_jobs_slowdown_api.py --outdir out

Optional arguments
------------------
--ramp_csv PATH   Use a local Ramp CSV instead of scraping.
--bls_key KEY     BLS registration key (optional; helpful if you hit rate/limit issues).
--end_date YYYY-MM-01  Force the analysis end month (must exist in BOTH datasets).
--start_year_pre  Default 2017
--start_year_post Default 2024

Notes / caveats
---------------
- Industry mapping:
  Ramp uses NAICS sectors; CES uses payroll employment industries. We map:
    * Public administration -> CES "Government"
    * Educational services -> CES "Private educational services"
    * Mining, quarrying, and oil & gas extraction -> CES "Mining and logging"
  Agriculture is excluded (CES is nonfarm).

- Ramp site structure can change. The scraping routine is defensive:
  it tries to discover a downloadable CSV/XLSX/JSON by scanning the page HTML and
  then attempts a short list of plausible "download" endpoints. If it fails, the
  script prints instructions for manual download.

Dependencies
------------
pip install pandas numpy matplotlib requests openpyxl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

RAMP_AI_INDEX_URL = "https://ramp.com/data/ai-index"
BLS_API_V2_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Mapping from Ramp sector key -> CES series id (seasonally adjusted employment, thousands)
SECTOR_MAP: Dict[str, str] = {
    "accommodation_and_food_services": "CES7072000001",
    "administrative_and_support_and_waste_management_and_remediation_services": "CES6056000001",
    "arts_entertainment_and_recreation": "CES7071000001",
    "construction": "CES2000000001",
    "educational_services": "CES6561000001",  # private educational services proxy
    "finance_and_insurance": "CES5552000001",
    "health_care_and_social_assistance": "CES6562000001",
    "information": "CES5000000001",
    "management_of_companies_and_enterprises": "CES6055000001",
    "manufacturing": "CES3000000001",
    "mining_quarrying_and_oil_and_gas_extraction": "CES1000000001",  # mining and logging proxy
    "other_services_except_public_administration": "CES8000000001",
    "professional_scientific_and_technical_services": "CES6054000001",
    "public_administration": "CES9000000001",  # government proxy
    "real_estate_and_rental_and_leasing": "CES5553000001",
    "retail_trade": "CES4200000001",
    "transportation_and_warehousing": "CES4300000001",
    "utilities": "CES4422000001",
    "wholesale_trade": "CES4142000001",
}

LABEL_MAP: Dict[str, str] = {
    "accommodation_and_food_services": "Accommodation and Food Services",
    "administrative_and_support_and_waste_management_and_remediation_services": "Administrative & Support + Waste Mgmt",
    "arts_entertainment_and_recreation": "Arts, Entertainment, and Recreation",
    "construction": "Construction",
    "educational_services": "Educational Services (Private)",
    "finance_and_insurance": "Finance and Insurance",
    "health_care_and_social_assistance": "Health Care and Social Assistance",
    "information": "Information",
    "management_of_companies_and_enterprises": "Management of Companies",
    "manufacturing": "Manufacturing",
    "mining_quarrying_and_oil_and_gas_extraction": "Mining and Logging",
    "other_services_except_public_administration": "Other Services (ex. Public Admin)",
    "professional_scientific_and_technical_services": "Professional, Scientific, Technical Svcs",
    "public_administration": "Government (proxy for Public Admin)",
    "real_estate_and_rental_and_leasing": "Real Estate, Rental, Leasing",
    "retail_trade": "Retail Trade",
    "transportation_and_warehousing": "Transportation and Warehousing",
    "utilities": "Utilities",
    "wholesale_trade": "Wholesale Trade",
}


# -----------------------------
# Ramp helpers
# -----------------------------

def _absolute_url(base: str, url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("//"):
        return "https:" + url
    # relative path
    if base.endswith("/"):
        base = base[:-1]
    if not url.startswith("/"):
        url = "/" + url
    return base + url


def _extract_candidate_urls(html: str, base_url: str) -> List[str]:
    """Return a de-duplicated list of plausible data URLs from an HTML page."""
    urls: List[str] = []

    # 1) absolute URLs in HTML text
    urls += re.findall(r"https?://[^\s\"'<>]+", html)

    # 2) href/src attributes (may be relative)
    for attr in ("href", "src"):
        for m in re.findall(fr'{attr}\s*=\s*"([^"]+)"', html, flags=re.IGNORECASE):
            urls.append(_absolute_url(base_url, m))
        for m in re.findall(fr"{attr}\s*=\s*'([^']+)'", html, flags=re.IGNORECASE):
            urls.append(_absolute_url(base_url, m))

    # Keep only likely file types or URLs containing ai-index or "download"
    keep: List[str] = []
    for u in urls:
        ul = u.lower()
        if any(ul.endswith(ext) for ext in (".csv", ".xlsx", ".xls", ".json")):
            keep.append(u)
            continue
        if ("ai-index" in ul or "ai_index" in ul) and ("data" in ul or "download" in ul):
            keep.append(u)
            continue

    # De-dupe while preserving order
    seen = set()
    out = []
    for u in keep:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _try_parse_as_dataframe(content: bytes, url: str) -> Optional[pd.DataFrame]:
    """Try to parse downloaded bytes as CSV/XLSX/JSON -> DataFrame."""
    ul = url.lower()
    try:
        if ul.endswith(".csv"):
            return pd.read_csv(BytesIO(content))
        if ul.endswith(".xlsx") or ul.endswith(".xls"):
            return pd.read_excel(BytesIO(content))
        if ul.endswith(".json"):
            obj = json.loads(content.decode("utf-8", errors="ignore"))
            # Common patterns: dict with 'data' or list of records
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                for k in ("data", "results", "rows", "items"):
                    if k in obj and isinstance(obj[k], (list, dict)):
                        try:
                            return pd.DataFrame(obj[k])
                        except Exception:
                            pass
                # last resort: flatten dict of dicts
                try:
                    return pd.json_normalize(obj)
                except Exception:
                    return None
        # content-type based fallback
        text = content.decode("utf-8", errors="ignore")
        if "," in text.splitlines()[0]:
            return pd.read_csv(StringIO(text))
    except Exception:
        return None
    return None


def _looks_like_ramp_ai_index_naics(df: pd.DataFrame) -> bool:
    """Heuristic: Ramp NAICS-sector export usually has a 'Date' column and many 'naics_sector_*_ai_user_share' cols."""
    cols = [c.lower() for c in df.columns.astype(str)]
    if "date" in cols:
        # wide-format export
        if any("naics_sector_" in c and "ai_user_share" in c for c in cols):
            return True
    # long-format possibility
    if any("ai_user_share" in c for c in cols) and any("naics" in c or "sector" in c for c in cols):
        return True
    return False


def load_ramp_ai_index(ramp_csv: Optional[Path] = None, cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Returns a *long* DataFrame with columns: date, sector_key, ai_share (0..1).
    """
    if ramp_csv is not None:
        df = pd.read_csv(ramp_csv)
        return normalize_ramp_df(df)

    # Optional cache (for reproducibility)
    if cache_path is not None and cache_path.exists():
        df = pd.read_csv(cache_path)
        return normalize_ramp_df(df)

    print(f"[Ramp] Attempting to scrape/download data from {RAMP_AI_INDEX_URL} ...")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; replication-kit/1.0; +https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    html_resp = requests.get(RAMP_AI_INDEX_URL, headers=headers, timeout=60)
    html_resp.raise_for_status()
    html = html_resp.text

    # 1) auto-discover file links in HTML
    candidates = _extract_candidate_urls(html, base_url="https://ramp.com")
    tried: List[str] = []
    for u in candidates:
        tried.append(u)
        try:
            r = requests.get(u, headers=headers, timeout=60)
            if r.status_code != 200 or len(r.content) < 200:
                continue
            df = _try_parse_as_dataframe(r.content, u)
            if df is None:
                continue
            if _looks_like_ramp_ai_index_naics(df):
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                print(f"[Ramp] Found dataset via: {u}")
                return normalize_ramp_df(df)
        except Exception:
            continue

    # 2) try a few plausible download endpoints (site structure can change)
    fallback_urls = [
        # common patterns
        "https://ramp.com/data/ai-index.csv",
        "https://ramp.com/data/ai-index.xlsx",
        "https://ramp.com/data/ai-index/data.csv",
        "https://ramp.com/data/ai-index/data.xlsx",
        "https://ramp.com/data/ai-index/download",
        "https://ramp.com/data/ai-index/download.csv",
        "https://ramp.com/data/ai-index?download=1",
        "https://ramp.com/data/ai-index?format=csv",
        "https://ramp.com/data/ai-index?output=csv",
    ]
    for u in fallback_urls:
        tried.append(u)
        try:
            r = requests.get(u, headers=headers, timeout=60)
            if r.status_code != 200 or len(r.content) < 200:
                continue
            df = _try_parse_as_dataframe(r.content, u)
            if df is None:
                continue
            if _looks_like_ramp_ai_index_naics(df):
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                print(f"[Ramp] Found dataset via fallback URL: {u}")
                return normalize_ramp_df(df)
        except Exception:
            continue

    msg = (
        "[Ramp] Could not auto-download the NAICS-sector dataset.\n"
        "Tried these URLs (first few shown):\n"
        f"  - " + "\n  - ".join(tried[:10]) + ("\n  - ..." if len(tried) > 10 else "") + "\n\n"
        "Manual fallback:\n"
        "  1) Open https://ramp.com/data/ai-index in a browser.\n"
        "  2) Use the chart/table download control (CSV) to download the NAICS-sector file.\n"
        "  3) Re-run this script with: --ramp_csv path/to/downloaded_file.csv\n"
    )
    raise RuntimeError(msg)


def normalize_ramp_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize possible Ramp exports to long format with:
        date (Timestamp), sector_key (str), ai_share (float in [0,1])
    """
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}

    # Standard wide format: "Date" + many naics_sector_*_ai_user_share columns
    date_col = None
    for c in cols:
        if lower[c] == "date":
            date_col = c
            break
    if date_col is not None and any("naics_sector_" in lower[c] and "ai_user_share" in lower[c] for c in cols):
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        long = out.melt(id_vars=[date_col], var_name="sector_col", value_name="ai_share_raw")
        long["sector_key"] = (
            long["sector_col"]
            .astype(str)
            .str.replace("naics_sector_", "", regex=False)
            .str.replace("_ai_user_share", "", regex=False)
        )
        long["ai_share"] = _parse_percent_to_share(long["ai_share_raw"])
        long = long.rename(columns={date_col: "date"})[["date", "sector_key", "ai_share"]]
        long = long.dropna(subset=["date", "sector_key", "ai_share"])
        return long

    # Attempt long-format inference
    # Common candidate col names
    possible_date_cols = [c for c in cols if "date" in lower[c] or "month" in lower[c]]
    possible_share_cols = [c for c in cols if "ai_user_share" in lower[c] or ("ai" in lower[c] and "share" in lower[c])]
    possible_sector_cols = [c for c in cols if "naics" in lower[c] or "sector" in lower[c] or "industry" in lower[c]]

    if possible_date_cols and possible_share_cols and possible_sector_cols:
        dcol = possible_date_cols[0]
        scol = possible_sector_cols[0]
        vcol = possible_share_cols[0]
        out = df[[dcol, scol, vcol]].copy()
        out[dcol] = pd.to_datetime(out[dcol])
        out["sector_key"] = out[scol].astype(str)
        out["ai_share"] = _parse_percent_to_share(out[vcol])
        return out.rename(columns={dcol: "date"})[["date", "sector_key", "ai_share"]].dropna()

    raise ValueError(
        "Unrecognized Ramp AI Index export format. "
        "Expected a 'Date' column and columns like 'naics_sector_*_ai_user_share'."
    )


def _parse_percent_to_share(s: pd.Series) -> pd.Series:
    """
    Convert values like '12.3%' or 0.123 or 12.3 to shares in [0,1].
    """
    ss = s.astype(str).str.strip()
    # Handle percent strings
    has_pct = ss.str.contains("%", na=False)
    out = pd.to_numeric(ss.str.replace("%", "", regex=False), errors="coerce")
    # if any had %, treat as percent points
    if has_pct.any():
        out = out / 100.0
    else:
        # If values look like 0..1, keep. If look like 0..100, divide.
        if out.max(skipna=True) is not None and out.max(skipna=True) > 1.5:
            out = out / 100.0
    return out


# -----------------------------
# BLS helpers
# -----------------------------

def fetch_bls_timeseries(series_ids: List[str], start_year: int, end_year: int, registration_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch a list of series IDs from the BLS Public Data API (v2).
    Returns a long DataFrame: series_id, date (Timestamp), value (float).
    """
    payload: Dict[str, object] = {"seriesid": series_ids, "startyear": str(start_year), "endyear": str(end_year)}
    if registration_key:
        payload["registrationkey"] = registration_key

    headers = {"Content-Type": "application/json"}
    r = requests.post(BLS_API_V2_URL, data=json.dumps(payload), headers=headers, timeout=120)
    r.raise_for_status()
    j = r.json()

    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed: status={j.get('status')} message={j.get('message')}")

    rows = []
    for series in j["Results"]["series"]:
        sid = series["seriesID"]
        for obs in series.get("data", []):
            period = obs.get("period", "")
            if not isinstance(period, str) or not period.startswith("M"):
                continue
            if period in ("M13", "M14"):
                continue
            try:
                year = int(obs["year"])
                month = int(period[1:])
                val = float(obs["value"])
            except Exception:
                continue
            rows.append({"series_id": sid, "date": pd.Timestamp(year=year, month=month, day=1), "value": val})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("BLS API returned no monthly observations for the requested series.")
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def annualized_log_growth(values: pd.Series) -> float:
    """
    Annualized log-linear growth rate (%), using an OLS fit of log(level) on a monthly index.

    If y_t is employment level, we estimate:
        log(y_t) = a + b * t
    where t is month index. Then annualized growth = 12*b*100.
    """
    v = values.dropna().astype(float)
    if len(v) < 6:
        return float("nan")
    y = np.log(v.to_numpy())
    x = np.arange(len(v), dtype=float)
    slope = np.polyfit(x, y, 1)[0]  # per month
    return float(slope * 12 * 100)


@dataclass
class GrowthComponents:
    end_date: pd.Timestamp
    post_growth_ann_pct: float
    pretrend_growth_ann_pct: float
    rel_pp: float
    emp_latest_thousands: float


def compute_growth_components(ts: pd.Series, end_date: pd.Timestamp, start_year_pre: int = 2017, start_year_post: int = 2024) -> GrowthComponents:
    """
    Compute:
      - post growth: from start_year_post-01 through end_date
      - pretrend: from start_year_pre-01 through 2023-12
    """
    ts = ts.sort_index()

    start_post = pd.Timestamp(start_year_post, 1, 1)
    end_pre = pd.Timestamp(start_year_post - 1, 12, 1)  # 2023-12 if start_year_post=2024
    start_pre = pd.Timestamp(start_year_pre, 1, 1)

    post = ts[(ts.index >= start_post) & (ts.index <= end_date)]
    pre = ts[(ts.index >= start_pre) & (ts.index <= end_pre)]

    g_post = annualized_log_growth(post)
    g_pre = annualized_log_growth(pre)

    latest = float(ts.get(end_date, np.nan))
    return GrowthComponents(
        end_date=end_date,
        post_growth_ann_pct=g_post,
        pretrend_growth_ann_pct=g_pre,
        rel_pp=g_post - g_pre,
        emp_latest_thousands=latest,
    )


# -----------------------------
# Plotting helpers
# -----------------------------

def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, R^2."""
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(m), float(b), float(r2)


def plot_scatter(
    df: pd.DataFrame,
    outpath: Path,
    bubbles: bool,
    start_year_pre: int = 2017,
    start_year_post: int = 2024,
) -> None:
    """
    Scatter of:
      x = ai_adoption_current_pct
      y = ann_growth_rel_to_trend_pp
    """
    df = df.dropna(subset=["ai_adoption_current_pct", "ann_growth_rel_to_trend_pp"]).copy()

    x = df["ai_adoption_current_pct"].to_numpy(float)
    y = df["ann_growth_rel_to_trend_pp"].to_numpy(float)

    m, b, r2 = fit_line(x, y)
    end_month = pd.to_datetime(df["end_date"].iloc[0]).strftime("%Y-%m")

    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor("#d9d9d9")
    ax.set_facecolor("#d9d9d9")

    if bubbles:
        emp = df["emp_latest_thousands"].to_numpy(float)
        # bubble area proportional to employment level
        scale = 3000.0 / np.nanmax(emp) if np.nanmax(emp) > 0 else 1.0
        sizes = np.maximum(emp * scale, 70.0)
        ax.scatter(x, y, s=sizes, color="#5fa8d3", edgecolors="#2f84c2", linewidths=1.0, alpha=0.65)
    else:
        ax.scatter(x, y, s=95, color="#5fa8d3", edgecolors="#2f84c2", linewidths=1.0, alpha=0.85)

    # Best fit line
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xs, m * xs + b, linestyle="--", linewidth=2.0, color="#2f7fb8", alpha=1.0)

    # axis limits and tick marks close to the reference chart style
    x_lo = max(0.0, np.floor((np.nanmin(x) - 2.0) / 5.0) * 5.0)
    x_hi = np.ceil((np.nanmax(x) + 2.0) / 5.0) * 5.0
    y_lo = np.floor(np.nanmin(y) - 0.3)
    y_hi = np.ceil(np.nanmax(y) + 0.3)
    y_lo = min(y_lo, 0.0)
    y_hi = max(y_hi, 0.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    xt_start = int(math.ceil(x_lo / 10.0) * 10)
    xt_end = int(math.floor(x_hi / 10.0) * 10)
    if xt_end >= xt_start:
        ax.set_xticks(np.arange(xt_start, xt_end + 1, 10))
    ax.set_yticks(np.arange(int(math.floor(y_lo)), int(math.ceil(y_hi)) + 1, 1))
    ax.tick_params(labelsize=12)

    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.set_xlabel(f"AI adoption rate (Ramp, {end_month}, percent)", fontsize=15)
    ax.set_ylabel("Payroll employment growth relative to 2017-23 trend\n(pp, annualized)", fontsize=15)

    ax.set_title(
        "AI adoption vs payroll employment growth relative to trend\n"
        f"End month: {end_month}  |  Y = growth since {start_year_post} minus {start_year_pre}-23 trend\n"
        "(pp, annualized)",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # regression text
    reg_txt = f"Unweighted OLS: y = {m:.4f}x + {b:.3f}  (R²={r2:.3f})"
    ax.text(0.98, 0.98, reg_txt, transform=ax.transAxes, ha="right", va="top", fontsize=12)

    # annotate all sectors with leader lines
    offsets = {
        "public_administration": (6, 6),
        "health_care_and_social_assistance": (6, 6),
        "accommodation_and_food_services": (6, 5),
        "other_services_except_public_administration": (6, 6),
        "arts_entertainment_and_recreation": (6, 6),
        "construction": (6, 5),
        "retail_trade": (6, 4),
        "mining_quarrying_and_oil_and_gas_extraction": (6, 6),
        "educational_services": (6, 6),
        "utilities": (6, 4),
        "wholesale_trade": (6, 6),
        "real_estate_and_rental_and_leasing": (6, 5),
        "administrative_and_support_and_waste_management_and_remediation_services": (6, 4),
        "manufacturing": (6, 5),
        "management_of_companies_and_enterprises": (6, 5),
        "finance_and_insurance": (6, 5),
        "professional_scientific_and_technical_services": (6, 5),
        "transportation_and_warehousing": (6, 5),
        "information": (6, 5),
    }
    for _, r in df.iterrows():
        xx = float(r["ai_adoption_current_pct"])
        yy = float(r["ann_growth_rel_to_trend_pp"])
        lab = str(r["label"])
        sk = str(r["sector_key"])
        dx, dy = offsets.get(sk, (6, 5))
        ax.annotate(
            lab,
            (xx, yy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    # source and notes under chart
    source_txt = "Source: BLS (CES), Ramp; author's calculations."
    size_note = (
        "Bubble size is proportional to CES payroll employment in the end month."
        if bubbles
        else "Dot size is fixed in this version."
    )
    notes_txt = (
        "Notes: X is Ramp AI adoption rate in the end month shown. Y is the difference between the annualized log-linear growth rate of CES payroll employment from "
        f"{start_year_post}-01 through the end month and the annualized log-linear trend from {start_year_pre}-01 through {start_year_post - 1}-12. {size_note}"
    )
    notes_txt = textwrap.fill(notes_txt, width=150)
    fig.text(0.01, 0.10, source_txt, ha="left", va="bottom", fontsize=13)
    fig.text(0.01, 0.018, notes_txt, ha="left", va="bottom", fontsize=11, linespacing=1.15)

    # reserve room for source/notes block
    fig.tight_layout(rect=(0, 0.19, 1, 1))

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("--ramp_csv", type=Path, default=None, help="Local Ramp AI Index CSV (optional)")
    ap.add_argument("--bls_key", type=str, default=None, help="BLS API registration key (optional)")
    ap.add_argument("--end_date", type=str, default=None, help="Force analysis end month (YYYY-MM-01), must exist in both datasets")
    ap.add_argument("--start_year_pre", type=int, default=2017, help="Start year for pre-trend window (default 2017)")
    ap.add_argument("--start_year_post", type=int, default=2024, help="Start year for post window (default 2024)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1) Ramp data (AI adoption)
    ramp_cache = args.outdir / "cache_ramp_ai_index.csv"
    ramp_long = load_ramp_ai_index(ramp_csv=args.ramp_csv, cache_path=ramp_cache if args.ramp_csv is None else None)

    # Keep only mapped sectors (nonfarm)
    ramp_long = ramp_long[ramp_long["sector_key"].isin(SECTOR_MAP.keys())].copy()

    # 2) BLS data (employment)
    series_ids = list(SECTOR_MAP.values())
    this_year = pd.Timestamp.today().year
    start_year = min(args.start_year_pre, args.start_year_post)
    end_year = this_year
    print(f"[BLS] Downloading CES series via API for {start_year}–{end_year} ...")
    bls_df = fetch_bls_timeseries(series_ids, start_year=start_year, end_year=end_year, registration_key=args.bls_key)

    # 3) Choose end_date as latest common month across BOTH datasets
    ramp_end = pd.Timestamp(ramp_long["date"].max())
    bls_end = pd.Timestamp(bls_df["date"].max())
    end_date = min(ramp_end, bls_end)

    if args.end_date is not None:
        end_date = pd.Timestamp(args.end_date)
        if end_date > ramp_end or end_date > bls_end:
            raise ValueError(
                f"--end_date {end_date.date()} is after the latest common month "
                f"(ramp_end={ramp_end.date()}, bls_end={bls_end.date()})."
            )

    print(f"[Align] Using end_date = {end_date.date()} (latest common month)")

    # 4) Current AI adoption level (latest month)
    current = ramp_long[ramp_long["date"] == end_date].set_index("sector_key")["ai_share"]

    # 5) Build sector metrics from BLS
    rows = []
    for sector_key, sid in SECTOR_MAP.items():
        ts = bls_df[bls_df["series_id"] == sid].set_index("date")["value"].sort_index()
        comp = compute_growth_components(
            ts, end_date=end_date, start_year_pre=args.start_year_pre, start_year_post=args.start_year_post
        )
        rows.append(
            {
                "sector_key": sector_key,
                "label": LABEL_MAP.get(sector_key, sector_key),
                "series_id": sid,
                "ai_adoption_current_pct": float(current.get(sector_key, np.nan) * 100.0),
                "emp_latest_thousands": comp.emp_latest_thousands,
                "ann_growth_post_pct": comp.post_growth_ann_pct,
                "ann_growth_trend_pre_pct": comp.pretrend_growth_ann_pct,
                "ann_growth_rel_to_trend_pp": comp.rel_pp,
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
        )

    df_plot = pd.DataFrame(rows)

    # 6) Export CSV + plots
    csv_path = args.outdir / "ai_adoption_vs_growth_rel_trend_pp_data.csv"
    df_plot.to_csv(csv_path, index=False)
    print(f"[Output] Wrote {csv_path}")

    plot_scatter(
        df_plot,
        args.outdir / "ai_adoption_vs_growth_rel_trend_pp_dots.png",
        bubbles=False,
        start_year_pre=args.start_year_pre,
        start_year_post=args.start_year_post,
    )
    print("[Output] Wrote dots PNG")

    plot_scatter(
        df_plot,
        args.outdir / "ai_adoption_vs_growth_rel_trend_pp_bubbles.png",
        bubbles=True,
        start_year_pre=args.start_year_pre,
        start_year_post=args.start_year_post,
    )
    print("[Output] Wrote bubbles PNG")

    print("[Done]")


if __name__ == "__main__":
    main()
