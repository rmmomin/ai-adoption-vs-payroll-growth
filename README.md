# Replication kit: AI adoption vs employment growth (BLS CES + Ramp AI Index)

This folder contains a self-contained Python script to recreate the chart:

- **x-axis**: latest available Ramp AI adoption rate by NAICS sector (percent of businesses paying for AI tools)
- **y-axis**: sector payroll employment growth since 2024-01 minus the 2017–2023 trend (pp, annualized)

It exports both:
- a **dots** version (fixed point size)
- a **bubbles** version (bubble size scaled to sector employment level)
- a **3x3 collage** of alternative y-metrics against AI adoption

## Outputs

Running the script writes to `--outdir`:

- `ai_adoption_vs_growth_rel_trend_pp_data.csv`
- `ai_adoption_vs_growth_rel_trend_pp_dots.png`
- `ai_adoption_vs_growth_rel_trend_pp_bubbles.png`
- `ai_adoption_vs_growth_metric_collage.png`

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (scrape Ramp + BLS API)

```bash
python replicate_ai_adoption_vs_jobs_slowdown_api.py --outdir out
```

### Optional: BLS API key

The BLS API works without a key for many use cases, but if you hit limits or want to be safe, pass:

```bash
python replicate_ai_adoption_vs_jobs_slowdown_api.py --outdir out --bls_key YOUR_KEY
```

(You can request a key from BLS.)

### Optional: Manual Ramp CSV download fallback

If the script cannot auto-download the Ramp NAICS-sector dataset (site structure changes), download it manually from the Ramp AI Index page and run:

```bash
python replicate_ai_adoption_vs_jobs_slowdown_api.py --outdir out --ramp_csv /path/to/ramp_ai_index_naics_sector.csv
```

### Optional: Force end month

By default the script uses the **latest common month** that exists in both Ramp and BLS series.
You can force it (must exist in both) with:

```bash
python replicate_ai_adoption_vs_jobs_slowdown_api.py --outdir out --end_date 2026-01-01
```

## Methodology notes

### Ramp AI adoption
- The script expects the Ramp NAICS-sector download to include a `Date` column and columns like:
  `naics_sector_<sector>_ai_user_share` (percent strings such as `12.3%`).
- AI adoption is taken from the **latest common month** in the analysis window.

### BLS payroll employment
- Uses CES (Current Employment Statistics) payroll employment by sector, via the BLS Public Data API v2.
- Units are **thousands of employees**.

### “Growth since 2024 minus 2017–23 trend (pp, annualized)”
For each sector’s employment level series \(E_t\):

1. **Post period growth**: fit an OLS line to \(\log(E_t)\) on a monthly time index from **2024-01** through the end month.
2. **Pre-trend**: same fit from **2017-01** through **2023-12**.
3. Convert slopes to annualized percent: \(12 \times b \times 100\).
4. Take the difference: post minus pretrend, in percentage points.

### Sector mapping
CES is nonfarm. We therefore:
- exclude Agriculture (present in Ramp NAICS export),
- proxy Public administration with CES **Government**,
- proxy Educational services with CES **Private educational services**,
- proxy Mining, quarrying, and oil & gas extraction with CES **Mining and logging**.

If you want to change the sector set, edit `SECTOR_MAP` and `LABEL_MAP` in the script.
