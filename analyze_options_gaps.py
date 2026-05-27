"""
analyze_options_gaps.py
─────────────────────────────────────────────────────────────────────────────
Audits the options_history.db to identify coverage gaps that limit
backtesting fidelity. Tells us what to fix BEFORE spending credits on
re-downloads.

Five gap dimensions analyzed:
  1. Greeks coverage      -- which records lack delta/gamma/theta/vega/iv?
  2. Strike-chain density -- are we capturing only 1 strike per (ticker, date)?
  3. Side balance         -- does every (ticker, date) have both put AND call?
  4. Date coverage        -- which months have suspiciously few records?
  5. Ticker depth         -- per-ticker record counts and date ranges

Usage:
  python analyze_options_gaps.py

Outputs:
  Console:                 ranked gap report
  options_gaps_report.csv  per-ticker breakdown for follow-up
"""

import os
import sqlite3
import sys
from datetime import datetime
import pandas as pd

DB_PATH = r'C:\trading-bot\data\options_history.db'
OUTPUT_CSV = 'options_gaps_report.csv'


def connect():
    if not os.path.exists(DB_PATH):
        print(f"[X] {DB_PATH} not found")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def section(title, char='='):
    print()
    print(char * 80)
    print(f"  {title}")
    print(char * 80)


def main():
    conn = connect()
    print(f"Reading {DB_PATH}...")

    # Pull the full table (it's only ~22 MB)
    df = pd.read_sql_query("SELECT * FROM options_history", conn)
    conn.close()
    print(f"  {len(df):,} records loaded")

    # Detect what columns we have
    columns = list(df.columns)
    has_greeks = any(c in columns for c in ['delta', 'gamma', 'theta', 'vega', 'iv'])

    section("OVERVIEW")
    print(f"  Total records:    {len(df):,}")
    print(f"  Unique tickers:   {df['symbol'].nunique()}")
    print(f"  Date range:       {df['trade_date'].min()}  ->  {df['trade_date'].max()}")
    print(f"  Sides captured:   {sorted(df['side'].unique().tolist())}")
    print(f"  Columns:          {len(columns)}")
    print(f"  Greeks columns:   {'present' if has_greeks else 'MISSING (delta, gamma, theta, vega, iv)'}")
    print()
    print(f"  Existing columns: {', '.join(columns)}")

    # ── Gap 1: Greeks ──────────────────────────────────────────────────────
    section("GAP 1: GREEKS COVERAGE")
    if has_greeks:
        for col in ['delta', 'gamma', 'theta', 'vega', 'iv']:
            if col in columns:
                pct_filled = 100 * df[col].notna().sum() / len(df)
                print(f"  {col:<8}  {df[col].notna().sum():>7,} / {len(df):,}  ({pct_filled:.1f}%)")
    else:
        print("  *** No greeks captured anywhere in the DB. ***")
        print()
        print("  Impact:")
        print("    - Cannot evaluate true delta-based strike selection (using % OTM proxy)")
        print("    - Cannot model time decay (theta) in wheel/CSP exits")
        print("    - Cannot compute probability of assignment")
        print("    - Cannot identify mispriced options (no IV reference)")
        print()
        print("  Recommendation: full re-download capturing the 'greeks' field from")
        print("  marketdata.app /v1/options/chain/ (already returned in every response)")

    # ── Gap 2: Strikes per (ticker, date, expiration, side) ─────────────────
    section("GAP 2: STRIKE-CHAIN DENSITY")
    grouped = df.groupby(['symbol', 'trade_date', 'expiration', 'side']).size()
    print(f"  Avg strikes per (ticker, date, expiration, side): {grouped.mean():.2f}")
    print(f"  Median:                                            {grouped.median():.0f}")
    print(f"  Max:                                               {grouped.max()}")
    print(f"  Records with only 1 strike captured:               {(grouped == 1).sum():,}")
    print(f"  Records with 5+ strikes captured:                  {(grouped >= 5).sum():,}")
    if grouped.mean() < 2:
        print()
        print("  *** Single-strike capture detected. ***")
        print("  Impact:")
        print("    - Backtests can only test 1 strike per day (the 8% OTM choice)")
        print("    - Cannot evaluate 5% / 10% / 15% OTM alternatives")
        print("    - Cannot model spreads (need 2 strikes simultaneously)")
        print("    - Cannot search for best-delta strike per regime")
        print()
        print("  Recommendation: re-download with full chain capture (top 20 strikes)")

    # ── Gap 3: Side balance ────────────────────────────────────────────────
    section("GAP 3: PUT/CALL BALANCE")
    pivot = df.groupby(['symbol', 'trade_date'])['side'].nunique().reset_index()
    only_puts  = (pivot['side'] == 1) & (~df.merge(pivot, on=['symbol','trade_date'])
                                              .query('side_y == 1')['side_x'].eq('call'))
    # Simpler: count days per side per ticker
    side_counts = df.groupby(['symbol', 'side'])['trade_date'].nunique().unstack(fill_value=0)
    side_counts.columns.name = None
    if 'call' not in side_counts.columns: side_counts['call'] = 0
    if 'put'  not in side_counts.columns: side_counts['put']  = 0
    side_counts['balance'] = (side_counts[['put','call']].min(axis=1) /
                               side_counts[['put','call']].max(axis=1).replace(0, 1))
    side_counts = side_counts.sort_values('balance')
    print(f"  Per-ticker put/call balance (1.0 = same dates for both sides):")
    print(f"  {'Ticker':<10}{'Put days':>10}{'Call days':>11}{'Balance':>10}")
    print(f"  {'-'*45}")
    for ticker, row in side_counts.head(15).iterrows():
        flag = ' (!)' if row['balance'] < 0.90 else ''
        print(f"  {ticker:<10}{int(row['put']):>10,}{int(row['call']):>11,}"
              f"{row['balance']:>10.2f}{flag}")
    if (side_counts['balance'] < 0.90).any():
        n_imbalanced = (side_counts['balance'] < 0.90).sum()
        print()
        print(f"  *** {n_imbalanced} ticker(s) have mismatched put/call coverage. ***")
        print("  Impact: cannot run wheel (needs both sides) on those dates.")

    # ── Gap 4: Date coverage ───────────────────────────────────────────────
    section("GAP 4: DATE COVERAGE (records per month)")
    df['month'] = df['trade_date'].str[:7]
    monthly = df.groupby('month').size()
    print(f"  Records per month (last 24 months):")
    for month in sorted(monthly.index)[-24:]:
        bar = '#' * min(int(monthly[month] / 50), 60)
        print(f"  {month}  {monthly[month]:>6,}  {bar}")

    if len(monthly) > 0:
        median_monthly = monthly.median()
        sparse_months = monthly[monthly < median_monthly * 0.30]
        if len(sparse_months) > 0:
            print()
            print(f"  *** {len(sparse_months)} month(s) with <30% of median coverage: ***")
            for m in sparse_months.index[:10]:
                print(f"    {m}: only {sparse_months[m]:,} records")

    # ── Gap 5: Ticker depth ────────────────────────────────────────────────
    section("GAP 5: PER-TICKER DEPTH")
    ticker_stats = df.groupby('symbol').agg(
        records=('symbol', 'size'),
        first_date=('trade_date', 'min'),
        last_date=('trade_date', 'max'),
        unique_dates=('trade_date', 'nunique'),
        avg_strikes_per_date=('strike', 'count'),
    )
    ticker_stats['avg_strikes_per_date'] = ticker_stats['records'] / ticker_stats['unique_dates']
    ticker_stats = ticker_stats.sort_values('records')
    print(f"  {'Ticker':<10}{'Records':>10}{'Dates':>8}{'First':>13}{'Last':>13}{'Strk/Day':>10}")
    print(f"  {'-'*64}")
    for ticker, row in ticker_stats.iterrows():
        flag = ''
        if row['records'] < 200:        flag = ' (thin)'
        elif row['records'] < 400:      flag = ' (modest)'
        print(f"  {ticker:<10}{int(row['records']):>10,}{int(row['unique_dates']):>8}"
              f"{row['first_date']:>13}{row['last_date']:>13}"
              f"{row['avg_strikes_per_date']:>10.2f}{flag}")

    # ── Output CSV ─────────────────────────────────────────────────────────
    ticker_stats.to_csv(OUTPUT_CSV)
    print()
    print(f"[Per-ticker details saved to {OUTPUT_CSV}]")

    # ── Summary recommendations ────────────────────────────────────────────
    section("RECOMMENDED FIXES (prioritized)", char='#')

    fixes = []
    if not has_greeks:
        fixes.append(("HIGH",  "Re-download with greeks captured  "
                              "(blocks delta-based analysis, theta decay modeling, IV-based filters)"))
    if grouped.mean() < 2:
        fixes.append(("HIGH",  "Re-download with full chain (top 20 strikes)  "
                              "(unblocks strike comparison, spread strategies, delta calibration)"))
    if (side_counts['balance'] < 0.90).any():
        fixes.append(("MED",   "Fill put/call mismatches for affected tickers"))
    if len(sparse_months) > 5:
        fixes.append(("MED",   "Backfill sparse months (>30% below median)"))
    thin_tickers = (ticker_stats['records'] < 200).sum()
    if thin_tickers > 0:
        fixes.append(("LOW",   f"Decide: drop or backfill {thin_tickers} thin tickers"))

    for sev, action in fixes:
        print(f"  [{sev}] {action}")

    if not fixes:
        print("  No major gaps detected.")

    print()
    print(f"  Cost estimate for full re-download (greeks + chain):")
    n_chain_calls = (grouped.shape[0])  # number of (ticker, date, expiration, side) tuples
    print(f"    ~{n_chain_calls:,} chain calls (one credit each, but capture all strikes + greeks)")
    print(f"    ~{n_chain_calls // 5:,} expiration lookup calls")
    print(f"    Total credits: ~{int(n_chain_calls * 1.2):,}")
    print(f"    Wall time at parallel x2: ~{int(n_chain_calls * 0.4 / 60 / 2):,} minutes")


if __name__ == '__main__':
    main()
    