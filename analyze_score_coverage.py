"""
analyze_score_coverage.py
─────────────────────────────────────────────────────────────────────────────
Examines backtest_daily_scores.csv and identifies tickers that:
  1. Score well consistently (would be live-bot candidates)
  2. Are NOT in our current options download universe
  3. Are worth adding to the options data harvest

Produces a ranked list with stats so we can decide which to add.

Usage:
  python analyze_score_coverage.py

Outputs:
  Console:                       ranked candidates with stats
  candidate_tickers.csv          machine-readable list for next download
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────

SCORES_CSV = r'C:\trading-bot\backtest_daily_scores.csv'
OPTIONS_DB = r'C:\trading-bot\data\options_history.db'
OUTPUT_CSV = 'candidate_tickers.csv'

# Thresholds for "interesting" ticker
MIN_QUALIFIED_DAYS  = 30    # must qualify (score >= 0.50) on at least N days
MIN_HIGH_SCORE_DAYS = 5     # must hit score >= 0.70 at least N times
MIN_AVG_SCORE       = 0.45  # avg score across all observations
TOP_N               = 50    # how many top candidates to show


# ── Load data ──────────────────────────────────────────────────────────────

def load_scores():
    if not os.path.exists(SCORES_CSV):
        print(f"[X] {SCORES_CSV} not found")
        return None
    print(f"Loading {SCORES_CSV}...")
    df = pd.read_csv(SCORES_CSV)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} unique tickers")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    return df


def load_options_universe():
    """Return set of tickers currently in options DB."""
    if not os.path.exists(OPTIONS_DB):
        print(f"[!] {OPTIONS_DB} not found -- assuming empty options universe")
        return set()
    conn = sqlite3.connect(OPTIONS_DB)
    df = pd.read_sql_query("SELECT DISTINCT symbol FROM options_history", conn)
    conn.close()
    tickers = set(df['symbol'].tolist())
    print(f"  Current options DB has {len(tickers)} unique tickers")
    return tickers


# ── Per-ticker analysis ────────────────────────────────────────────────────

def compute_ticker_stats(df):
    """For each ticker, compute scoring statistics."""
    stats = []
    for tkr, grp in df.groupby('ticker'):
        n_days = len(grp)
        if n_days < 30:  # skip tickers with too little history
            continue

        qualified_days = (grp['score'] >= 0.50).sum()
        high_score_days = (grp['score'] >= 0.70).sum()
        very_high_days = (grp['score'] >= 0.85).sum()   # CSP-eligible threshold
        avg_score = grp['score'].mean()
        max_score = grp['score'].max()
        median_score = grp['score'].median()

        # Regime distribution -- what regimes does this ticker tend to fire in?
        regime_dist = grp[grp['score'] >= 0.50]['regime'].value_counts().to_dict()

        # Quality of scoring -- when it qualifies, how high is it?
        qualified_scores = grp[grp['score'] >= 0.50]['score']
        avg_when_qualified = qualified_scores.mean() if not qualified_scores.empty else 0

        stats.append({
            'ticker': tkr,
            'n_days_scored': n_days,
            'qualified_days': qualified_days,
            'pct_qualified': round(100 * qualified_days / n_days, 1),
            'high_score_days': high_score_days,    # >= 0.70
            'very_high_days': very_high_days,      # >= 0.85 (CSP threshold)
            'avg_score': round(avg_score, 4),
            'avg_when_qualified': round(avg_when_qualified, 4),
            'median_score': round(median_score, 4),
            'max_score': round(max_score, 4),
            'flow_days': regime_dist.get('flow', 0),
            'neutral_days': regime_dist.get('neutral', 0),
            'volcautious_days': regime_dist.get('volatility-cautious', 0),
        })
    return pd.DataFrame(stats)


# ── Ranking ────────────────────────────────────────────────────────────────

def rank_candidates(stats_df, current_universe):
    """Add columns: in_options_db, candidate_score, recommendation."""
    stats_df = stats_df.copy()
    stats_df['in_options_db'] = stats_df['ticker'].isin(current_universe)

    # Composite "should we add this" score:
    #   Heavy weight on consistent qualification (not just one-off spikes)
    #   Bonus for very-high (CSP-eligible) score days
    #   Penalty for already being in the DB (we already have this data)
    stats_df['candidate_score'] = (
        stats_df['qualified_days'] * 1.0
        + stats_df['high_score_days'] * 2.0
        + stats_df['very_high_days'] * 4.0
        + stats_df['avg_score'] * 50
    )

    # Reasons to skip
    stats_df['skip_reason'] = ''
    stats_df.loc[stats_df['qualified_days'] < MIN_QUALIFIED_DAYS, 'skip_reason'] = 'too few qualified days'
    stats_df.loc[stats_df['high_score_days'] < MIN_HIGH_SCORE_DAYS, 'skip_reason'] = 'no high-score days'
    stats_df.loc[stats_df['avg_score'] < MIN_AVG_SCORE, 'skip_reason'] = 'avg score too low'

    # Final recommendation
    def reco(row):
        if row['in_options_db']:
            return 'already_in_db'
        if row['skip_reason']:
            return f"skip ({row['skip_reason']})"
        return 'ADD'

    stats_df['recommendation'] = stats_df.apply(reco, axis=1)
    return stats_df.sort_values('candidate_score', ascending=False)


# ── Print report ───────────────────────────────────────────────────────────

def print_report(stats_df):
    print()
    print('=' * 100)
    print(' SCORE COVERAGE ANALYSIS')
    print('=' * 100)

    # 1. Already in DB -- how well are we using them?
    in_db = stats_df[stats_df['in_options_db']].copy()
    print()
    print(f'-- TICKERS ALREADY IN OPTIONS DB ({len(in_db)} tickers) --')
    print(f'   How well-used is current data?')
    print()
    print(f"{'Ticker':<8}{'Days':>6}{'Qual':>6}{'Qual%':>7}{'High':>6}{'V.High':>7}"
          f"{'AvgSc':>8}{'AvgQual':>9}{'Flow':>6}{'Ntrl':>6}")
    print('-' * 75)
    for _, row in in_db.head(25).iterrows():
        print(f"{row['ticker']:<8}"
              f"{row['n_days_scored']:>6}"
              f"{row['qualified_days']:>6}"
              f"{row['pct_qualified']:>6.1f}%"
              f"{row['high_score_days']:>6}"
              f"{row['very_high_days']:>7}"
              f"{row['avg_score']:>8.3f}"
              f"{row['avg_when_qualified']:>9.3f}"
              f"{row['flow_days']:>6}"
              f"{row['neutral_days']:>6}")

    # 2. Candidates to add
    candidates = stats_df[
        (~stats_df['in_options_db']) &
        (stats_df['skip_reason'] == '')
    ].copy()

    print()
    print(f'-- TOP CANDIDATES TO ADD ({len(candidates)} total qualify) --')
    print(f'   Tickers NOT in options DB that score consistently:')
    print()
    print(f"{'Rank':<5}{'Ticker':<8}{'Days':>6}{'Qual':>6}{'Qual%':>7}{'High':>6}{'V.High':>7}"
          f"{'AvgSc':>8}{'AvgQual':>9}{'Flow':>6}{'Score':>9}")
    print('-' * 80)
    for i, (_, row) in enumerate(candidates.head(TOP_N).iterrows(), 1):
        print(f"{i:<5}"
              f"{row['ticker']:<8}"
              f"{row['n_days_scored']:>6}"
              f"{row['qualified_days']:>6}"
              f"{row['pct_qualified']:>6.1f}%"
              f"{row['high_score_days']:>6}"
              f"{row['very_high_days']:>7}"
              f"{row['avg_score']:>8.3f}"
              f"{row['avg_when_qualified']:>9.3f}"
              f"{row['flow_days']:>6}"
              f"{row['candidate_score']:>9.1f}")

    # 3. Coverage gap analysis
    print()
    print('-- COVERAGE GAP --')
    in_db_qualified = in_db['qualified_days'].sum() if not in_db.empty else 0
    not_in_db_qualified = candidates['qualified_days'].sum() if not candidates.empty else 0
    total_qualified = in_db_qualified + not_in_db_qualified
    if total_qualified > 0:
        in_db_pct = 100 * in_db_qualified / total_qualified
        print(f'   Total qualified-day events in scoring data: {total_qualified:,}')
        print(f'   Covered by options DB:    {in_db_qualified:,} ({in_db_pct:.1f}%)')
        print(f'   NOT covered (add-list):   {not_in_db_qualified:,} ({100-in_db_pct:.1f}%)')
        print()
        if (100 - in_db_pct) > 50:
            print(f'   *** SIGNIFICANT GAP ***')
            print(f'   Adding candidates would more than double our options coverage of')
            print(f'   the days when the live bot would actually be trading.')


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    df = load_scores()
    if df is None:
        return

    print()
    print('Loading current options universe...')
    current_universe = load_options_universe()

    print()
    print('Computing per-ticker statistics...')
    stats_df = compute_ticker_stats(df)
    print(f'  {len(stats_df)} tickers analyzed')

    stats_df = rank_candidates(stats_df, current_universe)

    print_report(stats_df)

    # Save full ranking
    stats_df.to_csv(OUTPUT_CSV, index=False)
    print()
    print(f'[Full ranking saved to {OUTPUT_CSV}]')

    # Tell user what to do next
    candidates_to_add = stats_df[
        (~stats_df['in_options_db']) &
        (stats_df['skip_reason'] == '')
    ]
    print()
    print('=' * 80)
    print(' NEXT STEPS')
    print('=' * 80)
    if len(candidates_to_add) > 0:
        top10 = candidates_to_add.head(10)['ticker'].tolist()
        print(f' {len(candidates_to_add)} candidates qualify for the add-list.')
        print(f' Top 10: {", ".join(top10)}')
        print()
        print(' To add these to the next options harvest:')
        print(f'   1. Review the top of {OUTPUT_CSV}')
        print(f'   2. Pick N tickers to add (start small -- 5-10 at a time)')
        print(f'   3. Add them to TICKERS list in download_calls_universe.py')
        print(f'      and download_options_history.py')
        print(f'   4. Run downloads on the server')
        print()
        print(' Cost estimate per ticker added:')
        print('   ~250 puts + ~250 calls = 500 records = ~600 credits = ~7 minutes')


if __name__ == '__main__':
    main()