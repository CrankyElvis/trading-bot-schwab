"""
reality_check.py
─────────────────────────────────────────────────────────────────────────────
Analyzes multi_strategy_backtester output to produce a 4-question reality check.

This is a DIAGNOSTIC tool, not a tuning tool. It reads:
  - all_trades.csv                       (every trade with score, regime, P&L)
  - headline_per_ticker_strategy.csv     (per-ticker per-strategy metrics)
  - strategy_regime_score_matrix.csv     (the 3D matrix)
  - backtest_summary.json                (primary backtester headline numbers)

And produces a one-page report answering 4 questions:

  1. Are aggregate returns in BELIEVABLE ranges? (sanity check vs reality)
  2. Does SCORE actually predict outcomes? (signal validation -- the key question)
  3. Do findings REPLICATE across tickers? (or are they single-ticker accidents?)
  4. Does any strategy beat BUY_HOLD on risk-adjusted basis? (alpha check)

Usage:
  python reality_check.py

Output:
  reality_check_report.txt   (the diagnostic report)
  Console:                   (same report, prettier)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
TRADES_CSV       = 'all_trades.csv'
HEADLINE_CSV     = 'headline_per_ticker_strategy.csv'
SCORE_MATRIX_CSV = 'strategy_regime_score_matrix.csv'
PRIMARY_SUMMARY  = 'backtest_summary.json'
OUTPUT_FILE      = 'reality_check_report.txt'

# Believability bounds for 1-year systematic options strategies on mid-cap names
# Generous bounds: if results land OUTSIDE these, it's a red flag.
BELIEVABLE_RANGES = {
    'WHEEL':     (-15, 25),    # premium harvesting, capped upside
    'PURE_CSP':  (-10, 20),    # similar, no assignment phase
    'LONG_CALL': (-80, 60),    # high variance is normal, asymmetric payoff
    'BUY_HOLD':  (-50, 100),   # ticker-dependent, wide range
}

# Sample size threshold below which results are NOISE
MIN_SAMPLE_FOR_SIGNAL = 20


# ── Helpers ────────────────────────────────────────────────────────────────

def write(out, msg=''):
    """Print and write to file."""
    print(msg)
    out.append(msg)


def section(out, title, char='='):
    write(out, '')
    write(out, char * 80)
    write(out, f"  {title}")
    write(out, char * 80)


def subsection(out, title):
    write(out, '')
    write(out, f"  -- {title} " + '-' * (75 - len(title)))


# ── Question 1: Believability ──────────────────────────────────────────────

def check_believability(headline_df, out):
    section(out, "Q1: Are returns in BELIEVABLE ranges?")
    write(out, "")
    write(out, "  Compares each (ticker, strategy) return to the rough bounds of what's")
    write(out, "  plausible for a 1-year systematic strategy. Returns far outside these")
    write(out, "  bounds usually mean look-ahead bias, accounting bug, or extreme luck.")
    write(out, "")

    flags = []
    by_strategy = headline_df.groupby('strategy')
    for strat, grp in by_strategy:
        lo, hi = BELIEVABLE_RANGES.get(strat, (-100, 200))
        in_range = grp[(grp['total_return_pct'] >= lo) & (grp['total_return_pct'] <= hi)]
        out_high = grp[grp['total_return_pct'] > hi]
        out_low  = grp[grp['total_return_pct'] < lo]
        pct_in   = 100 * len(in_range) / len(grp) if len(grp) else 0

        verdict = "OK" if pct_in >= 80 else "FLAG"
        write(out, f"  {strat:<12} bounds: [{lo:+d}%, {hi:+d}%]   "
                   f"in-range: {len(in_range)}/{len(grp)} ({pct_in:.0f}%)   "
                   f"[{verdict}]")

        if len(out_high) > 0:
            extreme_hi = out_high.nlargest(3, 'total_return_pct')
            for _, row in extreme_hi.iterrows():
                flags.append(f"    [HIGH] {row['ticker']}: {row['total_return_pct']:+.1f}% "
                             f"(believable max: +{hi}%)")

        if len(out_low) > 0:
            extreme_lo = out_low.nsmallest(3, 'total_return_pct')
            for _, row in extreme_lo.iterrows():
                flags.append(f"    [LOW]  {row['ticker']}: {row['total_return_pct']:+.1f}% "
                             f"(believable min: {lo}%)")

    if flags:
        write(out, "")
        write(out, "  Outliers (worth investigating):")
        for f in flags[:15]:  # cap at 15 to keep readable
            write(out, f)
    else:
        write(out, "")
        write(out, "  No extreme outliers. Results land in plausible ranges.")


# ── Question 2: Score monotonicity ─────────────────────────────────────────

def check_score_signal(score_matrix_df, out):
    section(out, "Q2: Does SCORE predict outcomes?  *** THE KEY QUESTION ***")
    write(out, "")
    write(out, "  If the signal stack has real edge, HIGHER scores should produce HIGHER")
    write(out, "  avg P&L within each (strategy, regime). Test: is the ordering")
    write(out, "  LOW < MEDIUM < HIGH for avg_pnl?  Reversed or flat = no edge.")
    write(out, "")

    if score_matrix_df.empty:
        write(out, "  No score matrix data. Cannot evaluate.")
        return

    # Filter to cells with meaningful sample size
    meaningful = score_matrix_df[score_matrix_df['n_trades'] >= MIN_SAMPLE_FOR_SIGNAL].copy()

    # For each (strategy, regime), check bucket ordering
    monotonic_pass = 0
    monotonic_fail = 0
    insufficient   = 0
    details = []

    for strat in sorted(meaningful['strategy'].unique()):
        for regime in sorted(meaningful[meaningful['strategy'] == strat]['regime'].unique()):
            cells = meaningful[
                (meaningful['strategy'] == strat) &
                (meaningful['regime'] == regime)
            ].copy()
            # Keep only LOW/MEDIUM/HIGH (drop NO_SCORE)
            cells = cells[cells['score_bucket'].isin(['LOW', 'MEDIUM', 'HIGH'])]
            if len(cells) < 2:
                insufficient += 1
                continue

            # Order buckets canonically
            order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            cells['order'] = cells['score_bucket'].map(order)
            cells = cells.sort_values('order')

            pnls = cells['avg_pnl'].values
            buckets = cells['score_bucket'].values

            # Monotonic non-decreasing?
            is_monotonic = all(pnls[i] <= pnls[i+1] for i in range(len(pnls)-1))
            # Reversed (higher score = lower P&L)?
            is_reversed = all(pnls[i] >= pnls[i+1] for i in range(len(pnls)-1))

            if is_monotonic and not is_reversed:
                monotonic_pass += 1
                verdict = "MONOTONIC (edge)"
            elif is_reversed:
                monotonic_fail += 1
                verdict = "REVERSED (negative edge)"
            else:
                monotonic_fail += 1
                verdict = "MIXED (no edge)"

            bucket_str = ' < '.join(f"{b}=${p:+.0f}" for b, p in zip(buckets, pnls))
            details.append(f"  {strat:<12}{regime:<22} {verdict}")
            details.append(f"                                 {bucket_str}")

    write(out, "")
    for line in details:
        write(out, line)

    total = monotonic_pass + monotonic_fail
    if total == 0:
        write(out, "")
        write(out, f"  Insufficient sample sizes in all cells. Cannot evaluate signal.")
        write(out, f"  (Need N>={MIN_SAMPLE_FOR_SIGNAL} trades per bucket.)")
        return

    pct = 100 * monotonic_pass / total
    write(out, "")
    write(out, f"  Score signal validation: {monotonic_pass}/{total} ({pct:.0f}%) "
               f"(strategy, regime) combos show monotonic edge.")

    if pct >= 60:
        write(out, "  VERDICT: Score has real predictive power. Signal stack is working.")
    elif pct >= 40:
        write(out, "  VERDICT: Mixed signal. Score helps in some cases, not others.")
        write(out, "           Need bigger sample to determine which cells are real.")
    else:
        write(out, "  VERDICT: Score does NOT systematically predict outcomes.")
        write(out, "           Signal stack may be noise on this dataset.")
        write(out, "           Primary backtester's headline returns are suspect.")


# ── Question 3: Cross-ticker replication ───────────────────────────────────

def check_replication(headline_df, out):
    section(out, "Q3: Do findings REPLICATE across tickers?")
    write(out, "")
    write(out, "  For each strategy, what fraction of tickers show similar performance?")
    write(out, "  If 1-2 tickers carry all the returns, that's not a strategy -- that's")
    write(out, "  ticker selection luck.")
    write(out, "")

    by_strategy = headline_df.groupby('strategy')
    for strat, grp in by_strategy:
        if grp.empty: continue
        rets = grp['total_return_pct'].values
        median_ret = np.median(rets)
        mean_ret = np.mean(rets)
        sd = np.std(rets)
        positive = (rets > 0).sum()
        n = len(rets)

        # Concentration metric: does the top ticker account for >50% of total return?
        total = rets.sum()
        top1 = rets.max() if total != 0 else 0
        concentration = (top1 / total) if total > 0 else 0

        # Sharpe across tickers (does median strategy compete with mean?)
        # Big gap between mean and median = a few outliers carry it
        mean_minus_median = mean_ret - median_ret

        write(out, f"  {strat:<12}  n_tickers={n:>3}  "
                   f"median={median_ret:+6.1f}%  mean={mean_ret:+6.1f}%  "
                   f"sd={sd:5.1f}%  positive={positive}/{n}")

        verdicts = []
        if abs(mean_minus_median) > sd:
            verdicts.append("LOPSIDED (mean-median gap suggests few tickers carry it)")
        if total > 0 and concentration > 0.5:
            verdicts.append(f"TOP-HEAVY ({concentration*100:.0f}% of total return from 1 ticker)")
        if positive / n < 0.50:
            verdicts.append(f"MOSTLY LOSING ({positive}/{n} positive)")

        if verdicts:
            for v in verdicts:
                write(out, f"               -> {v}")


# ── Question 4: Beats BUY_HOLD? ────────────────────────────────────────────

def check_vs_buyhold(headline_df, out):
    section(out, "Q4: Does any option strategy BEAT BUY_HOLD?")
    write(out, "")
    write(out, "  For each ticker, compare each option strategy's Sharpe and return")
    write(out, "  to BUY_HOLD. Strategies must beat BUY_HOLD on RISK-ADJUSTED basis")
    write(out, "  to justify their complexity.")
    write(out, "")

    # Pivot: rows = ticker, cols = strategy, values = sharpe and return
    pivot_ret = headline_df.pivot_table(
        index='ticker', columns='strategy', values='total_return_pct', aggfunc='first')
    pivot_sharpe = headline_df.pivot_table(
        index='ticker', columns='strategy', values='sharpe_ratio', aggfunc='first')

    if 'BUY_HOLD' not in pivot_ret.columns:
        write(out, "  BUY_HOLD column missing. Cannot compare.")
        return

    bh_ret = pivot_ret['BUY_HOLD']
    bh_sharpe = pivot_sharpe['BUY_HOLD']

    for strat in ['WHEEL', 'PURE_CSP', 'LONG_CALL']:
        if strat not in pivot_ret.columns:
            continue
        beats_return = (pivot_ret[strat] > bh_ret).sum()
        beats_sharpe = (pivot_sharpe[strat] > bh_sharpe).sum()
        n = pivot_ret[strat].notna().sum()
        avg_diff = (pivot_ret[strat] - bh_ret).mean()
        avg_sharpe_diff = (pivot_sharpe[strat] - bh_sharpe).mean()

        write(out, f"  {strat:<12} vs BUY_HOLD")
        write(out, f"    beats return:  {beats_return:>3}/{n} tickers   "
                   f"(avg diff: {avg_diff:+6.1f}%)")
        write(out, f"    beats Sharpe:  {beats_sharpe:>3}/{n} tickers   "
                   f"(avg diff: {avg_sharpe_diff:+6.3f})")
        write(out, "")

    write(out, "  INTERPRETATION:")
    write(out, "    'Beats Sharpe' is the meaningful metric -- it measures alpha")
    write(out, "    after accounting for risk. If a strategy beats BUY_HOLD on")
    write(out, "    Sharpe <50% of tickers, it's not a strategy, it's overhead.")


# ── Cross-reference vs primary backtester ──────────────────────────────────

def check_vs_primary(headline_df, out):
    section(out, "Q5: SANITY CHECK vs Primary Backtester")
    write(out, "")
    write(out, "  The primary backtester ran on a 256-ticker universe over 5 years with")
    write(out, "  many strategies stacked. The multi-strategy ran ~22 tickers x 1 year")
    write(out, "  with 4 isolated strategies. Big differences in scope -- but the")
    write(out, "  ORDER OF MAGNITUDE for typical option strategy returns should align.")
    write(out, "")

    if not os.path.exists(PRIMARY_SUMMARY):
        write(out, f"  {PRIMARY_SUMMARY} not found. Skipping primary comparison.")
        return

    with open(PRIMARY_SUMMARY) as f:
        primary = json.load(f)

    primary_annual = primary.get('annual_return_pct', 0)
    primary_sharpe = primary.get('sharpe_ratio', 0)
    primary_dd = primary.get('max_drawdown_pct', 0)

    # Best multi-strategy ticker return
    if not headline_df.empty:
        best_row = headline_df.loc[headline_df['total_return_pct'].idxmax()]
        worst_row = headline_df.loc[headline_df['total_return_pct'].idxmin()]
        median_ret = headline_df['total_return_pct'].median()

        write(out, f"  Primary backtester:")
        write(out, f"    Annual return:   {primary_annual:+8.1f}%   "
                   f"Sharpe: {primary_sharpe:.3f}   MaxDD: {primary_dd:.1f}%")
        write(out, "")
        write(out, f"  Multi-strategy (per (ticker, strategy), 1yr):")
        write(out, f"    Best:            {best_row['ticker']} / {best_row['strategy']}: "
                   f"{best_row['total_return_pct']:+.1f}%   Sharpe: {best_row['sharpe_ratio']:.3f}")
        write(out, f"    Worst:           {worst_row['ticker']} / {worst_row['strategy']}: "
                   f"{worst_row['total_return_pct']:+.1f}%   Sharpe: {worst_row['sharpe_ratio']:.3f}")
        write(out, f"    Median:          {median_ret:+.1f}% across all combos")

        write(out, "")
        if primary_annual > 50 and median_ret < 10:
            write(out, "  *** RED FLAG ***")
            write(out, f"  Primary claims {primary_annual:.0f}%/yr but multi-strategy median is {median_ret:.1f}%.")
            write(out, "  The primary's headline likely comes from:")
            write(out, "    - Signal proxies with look-ahead bias (politician/insider)")
            write(out, "    - Stacking many strategies + protective rules tuned to history")
            write(out, "    - Lucky ticker concentration on the 256-name universe")
            write(out, "  Live trading returns will be FAR lower than the backtest implies.")
        elif primary_annual > 30 and median_ret > 5:
            write(out, "  PARTIAL ALIGNMENT")
            write(out, "  Primary's returns are high but multi-strategy shows positive signal.")
            write(out, "  Some edge is real; some is likely fitting. Paper trading required.")
        else:
            write(out, "  ALIGNED")
            write(out, "  Primary and multi-strategy results are in roughly compatible ranges.")


# ── Final synthesis ────────────────────────────────────────────────────────

def final_verdict(out, score_pass_pct=None):
    section(out, "OVERALL VERDICT", char='#')
    write(out, "")
    write(out, "  This is a REALITY CHECK, not a tuning report.")
    write(out, "")
    write(out, "  The honest path forward:")
    write(out, "")
    write(out, "  IF the score signal is monotonic (Q2 shows >60% monotonic):")
    write(out, "    - Real edge exists in some configurations")
    write(out, "    - Don't change weights based on this dataset (would be overfit)")
    write(out, "    - Add score logging to live paper bot for OUT-OF-SAMPLE validation")
    write(out, "    - Re-evaluate in 3-6 months with real forward-looking data")
    write(out, "")
    write(out, "  IF the score signal is mixed or absent (Q2 shows <40% monotonic):")
    write(out, "    - Primary backtester's headline is largely artifact")
    write(out, "    - Top priorities: fix look-ahead bias in politician/insider signals")
    write(out, "    - Don't add new features, signals, or strategies until base is honest")
    write(out, "    - Revise live bot expectations down significantly")
    write(out, "")
    write(out, "  WHAT TO AVOID (overfitting traps):")
    write(out, "    - Raising MIN_SCORE because high scores 'worked' on this data")
    write(out, "    - Disabling strategies in regimes where they 'failed' (small sample)")
    write(out, "    - Reweighting signals based on edge column (already flagged as biased)")
    write(out, "    - Any change that just makes the historical Sharpe number bigger")
    write(out, "")
    write(out, "  WHAT IS LEGITIMATE:")
    write(out, "    - Fixing actual bugs (CSP simulation, accounting errors)")
    write(out, "    - Replacing biased signal proxies with real data sources")
    write(out, "    - Logging scores/decisions/outcomes for future validation")
    write(out, "    - Acknowledging that backtests cap at ~Sharpe 1.0 for honest models")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    out = []

    write(out, f"# Reality Check Report")
    write(out, f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verify inputs exist
    missing = []
    for path in [TRADES_CSV, HEADLINE_CSV, SCORE_MATRIX_CSV]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        write(out, "")
        write(out, "[X] Missing input files:")
        for m in missing:
            write(out, f"    {m}")
        write(out, "")
        write(out, "Run multi_strategy_backtester.py first.")
        with open(OUTPUT_FILE, 'w') as f:
            f.write('\n'.join(out))
        sys.exit(1)

    # Load
    trades_df = pd.read_csv(TRADES_CSV)
    headline_df = pd.read_csv(HEADLINE_CSV)
    score_matrix_df = pd.read_csv(SCORE_MATRIX_CSV)

    write(out, "")
    write(out, f"Loaded:")
    write(out, f"  {len(trades_df):>5} trades")
    write(out, f"  {len(headline_df):>5} (ticker, strategy) headline rows")
    write(out, f"  {len(score_matrix_df):>5} score-matrix cells")

    # Run all checks
    check_believability(headline_df, out)
    check_score_signal(score_matrix_df, out)
    check_replication(headline_df, out)
    check_vs_buyhold(headline_df, out)
    check_vs_primary(headline_df, out)
    final_verdict(out)

    # Write report
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(out))

    print(f"\n[Saved to {OUTPUT_FILE}]")


if __name__ == '__main__':
    main()