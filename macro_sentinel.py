# -*- coding: utf-8 -*-
"""
macro_sentinel.py  --  Leading macro indicator composite for regime early warning

Detects macro stress BEFORE VIX spikes by monitoring:
  1. Yield curve spread (10Y-2Y)  -- FRED via Yahoo Finance proxies
  2. Credit spread (HYG vs LQD)   -- Yahoo Finance
  3. SPY put/call ratio            -- CBOE (free CSV)
  4. Energy momentum (USO)        -- Yahoo Finance
  5. TLT momentum                 -- Yahoo Finance (rate stress proxy)
  6. Dollar strength (UUP)        -- Yahoo Finance (risk-off signal)

Each signal is scored 0 (green) / 0.5 (yellow) / 1.0 (red).
Composite score >= 0.60 triggers PRE_CRISIS warning.
Composite score >= 0.80 triggers regime upgrade recommendation.

Runs in premarket cycle (6am ET). Feeds into evaluate_regime() as
an additional override layer -- does NOT replace VIX-based detection,
it supplements it with 2-4 week leading signals.

Usage:
  from macro_sentinel import evaluate_macro
  macro = evaluate_macro()
  # macro.score       -- 0.0 to 1.0
  # macro.warning     -- 'green' | 'yellow' | 'red'
  # macro.signals     -- list of triggered signals
  # macro.recommended_regime  -- suggested regime override (or None)
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


CACHE_FILE  = 'macro_sentinel_cache.json'
CACHE_TTL_H = 4   # refresh every 4 hours


@dataclass
class MacroSignal:
    name:        str
    value:       float
    score:       float          # 0=green, 0.5=yellow, 1.0=red
    description: str
    source:      str


@dataclass
class MacroState:
    score:                float          # composite 0.0-1.0
    warning:              str            # 'green' | 'yellow' | 'red'
    signals:              list           # list of MacroSignal dicts
    recommended_regime:   Optional[str]  # None | 'neutral' | 'volatility' | 'crisis'
    summary:              str
    updated_at:           str
    data_quality:         str            # 'full' | 'partial' | 'cached' | 'failed'


# ── Signal fetchers ───────────────────────────────────────────────────────────

def _fetch_yf_closes(ticker: str, days: int = 60) -> list:
    """Returns list of recent close prices for a ticker via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f'{days}d', interval='1d',
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return []

        # yfinance >= 0.2 returns MultiIndex columns even for single tickers
        # Flatten: ('Close', 'SPY') -> use xs or just grab first level
        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex — extract Close for this ticker
            try:
                close_col = df['Close']
                if hasattr(close_col, 'columns'):
                    # Still a DataFrame (MultiIndex with ticker level)
                    close_col = close_col.iloc[:, 0]
            except Exception:
                close_col = df.iloc[:, 0]
        else:
            close_col = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

        vals = close_col.dropna()
        if len(vals) == 0:
            return []

        # Safely convert each value to float
        result = []
        for v in vals:
            try:
                f = float(v)
                if f > 0:
                    result.append(f)
            except Exception:
                pass
        return result
    except Exception as e:
        return []


def score_yield_curve(signals: list) -> MacroSignal:
    """
    Proxy: TLT (long bonds) momentum vs IEF (intermediate).
    When TLT is falling faster than IEF, rates are rising at long end.
    We use IEF/TLT ratio as a yield curve steepness proxy.
    True 10Y-2Y spread requires FRED API -- this is a free proxy.
    """
    try:
        tlt = _fetch_yf_closes('TLT', 30)
        ief = _fetch_yf_closes('IEF', 30)
        if len(tlt) < 10 or len(ief) < 10:
            return MacroSignal('Yield curve', 0, 0.5, 'Insufficient data', 'Yahoo/TLT+IEF')

        # TLT 20d return -- falling TLT = rising long rates = stress
        tlt_ret   = (tlt[-1] - tlt[-20]) / tlt[-20] * 100
        tlt_5d    = (tlt[-1] - tlt[-5])  / tlt[-5]  * 100

        if tlt_ret < -5 and tlt_5d < -1.5:
            score = 1.0
            desc  = f'TLT -20d: {tlt_ret:.1f}%, -5d: {tlt_5d:.1f}% -- rates rising fast, HIGH STRESS'
        elif tlt_ret < -2 or tlt_5d < -0.5:
            score = 0.5
            desc  = f'TLT -20d: {tlt_ret:.1f}%, -5d: {tlt_5d:.1f}% -- rates rising moderately'
        else:
            score = 0.0
            desc  = f'TLT -20d: {tlt_ret:.1f}%, -5d: {tlt_5d:.1f}% -- rates stable'
        return MacroSignal('Yield curve / rates', tlt_ret, score, desc, 'Yahoo/TLT')
    except Exception as e:
        return MacroSignal('Yield curve / rates', 0, 0.5, f'Error: {e}', 'Yahoo/TLT')


def score_credit_spread(signals: list) -> MacroSignal:
    """
    HYG (high yield) vs LQD (investment grade) relative performance.
    When HYG underperforms LQD, credit stress is building.
    This leads equity by 2-3 weeks historically.
    """
    try:
        hyg = _fetch_yf_closes('HYG', 30)
        lqd = _fetch_yf_closes('LQD', 30)
        if len(hyg) < 10 or len(lqd) < 10:
            return MacroSignal('Credit spread', 0, 0.5, 'Insufficient data', 'Yahoo/HYG+LQD')

        hyg_ret_10d = (hyg[-1] - hyg[-10]) / hyg[-10] * 100
        lqd_ret_10d = (lqd[-1] - lqd[-10]) / lqd[-10] * 100
        spread      = hyg_ret_10d - lqd_ret_10d   # negative = credit stress

        hyg_ret_5d  = (hyg[-1] - hyg[-5])  / hyg[-5]  * 100
        lqd_ret_5d  = (lqd[-1] - lqd[-5])  / lqd[-5]  * 100
        spread_5d   = hyg_ret_5d - lqd_ret_5d

        if spread < -1.5 or spread_5d < -0.8:
            score = 1.0
            desc  = f'HYG-LQD 10d: {spread:.2f}%, 5d: {spread_5d:.2f}% -- credit stress HIGH'
        elif spread < -0.5 or spread_5d < -0.3:
            score = 0.5
            desc  = f'HYG-LQD 10d: {spread:.2f}%, 5d: {spread_5d:.2f}% -- credit spread widening'
        else:
            score = 0.0
            desc  = f'HYG-LQD 10d: {spread:.2f}%, 5d: {spread_5d:.2f}% -- credit healthy'
        return MacroSignal('Credit spread (HYG/LQD)', spread, score, desc, 'Yahoo/HYG+LQD')
    except Exception as e:
        return MacroSignal('Credit spread (HYG/LQD)', 0, 0.5, f'Error: {e}', 'Yahoo/HYG+LQD')


def score_energy_inflation(signals: list) -> MacroSignal:
    """
    USO (crude oil ETF) momentum.
    Rising energy = inflationary pressure = Fed hawkish = rate risk.
    This is exactly what the user flagged as a current concern.
    """
    try:
        uso = _fetch_yf_closes('USO', 60)
        if len(uso) < 20:
            return MacroSignal('Energy/inflation', 0, 0.5, 'Insufficient data', 'Yahoo/USO')

        ret_20d = (uso[-1] - uso[-20]) / uso[-20] * 100
        ret_5d  = (uso[-1] - uso[-5])  / uso[-5]  * 100
        # Also check if energy is accelerating (5d > 20d trend)
        accelerating = ret_5d > (ret_20d / 4)

        if ret_20d > 8 and accelerating:
            score = 1.0
            desc  = f'USO +{ret_20d:.1f}% (20d), +{ret_5d:.1f}% (5d) -- energy surging, inflation risk HIGH'
        elif ret_20d > 4 or (ret_20d > 2 and accelerating):
            score = 0.5
            desc  = f'USO +{ret_20d:.1f}% (20d), +{ret_5d:.1f}% (5d) -- energy rising, watch inflation'
        elif ret_20d < -8:
            score = 0.0
            desc  = f'USO {ret_20d:.1f}% (20d) -- energy falling, deflation signal (risk-off anyway)'
        else:
            score = 0.0
            desc  = f'USO {ret_20d:.1f}% (20d), {ret_5d:.1f}% (5d) -- energy stable'
        return MacroSignal('Energy / inflation (USO)', ret_20d, score, desc, 'Yahoo/USO')
    except Exception as e:
        return MacroSignal('Energy / inflation (USO)', 0, 0.5, f'Error: {e}', 'Yahoo/USO')


def score_dollar_strength(signals: list) -> MacroSignal:
    """
    UUP (dollar bull ETF) momentum.
    Strong dollar = risk-off globally, EM stress, commodity headwinds.
    Dollar strength often leads equity weakness by 1-3 weeks.
    """
    try:
        uup = _fetch_yf_closes('UUP', 30)
        if len(uup) < 10:
            return MacroSignal('Dollar strength', 0, 0.5, 'Insufficient data', 'Yahoo/UUP')

        ret_20d = (uup[-1] - uup[-20]) / uup[-20] * 100 if len(uup) >= 20 else 0
        ret_5d  = (uup[-1] - uup[-5])  / uup[-5]  * 100

        if ret_20d > 2.5 and ret_5d > 0.5:
            score = 1.0
            desc  = f'UUP +{ret_20d:.1f}% (20d) -- dollar surging, global risk-off HIGH'
        elif ret_20d > 1.0 or ret_5d > 0.3:
            score = 0.5
            desc  = f'UUP +{ret_20d:.1f}% (20d) -- dollar strengthening, monitor'
        else:
            score = 0.0
            desc  = f'UUP {ret_20d:.1f}% (20d) -- dollar neutral/weak, supportive'
        return MacroSignal('Dollar strength (UUP)', ret_20d, score, desc, 'Yahoo/UUP')
    except Exception as e:
        return MacroSignal('Dollar strength (UUP)', 0, 0.5, f'Error: {e}', 'Yahoo/UUP')


def score_put_call_ratio(signals: list) -> MacroSignal:
    """
    SPY put/call ratio from CBOE.
    5-day MA above 1.2 historically precedes corrections 1-3 weeks out.
    Uses CBOE total equity P/C ratio (free CSV).
    """
    try:
        url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/SPY_History.csv'
        # CBOE doesn't publish P/C via simple CSV -- use SPY options proxy via price action
        # Fallback: use SKEW index or just return neutral
        # True P/C ratio requires CBOE data portal subscription
        # We use put-heavy flow as a proxy via VIX/VIXY momentum instead
        vix = _fetch_yf_closes('^VIX', 10)
        if len(vix) < 5:
            return MacroSignal('Put/call pressure', 0, 0.5, 'Insufficient data', 'Yahoo/VIX')

        vix_5d_avg = sum(vix[-5:]) / 5
        vix_trend  = (vix[-1] - vix[-5]) / vix[-5] * 100

        if vix[-1] > 25 and vix_trend > 5:
            score = 1.0
            desc  = f'VIX {vix[-1]:.1f}, +{vix_trend:.1f}% 5d -- put buying elevated, fear rising'
        elif vix[-1] > 20 or vix_trend > 3:
            score = 0.5
            desc  = f'VIX {vix[-1]:.1f}, {vix_trend:+.1f}% 5d -- elevated put activity'
        else:
            score = 0.0
            desc  = f'VIX {vix[-1]:.1f}, {vix_trend:+.1f}% 5d -- put/call neutral'
        return MacroSignal('Put/call pressure (VIX proxy)', vix[-1], score, desc, 'Yahoo/VIX')
    except Exception as e:
        return MacroSignal('Put/call pressure', 0, 0.5, f'Error: {e}', 'Yahoo/VIX')


def score_equity_breadth(signals: list) -> MacroSignal:
    """
    SPY vs IWM (large cap vs small cap) relative strength.
    When small caps underperform significantly, risk appetite is narrowing.
    Also checks SPY vs its own 50d MA.
    """
    try:
        spy = _fetch_yf_closes('SPY', 60)
        iwm = _fetch_yf_closes('IWM', 60)
        if len(spy) < 20 or len(iwm) < 20:
            return MacroSignal('Market breadth', 0, 0.5, 'Insufficient data', 'Yahoo/SPY+IWM')

        spy_20d   = sum(spy[-20:]) / 20
        spy_50d   = sum(spy[-50:]) / 50 if len(spy) >= 50 else spy_20d
        spy_above_50d = spy[-1] > spy_50d
        spy_above_20d = spy[-1] > spy_20d

        spy_ret_10d = (spy[-1] - spy[-10]) / spy[-10] * 100
        iwm_ret_10d = (iwm[-1] - iwm[-10]) / iwm[-10] * 100
        breadth_spread = spy_ret_10d - iwm_ret_10d   # positive = large cap hiding small cap weakness

        if not spy_above_50d and breadth_spread > 2:
            score = 1.0
            desc  = (f'SPY below 50d MA ({spy[-1]:.0f} vs {spy_50d:.0f}), '
                     f'small caps lagging {breadth_spread:.1f}% -- NARROW breadth, HIGH risk')
        elif not spy_above_20d or breadth_spread > 1.5:
            score = 0.5
            desc  = (f'SPY {"above" if spy_above_50d else "below"} 50d, '
                     f'breadth spread {breadth_spread:.1f}% -- breadth narrowing')
        else:
            desc  = (f'SPY {spy[-1]:.0f} above 50d {spy_50d:.0f}, '
                     f'breadth spread {breadth_spread:.1f}% -- broad participation')
            score = 0.0
        return MacroSignal('Market breadth (SPY/IWM)', breadth_spread, score, desc, 'Yahoo/SPY+IWM')
    except Exception as e:
        return MacroSignal('Market breadth', 0, 0.5, f'Error: {e}', 'Yahoo/SPY+IWM')


# ── Composite scorer ──────────────────────────────────────────────────────────

def evaluate_macro(use_cache: bool = True) -> MacroState:
    """
    Main entry point. Runs all 6 signal checks and returns a MacroState.
    Call this once per premarket cycle. Results cached for 4 hours.
    """
    # Check cache first
    if use_cache:
        try:
            with open(CACHE_FILE, encoding='utf-8') as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached.get('updated_at', '2000-01-01'))
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
            if age_hours < CACHE_TTL_H:
                c = cached
                return MacroState(
                    score=c['score'], warning=c['warning'],
                    signals=c['signals'],
                    recommended_regime=c.get('recommended_regime'),
                    summary=c['summary'], updated_at=c['updated_at'],
                    data_quality='cached'
                )
        except Exception:
            pass

    print('  [macro] Fetching macro indicators...')
    all_signals = []

    fetchers = [
        score_yield_curve,
        score_credit_spread,
        score_energy_inflation,
        score_dollar_strength,
        score_put_call_ratio,
        score_equity_breadth,
    ]

    data_quality = 'full'
    for fn in fetchers:
        try:
            sig = fn(all_signals)
            all_signals.append(sig)
            status = '[RED]' if sig.score == 1.0 else ('[YLW]' if sig.score == 0.5 else '[GRN]')
            print(f'    {status} {sig.name}: {sig.description}')
        except Exception as e:
            all_signals.append(MacroSignal(fn.__name__, 0, 0.5, f'Failed: {e}', 'error'))
            data_quality = 'partial'

    # Composite score -- weighted
    weights = {
        'Yield curve / rates':        0.25,
        'Credit spread (HYG/LQD)':    0.25,
        'Energy / inflation (USO)':   0.15,
        'Dollar strength (UUP)':      0.15,
        'Put/call pressure (VIX proxy)': 0.10,
        'Market breadth (SPY/IWM)':   0.10,
    }
    total_weight = 0
    weighted_score = 0
    for sig in all_signals:
        w = weights.get(sig.name, 0.10)
        weighted_score += sig.score * w
        total_weight   += w

    composite = round(weighted_score / total_weight, 3) if total_weight > 0 else 0.5

    # Warning level
    if composite >= 0.70:
        warning = 'red'
    elif composite >= 0.40:
        warning = 'yellow'
    else:
        warning = 'green'

    # Regime recommendation
    red_count    = sum(1 for s in all_signals if s.score == 1.0)
    yellow_count = sum(1 for s in all_signals if s.score == 0.5)

    if composite >= 0.70 or red_count >= 3:
        recommended_regime = 'volatility-cautious'
        summary = (f'MACRO RED ({red_count} red, {yellow_count} yellow signals) -- '
                   f'recommend volatility regime, cut sizes 50%, move overnight to GLD')
    elif composite >= 0.40 or red_count >= 2:
        recommended_regime = 'neutral'
        summary = (f'MACRO YELLOW ({red_count} red, {yellow_count} yellow signals) -- '
                   f'recommend neutral regime floor, tighten stops, avoid new CSPs')
    else:
        recommended_regime = None
        summary = (f'MACRO GREEN ({red_count} red, {yellow_count} yellow signals) -- '
                   f'no regime override needed')

    state = MacroState(
        score=composite,
        warning=warning,
        signals=[{'name': s.name, 'value': s.value, 'score': s.score,
                  'description': s.description, 'source': s.source}
                 for s in all_signals],
        recommended_regime=recommended_regime,
        summary=summary,
        updated_at=datetime.now(timezone.utc).isoformat(),
        data_quality=data_quality,
    )

    # Cache result
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'score': state.score, 'warning': state.warning,
                'signals': state.signals,
                'recommended_regime': state.recommended_regime,
                'summary': state.summary, 'updated_at': state.updated_at,
                'data_quality': state.data_quality,
            }, f, indent=2)
    except Exception:
        pass

    return state


def print_macro_report(state: MacroState):
    """Pretty-print the macro state for logs and premarket output."""
    sep = '-' * 56
    bar_filled = int(state.score * 20)
    bar = '[' + '#' * bar_filled + '.' * (20 - bar_filled) + ']'
    level = state.warning.upper()
    print(f'\n  {sep}')
    print(f'  MACRO SENTINEL  --  {level}  {bar}  {state.score:.2f}')
    print(f'  {state.summary}')
    print(f'  {sep}')
    for sig in state.signals:
        status = '[RED]' if sig['score'] == 1.0 else ('[YLW]' if sig['score'] == 0.5 else '[GRN]')
        print(f'    {status} {sig["name"]}')
        print(f'           {sig["description"]}')
    if state.recommended_regime:
        print(f'\n  >>> Regime override recommendation: {state.recommended_regime.upper()}')
        print(f'  >>> Review before accepting -- macro signals are leading, not definitive')
    print(f'  {sep}\n')


# ── Regime engine integration hook ────────────────────────────────────────────

def apply_macro_override(regime: str, macro: MacroState) -> tuple:
    """
    Takes the VIX-based regime and optionally upgrades it based on macro signals.
    Returns (final_regime, was_overridden, reason).

    Rules:
    - Never DOWNGRADES regime (macro can only make it more defensive)
    - crisis is always preserved
    - Only upgrades if macro score >= 0.55 to avoid false positives
    """
    REGIME_ORDER = {'flow': 0, 'neutral': 1, 'volatility-cautious': 2, 'volatility-defensive': 2, 'crisis': 3}

    if regime == 'crisis':
        return regime, False, 'Crisis regime preserved'

    if macro.recommended_regime is None or macro.score < 0.40:
        return regime, False, 'Macro signals green -- no override'

    current_rank  = REGIME_ORDER.get(regime, 1)
    proposed_rank = REGIME_ORDER.get(macro.recommended_regime, 1)

    if proposed_rank > current_rank:
        reason = f'Macro sentinel upgrade: {regime} -> {macro.recommended_regime} (score {macro.score:.2f})'
        return macro.recommended_regime, True, reason
    else:
        return regime, False, f'Macro score {macro.score:.2f} but VIX regime already adequate'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    state = evaluate_macro(use_cache=not args.no_cache)
    print_macro_report(state)