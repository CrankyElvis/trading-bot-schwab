"""
regime_engine.py
Detects the current market regime using VIXY (VIX proxy) vs its 30-day
rolling average, then outputs:
  - regime:        'flow' | 'neutral' | 'volatility-cautious' | 'volatility-defensive' | 'crisis'
  - position_size: scalar 0.0–1.0 (fraction of max capital per trade)
  - in_pause:      True if we just switched regimes (sit out 1 cycle)

Regime rules:
  VIXY > 40 (absolute)           →  crisis     (cash/inverse only, 10% size)
  VIXY ratio 1.15-1.30 + SPY>EMA →  volatility-cautious (reduced sizing)
  VIXY ratio > 1.30 OR SPY<EMA    →  volatility-defensive (no new entries)
  VIXY < 30d avg * 0.85          →  flow       (ride momentum, full sizes)
  otherwise                      →  neutral    (reduced sizing, flow signals only)

Position sizing (inverse VIXY scale):
  VIXY < 20   →  100% of max size
  VIXY 20-25  →   75%
  VIXY 25-30  →   50%
  VIXY 30-40  →   25%
  VIXY > 40   →   10% (crisis — capital preservation)
"""

import json
import os
from datetime import datetime
from macro_sentinel import evaluate_macro, apply_macro_override, print_macro_report
from dataclasses import dataclass, asdict

STATE_FILE = 'regime_state.json'


@dataclass
class RegimeState:
    regime: str             # 'flow' | 'neutral' | 'volatility-cautious' | 'volatility-defensive' | 'crisis'
    vixy: float             # current VIXY price
    vixy_30d_avg: float     # 30-day rolling average
    position_size: float    # 0.0 – 1.0
    in_pause: bool          # True = skip this cycle (just switched)
    previous_regime: str    # what regime was before this one
    updated_at: str         # ISO timestamp
    term_signal: str        # VIX term structure signal: contango/flat/backwardation/inversion
    halt_entries: bool      # True = term structure says halt new entries
    macro_score: float = 0.0   # macro sentinel composite score 0.0-1.0
    macro_warning: str = 'green'  # 'green' | 'yellow' | 'red'
    macro_override: bool = False  # True if macro upgraded the regime


# ── Core Logic ───────────────────────────────────────────────────────────────

def detect_regime(vixy: float, vixy_30d_avg: float,
                  spy_price: float = 0.0, spy_ema50: float = 0.0) -> str:
    """
    Compares current VIXY level and ratio to 30-day average.
    Crisis takes priority over all other signals — absolute VIXY > 40.

    Volatility split (mirrors backtester #26):
      volatility-defensive: ratio > 1.30 OR SPY below 50d EMA → no new entries
      volatility-cautious:  ratio 1.15-1.30 AND SPY above 50d EMA → reduced sizing
    """
    if vixy_30d_avg <= 0:
        return 'neutral'

    # Crisis: absolute VIXY spike regardless of ratio
    if vixy > 40:
        return 'crisis'

    ratio = vixy / vixy_30d_avg

    if ratio > 1.15:
        spy_below_ema = (spy_price > 0 and spy_ema50 > 0 and spy_price < spy_ema50)
        if ratio > 1.30 or spy_below_ema:
            return 'volatility-defensive'
        return 'volatility-cautious'
    elif ratio < 0.85:
        return 'flow'
    else:
        return 'neutral'


def get_position_size(vixy: float) -> float:
    """
    Returns a position size scalar (0.0–1.0) inversely proportional to VIXY.
    Higher VIXY = smaller positions = less risk.
    Crisis mode (VIXY > 40) drops to 10% — capital preservation only.
    """
    if vixy > 40:
        return 0.10   # crisis — near-cash mode
    elif vixy >= 30:
        return 0.25
    elif vixy >= 25:
        return 0.50
    elif vixy >= 20:
        return 0.75
    else:
        return 1.00


def compute_vixy_30d_avg(vix_history_df) -> float:
    """
    Computes the 30-day rolling average of VIXY closes.
    Expects a DataFrame with a 'vix' column (from data_collector.get_vix_history).
    Returns 0.0 if insufficient data.
    """
    if vix_history_df is None or vix_history_df.empty:
        return 0.0
    col = 'vix' if 'vix' in vix_history_df.columns else vix_history_df.columns[-1]
    closes = vix_history_df[col].dropna()
    if len(closes) < 5:   # need at least 5 days to be meaningful
        return 0.0
    return round(float(closes.mean()), 4)


# ── State Persistence ────────────────────────────────────────────────────────

def _load_previous_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: RegimeState):
    with open(STATE_FILE, 'w') as f:
        json.dump(asdict(state), f, indent=2)


# ── Main Entry Point ─────────────────────────────────────────────────────────

def evaluate_regime(vixy: float, vix_history_df,
                    term_structure: dict = None) -> RegimeState:
    """
    Full regime evaluation. Call this once per bot cycle.

    Args:
        vixy:           Current VIXY price (from data_collector.get_vix)
        vix_history_df: DataFrame from data_collector.get_vix_history (30 days)
        term_structure: Optional dict from data_collector.get_vix_term_structure()
                        Used to upgrade neutral->volatility on backwardation signal

    Returns:
        RegimeState dataclass with all fields populated.
    """
    vixy_30d_avg = compute_vixy_30d_avg(vix_history_df)

    # Fall back to neutral with reduced sizing if we have no history
    if vixy_30d_avg == 0.0:
        print("  [!] No VIXY history available — defaulting to neutral regime")
        vixy_30d_avg = vixy   # treat current as average

    # SPY 50d EMA for volatility split (cautious vs defensive)
    spy_price = 0.0
    spy_ema50 = 0.0
    try:
        import yfinance as yf
        spy_df = yf.download('SPY', period='60d', interval='1d',
                              auto_adjust=True, progress=False, threads=False)
        if spy_df is not None and not spy_df.empty:
            closes = spy_df['Close']
            if hasattr(closes, 'columns'):
                closes = closes.iloc[:, 0]
            closes = closes.dropna()
            if len(closes) >= 10:
                spy_price = float(closes.iloc[-1])
                spy_ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    except Exception:
        pass  # SPY unavailable — detect_regime defaults to cautious (safer)

    new_regime = detect_regime(vixy, vixy_30d_avg, spy_price, spy_ema50)

    # VIX term structure override: backwardation = upgrade to volatility regime
    if term_structure and new_regime not in ('crisis',):
        ts_signal = term_structure.get('signal', 'flat')
        if ts_signal == 'backwardation' and new_regime == 'flow':
            new_regime = 'neutral'
            print(f"  ⚠️  Term structure backwardation — upgrading flow→neutral")
        elif ts_signal in ('backwardation', 'inversion') and new_regime == 'neutral':
            new_regime = 'volatility-cautious'
            print(f"  ⚠️  Term structure {ts_signal} — upgrading neutral→volatility-cautious")

    position_size = get_position_size(vixy)

    # Check for regime switch (triggers 1-cycle pause)
    prev = _load_previous_state()
    previous_regime = prev.get('regime', new_regime)
    switched = (previous_regime != new_regime) and bool(prev)
    in_pause = switched

    ts_signal    = term_structure.get('signal', 'flat') if term_structure else 'flat'
    halt_entries = term_structure.get('halt_entries', False) if term_structure else False

    state = RegimeState(
        regime=new_regime,
        vixy=round(vixy, 4),
        vixy_30d_avg=vixy_30d_avg,
        position_size=position_size,
        in_pause=in_pause,
        previous_regime=previous_regime,
        updated_at=datetime.now().isoformat(),
        term_signal=ts_signal,
        halt_entries=halt_entries,
    )


    # ── Macro sentinel override ──────────────────────────────────────────
    try:
        macro = evaluate_macro(use_cache=True)
        final_regime, was_overridden, reason = apply_macro_override(state.regime, macro)
        if was_overridden:
            print(f'  [macro] {reason}')
            state.regime        = final_regime
            state.position_size = get_position_size(
                max(state.vixy, 25.0 if final_regime in ('volatility-cautious', 'volatility-defensive') else state.vixy))
        state.macro_score    = macro.score
        state.macro_warning  = macro.warning
        state.macro_override = was_overridden
    except Exception as e:
        print(f'  [macro] Sentinel unavailable: {e}')

    _save_state(state)
    return state


def print_regime_summary(state: RegimeState):
    icons = {'flow': '🟢', 'neutral': '🟡', 'crisis': '🚨', 'volatility-cautious': '🟠', 'volatility-defensive': '🔴'}
    icon = icons.get(state.regime, '⚪')
    pause_str = '  ⏸  PAUSE CYCLE (regime just switched)' if state.in_pause else ''

    print(f"\n{'='*50}")
    print(f"  REGIME ENGINE")
    print(f"{'='*50}")
    print(f"  {icon} Regime:        {state.regime.upper()}")
    print(f"  VIXY:           {state.vixy:.2f}")
    print(f"  VIXY 30d avg:   {state.vixy_30d_avg:.2f}")
    print(f"  Position size:  {int(state.position_size * 100)}% of max")
    print(f"  Term structure: {state.term_signal.upper()}{'  🛑 HALT ENTRIES' if state.halt_entries else ''}")
    if state.in_pause:
        print(f"  {pause_str}")
    print(f"  Updated:        {state.updated_at[:19]}")
    print(f"{'='*50}\n")


# ── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from auth import authenticate
    from data_collector import get_vix, get_vix_history

    print("🔌 Authenticating...")
    client, paper = authenticate()
    paper = True  # assume paper mode
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    print("📡 Fetching VIXY data...")
    vixy = get_vix(client)
    vix_history = get_vix_history(client, days=30)

    print(f"  VIXY current:  {vixy:.2f}")
    print(f"  History rows:  {len(vix_history)}")

    state = evaluate_regime(vixy, vix_history)
    print_regime_summary(state)

    # Simulate a few scenarios
    print("📊 Scenario table:")
    print(f"  {'VIXY':<8} {'30d avg':<10} {'Regime':<12} {'Size'}")
    print(f"  {'-'*40}")
    scenarios = [
        (15, 18), (20, 22), (22, 20), (28, 22), (35, 22), (18, 22), (42, 28), (55, 30)
    ]
    for v, avg in scenarios:
        import pandas as pd
        mock_df = pd.DataFrame({'vix': [avg] * 20})
        s = evaluate_regime(v, mock_df)
        print(f"  {v:<8} {avg:<10} {s.regime:<12} {int(s.position_size*100)}%")

    # Restore real state
    evaluate_regime(vixy, vix_history)
    print("\n✅ regime_engine.py working correctly.")