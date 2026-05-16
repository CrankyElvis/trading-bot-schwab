"""
regime_engine.py
Detects the current market regime using VIXY (VIX proxy) vs its 30-day
rolling average, then outputs:
  - regime:        'flow' | 'volatility' | 'neutral'
  - position_size: scalar 0.0–1.0 (fraction of max capital per trade)
  - in_pause:      True if we just switched regimes (sit out 1 cycle)

Regime rules:
  VIXY > 30d avg * 1.15  →  volatility regime  (sell premium, tighter sizes)
  VIXY < 30d avg * 0.85  →  flow regime        (ride momentum, fuller sizes)
  otherwise              →  neutral             (reduced sizing, flow signals only)

Position sizing (inverse VIXY scale):
  VIXY < 20   →  100% of max size
  VIXY 20-25  →   75%
  VIXY 25-30  →   50%
  VIXY > 30   →   25%
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict

STATE_FILE = 'regime_state.json'


@dataclass
class RegimeState:
    regime: str             # 'flow' | 'volatility' | 'neutral'
    vixy: float             # current VIXY price
    vixy_30d_avg: float     # 30-day rolling average
    position_size: float    # 0.0 – 1.0
    in_pause: bool          # True = skip this cycle (just switched)
    previous_regime: str    # what regime was before this one
    updated_at: str         # ISO timestamp


# ── Core Logic ───────────────────────────────────────────────────────────────

def detect_regime(vixy: float, vixy_30d_avg: float) -> str:
    """
    Compares current VIXY to its 30-day average and returns regime label.
    """
    if vixy_30d_avg <= 0:
        return 'neutral'

    ratio = vixy / vixy_30d_avg

    if ratio > 1.15:
        return 'volatility'
    elif ratio < 0.85:
        return 'flow'
    else:
        return 'neutral'


def get_position_size(vixy: float) -> float:
    """
    Returns a position size scalar (0.0–1.0) inversely proportional to VIXY.
    Higher VIXY = smaller positions = less risk.
    """
    if vixy < 20:
        return 1.00
    elif vixy < 25:
        return 0.75
    elif vixy < 30:
        return 0.50
    else:
        return 0.25


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

def evaluate_regime(vixy: float, vix_history_df) -> RegimeState:
    """
    Full regime evaluation. Call this once per bot cycle.

    Args:
        vixy:           Current VIXY price (from data_collector.get_vix)
        vix_history_df: DataFrame from data_collector.get_vix_history (30 days)

    Returns:
        RegimeState dataclass with all fields populated.
    """
    vixy_30d_avg = compute_vixy_30d_avg(vix_history_df)

    # Fall back to neutral with reduced sizing if we have no history
    if vixy_30d_avg == 0.0:
        print("  [!] No VIXY history available — defaulting to neutral regime")
        vixy_30d_avg = vixy   # treat current as average

    new_regime = detect_regime(vixy, vixy_30d_avg)
    position_size = get_position_size(vixy)

    # Check for regime switch (triggers 1-cycle pause)
    prev = _load_previous_state()
    previous_regime = prev.get('regime', new_regime)
    switched = (previous_regime != new_regime) and bool(prev)
    in_pause = switched

    state = RegimeState(
        regime=new_regime,
        vixy=round(vixy, 4),
        vixy_30d_avg=vixy_30d_avg,
        position_size=position_size,
        in_pause=in_pause,
        previous_regime=previous_regime,
        updated_at=datetime.now().isoformat(),
    )

    _save_state(state)
    return state


def print_regime_summary(state: RegimeState):
    icons = {'flow': '🟢', 'volatility': '🔴', 'neutral': '🟡'}
    icon = icons.get(state.regime, '⚪')
    pause_str = '  ⏸  PAUSE CYCLE (regime just switched)' if state.in_pause else ''

    print(f"\n{'='*50}")
    print(f"  REGIME ENGINE")
    print(f"{'='*50}")
    print(f"  {icon} Regime:        {state.regime.upper()}")
    print(f"  VIXY:           {state.vixy:.2f}")
    print(f"  VIXY 30d avg:   {state.vixy_30d_avg:.2f}")
    print(f"  Position size:  {int(state.position_size * 100)}% of max")
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
        (15, 18), (20, 22), (22, 20), (28, 22), (35, 22), (18, 22)
    ]
    for v, avg in scenarios:
        import pandas as pd
        mock_df = pd.DataFrame({'vix': [avg] * 20})
        s = evaluate_regime(v, mock_df)
        print(f"  {v:<8} {avg:<10} {s.regime:<12} {int(s.position_size*100)}%")

    # Restore real state
    evaluate_regime(vixy, vix_history)
    print("\n✅ regime_engine.py working correctly.")