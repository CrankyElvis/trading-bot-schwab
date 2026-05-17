"""
risk_manager.py
Runs 6 pre-trade checks before any position is opened.
All 6 must pass (return False for is_blocked) for a trade to proceed.

Blockers:
  1. Drawdown gate      — portfolio down > 10% from starting cash
  2. Max positions      — already at 5 open positions (2 in crisis)
  3. Earnings blocker   — earnings within 48 hours
  4. FDA blocker        — biotech/pharma catalyst risk
  5. Macro blocker      — major macro event within 24 hours
  6. Market tide        — SPY in a bearish trend (5d EMA < 20d EMA)

Crisis regime override:
  In crisis regime, max positions drops to 2 and all checks still apply.
"""

import os
import json
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

UW_API_KEY  = os.getenv('UW_API_KEY')
UW_BASE_URL = 'https://api.unusualwhales.com/api'

MAX_POSITIONS    = 5
CRISIS_MAX_POS   = 2
MAX_DRAWDOWN_PCT = 0.10
EARNINGS_WINDOW_H = 48
MACRO_WINDOW_H    = 24

MACRO_KEYWORDS = [
    'fomc', 'federal reserve', 'fed meeting', 'interest rate decision',
    'cpi', 'consumer price', 'ppi', 'producer price',
    'jobs report', 'nonfarm payroll', 'unemployment',
    'gdp', 'gross domestic product', 'jackson hole', 'powell',
]

FDA_SECTORS = [
    'biotechnology', 'pharmaceutical', 'drug', 'biotech', 'life sciences',
]


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class RiskCheckResult:
    symbol: str
    passed: bool
    blockers: list = field(default_factory=list)
    details: dict  = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed:
            return f"✅ {self.symbol} — CLEAR TO TRADE"
        return f"🚫 {self.symbol} — BLOCKED by: {', '.join(self.blockers)}"


# ── UW Helper ─────────────────────────────────────────────────────────────────

def _uw(endpoint: str, params: dict = None) -> dict:
    try:
        headers = {'Authorization': f'Bearer {UW_API_KEY}'}
        r = requests.get(f"{UW_BASE_URL}{endpoint}", headers=headers,
                         params=params or {}, timeout=8)
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception:
        return {}


# ── Blocker 1: Drawdown Gate ──────────────────────────────────────────────────

def check_drawdown(portfolio: dict) -> tuple[bool, dict]:
    try:
        starting = float(portfolio.get('starting_cash', 0))
        if starting <= 0:
            return False, {}
        cash = float(portfolio.get('cash', starting))
        positions = portfolio.get('positions', {})
        pos_value = sum(
            p.get('quantity', 0) * p.get('current_price', p.get('avg_price', 0))
            for p in positions.values()
        )
        total = cash + pos_value
        drawdown = (starting - total) / starting
        return drawdown >= MAX_DRAWDOWN_PCT, {
            'starting':     f'${starting:,.2f}',
            'current':      f'${total:,.2f}',
            'drawdown_pct': f'{drawdown*100:.2f}%',
            'threshold':    f'{MAX_DRAWDOWN_PCT*100:.0f}%',
        }
    except Exception as e:
        return False, {'error': str(e)}


# ── Blocker 2: Max Positions ──────────────────────────────────────────────────

def check_max_positions(portfolio: dict, crisis: bool = False) -> tuple[bool, dict]:
    limit = CRISIS_MAX_POS if crisis else MAX_POSITIONS
    positions = portfolio.get('positions', {})
    count = len([p for p in positions.values() if p.get('quantity', 0) > 0])
    return count >= limit, {
        'open':  count,
        'limit': limit,
        'mode':  'crisis' if crisis else 'normal',
    }


# ── Blocker 3: Earnings ───────────────────────────────────────────────────────

def check_earnings(symbol: str) -> tuple[bool, dict]:
    """
    Uses Finnhub as primary earnings source (accurate and free).
    Falls back to UW if Finnhub returns nothing.
    """
    from data_collector import has_earnings_soon
    blocked, date_str = has_earnings_soon(symbol, hours=EARNINGS_WINDOW_H)
    if blocked:
        return True, {'earnings_date': date_str, 'source': 'finnhub'}
    if date_str == '' and not blocked:
        # Try UW as fallback
        data  = _uw(f'/earnings/{symbol}')
        items = data.get('data', [])
        if items:
            now    = datetime.now(timezone.utc)
            window = now + timedelta(hours=EARNINGS_WINDOW_H)
            for item in items:
                date_str = item.get('date') or item.get('report_date') or ''
                if not date_str: continue
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if now <= dt <= window:
                        hours_away = (dt - now).total_seconds() / 3600
                        return True, {'earnings_date': date_str,
                                      'hours_away': f'{hours_away:.1f}h',
                                      'source': 'uw'}
                except Exception:
                    continue
            return False, {'next_earnings': items[0].get('date', 'unknown'), 'source': 'uw'}
    return False, {'note': 'No earnings in window', 'source': 'finnhub'}


# ── Blocker 4: FDA / Biotech ──────────────────────────────────────────────────

def check_fda(symbol: str) -> tuple[bool, dict]:
    data    = _uw(f'/companies/{symbol}/profile')
    profile = data.get('data', {})
    if not profile:
        return False, {'note': 'No profile data'}

    sector   = (profile.get('sector')   or '').lower()
    industry = (profile.get('industry') or '').lower()
    is_fda   = any(k in sector or k in industry for k in FDA_SECTORS)

    if is_fda:
        return True, {
            'sector':   sector,
            'industry': industry,
            'note':     'Biotech/pharma — FDA catalyst risk, skipping',
        }
    return False, {'sector': sector, 'industry': industry}


# ── Blocker 5: Macro Events ───────────────────────────────────────────────────

def check_macro() -> tuple[bool, dict]:
    now  = datetime.now()
    dow  = now.weekday()   # 0=Mon … 6=Sun
    dom  = now.day
    hour = now.hour

    if dow == 2 and hour < MACRO_WINDOW_H:
        return True, {'note': 'Wednesday — potential FOMC day'}
    if dow == 4 and dom <= 7:
        return True, {'note': 'First Friday — likely NFP jobs report day'}
    if dow in (1, 2) and 8 <= dom <= 14:
        return True, {'note': 'Mid-month Tue/Wed — potential CPI release day'}

    return False, {'day': now.strftime('%A'), 'date': now.strftime('%Y-%m-%d')}


# ── Blocker 6: Market Tide ────────────────────────────────────────────────────

def check_market_tide(spy_history_df) -> tuple[bool, dict]:
    try:
        if spy_history_df is None or spy_history_df.empty or len(spy_history_df) < 20:
            return False, {'note': 'Insufficient SPY history'}

        closes  = spy_history_df['close']
        ema5    = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
        ema20   = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        bearish = ema5 < ema20

        return bearish, {
            'spy_ema5':  f'${ema5:.2f}',
            'spy_ema20': f'${ema20:.2f}',
            'tide':      '🔴 bearish' if bearish else '🟢 bullish',
        }
    except Exception as e:
        return False, {'error': str(e)}


# ── Blocker 7: Momentum Confirmation ─────────────────────────────────────────

def check_momentum(symbol: str, price_history: dict) -> tuple[bool, dict]:
    """
    Blocked if price is below its 10-day EMA — no catching falling knives.
    Only enter when price has upward momentum confirmed.
    """
    try:
        df = price_history.get(symbol, None)
        if df is None or df.empty or len(df) < 11:
            return False, {'note': 'Insufficient history for momentum check'}

        closes = df['close']
        ema10  = closes.ewm(span=10, adjust=False).mean().iloc[-1]
        last   = closes.iloc[-1]
        above  = last >= ema10

        return not above, {
            'last_price': round(float(last), 2),
            'ema10':      round(float(ema10), 2),
            'momentum':   '✅ above EMA10' if above else '❌ below EMA10',
        }
    except Exception as e:
        return False, {'error': str(e)}


# ── Blocker 8: Volatility Regime Entry Block ─────────────────────────────────

def check_volatility_regime(regime: str, term_structure: dict = None) -> tuple:
    if regime == 'volatility':
        return True, {
            'reason': 'Volatility regime — no new entries, exits only',
            'regime': regime,
        }
    if term_structure and term_structure.get('halt_entries', False):
        ts_signal = term_structure.get('signal', 'flat')
        return True, {
            'reason':  f'VIX term structure {ts_signal} — halting entries',
            'vix9d':   term_structure.get('vix9d', 0),
            'vix3m':   term_structure.get('vix3m', 0),
            'spread':  term_structure.get('spread', 0),
        }
    return False, {
        'regime':      regime,
        'term_signal': term_structure.get('signal', 'flat') if term_structure else 'n/a',
    }


# ── Blocker 9: Rolling 30-Day Circuit Breaker ─────────────────────────────────

def check_circuit_breaker(portfolio: dict, threshold: float = 0.10) -> tuple:
    try:
        trade_log = portfolio.get('trade_log', [])
        if not trade_log:
            return False, {'note': 'No trade history'}
        from datetime import datetime, timezone, timedelta
        now      = datetime.now(timezone.utc)
        cutoff   = now - timedelta(days=30)
        starting = float(portfolio.get('starting_cash', 25000))
        recent_pnl = sum(
            t.get('profit_loss', 0) for t in trade_log
            if t.get('type') == 'SELL'
            and datetime.fromisoformat(
                t.get('timestamp', now.isoformat()).replace('Z', '+00:00')
            ).replace(tzinfo=timezone.utc) >= cutoff
        )
        pnl_pct   = recent_pnl / starting if starting > 0 else 0
        triggered = pnl_pct <= -threshold
        return triggered, {
            'rolling_30d_pnl': f'${recent_pnl:,.2f}',
            'rolling_30d_pct': f'{pnl_pct*100:.2f}%',
            'threshold':       f'-{threshold*100:.0f}%',
            'status':          'TRIGGERED' if triggered else 'OK',
        }
    except Exception as e:
        return False, {'error': str(e)}


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_risk_checks(
    symbol: str,
    portfolio: dict,
    spy_history_df,
    regime: str = 'neutral',
    skip: list  = None,
    price_history: dict  = None,
    term_structure: dict = None,
) -> RiskCheckResult:
    """
    Runs all 9 risk checks for a symbol. Returns RiskCheckResult.

    Args:
        symbol:          Ticker to evaluate
        portfolio:       Current portfolio dict
        spy_history_df:  DataFrame with SPY OHLCV (needs 'close' column)
        regime:          Current regime — tightens limits in 'crisis'
        skip:            Blocker names to bypass (for testing)
        price_history:   Dict of {symbol: DataFrame} for momentum check
    """
    skip   = skip or []
    crisis = (regime == 'crisis')
    blocked = []
    details = {}

    ph = price_history or {}
    checks = [
        ('drawdown',      lambda: check_drawdown(portfolio)),
        ('max_positions', lambda: check_max_positions(portfolio, crisis)),
        ('earnings',      lambda: check_earnings(symbol)),
        ('fda',           lambda: check_fda(symbol)),
        ('macro',         lambda: check_macro()),
        ('market_tide',   lambda: check_market_tide(spy_history_df)),
        ('momentum',         lambda: check_momentum(symbol, ph)),
        ('volatility_regime',lambda: check_volatility_regime(regime, term_structure)),
        ('circuit_breaker',  lambda: check_circuit_breaker(portfolio)),
    ]

    for name, fn in checks:
        if name in skip:
            continue
        is_blocked, info = fn()
        details[name] = info
        if is_blocked:
            blocked.append(name)

    return RiskCheckResult(
        symbol=symbol,
        passed=len(blocked) == 0,
        blockers=blocked,
        details=details,
    )


def print_risk_summary(result: RiskCheckResult):
    print(f"\n{'='*52}")
    print(f"  RISK CHECK  —  {result.symbol}")
    print(f"{'='*52}")
    print(f"  {result.summary()}")
    for name, info in result.details.items():
        if info:
            fired = '🚫' if name in result.blockers else '✅'
            print(f"  {fired} [{name}]  {info}")
    print(f"{'='*52}")


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd
    from auth import authenticate
    from data_collector import get_price_history, get_vix, get_vix_history
    from regime_engine import evaluate_regime

    print("🔌 Authenticating...")
    client, paper = authenticate()
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    # Load portfolio
    if os.path.exists('paper_portfolio.json'):
        with open('paper_portfolio.json') as f:
            portfolio = json.load(f)
        print(f"📂 Portfolio — cash: ${portfolio.get('cash', 0):,.2f}  "
              f"positions: {len(portfolio.get('positions', {}))}")
    else:
        portfolio = {'starting_cash': 25000, 'cash': 25000, 'positions': {}}
        print("📂 Using default portfolio")

    # Get regime
    vixy = get_vix(client)
    vix_hist = get_vix_history(client, days=30)
    regime_state = evaluate_regime(vixy, vix_hist)
    print(f"📊 Regime: {regime_state.regime.upper()}  (VIXY {vixy:.2f})\n")

    # SPY history for market tide
    print("📡 Fetching SPY history...")
    spy_df = get_price_history(client, 'SPY', days=30)
    print(f"  SPY rows: {len(spy_df)}\n")

    # Normal checks
    for sym in ['SPY', 'AAPL', 'NVDA']:
        result = run_risk_checks(sym, portfolio, spy_df, regime=regime_state.regime)
        print_risk_summary(result)

    # Crisis mode demo
    print("\n\n📋 Crisis mode demo (VIXY=55, max 2 positions):")
    mock_hist = pd.DataFrame({'vix': [30] * 20})
    evaluate_regime(55, mock_hist)
    result = run_risk_checks('AAPL', portfolio, spy_df, regime='crisis')
    print_risk_summary(result)

    # Restore real state
    evaluate_regime(vixy, vix_hist)
    print("\n✅ risk_manager.py working correctly.")