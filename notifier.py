# -*- coding: utf-8 -*-
"""
notifier.py  --  Anomaly alerts + daily summary email via SendGrid

Setup:
  1. Sign up at sendgrid.com (free, 100 emails/day)
  2. Create an API key (Settings -> API Keys -> Full Access)
  3. Add to .env file:
       SENDGRID_API_KEY=SG.xxxxxxxx
       ALERT_EMAIL_TO=you@gmail.com
       ALERT_EMAIL_FROM=bot@yourdomain.com  (must be verified sender in SendGrid)

Anomaly triggers (7):
  1. Drawdown > 10% in 30 days        CRITICAL
  2. Stop loss cascade (3+ in one day) WARNING
  3. Regime flip to crisis             CRITICAL
  4. VIX spike > 35                    WARNING
  5. Bot crash / restart               CRITICAL
  6. CSP assignment                    WARNING
  7. Position sizing error             WARNING

Daily summary: fires after-hours cycle (6pm ET) after overnight position taken
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SENDGRID_API_KEY  = os.getenv('SENDGRID_API_KEY', '')
ALERT_EMAIL_TO    = os.getenv('ALERT_EMAIL_TO', '')
ALERT_EMAIL_FROM  = os.getenv('ALERT_EMAIL_FROM', '')

SEVERITY_EMOJI = {
    'CRITICAL': '[!!]',
    'WARNING':  '[! ]',
    'INFO':     '[  ]',
}


# ── Core send function ────────────────────────────────────────────────────────

def send_email(subject: str, body: str, severity: str = 'INFO') -> bool:
    """Send email via SendGrid API. Returns True if successful."""
    if not SENDGRID_API_KEY or not ALERT_EMAIL_TO or not ALERT_EMAIL_FROM:
        print(f'  [notifier] Email not configured -- skipping: {subject}')
        return False

    emoji = SEVERITY_EMOJI.get(severity, '[  ]')
    full_subject = f'{emoji} Trading Bot: {subject}'

    payload = json.dumps({
        'personalizations': [{'to': [{'email': ALERT_EMAIL_TO}]}],
        'from':             {'email': ALERT_EMAIL_FROM},
        'subject':          full_subject,
        'content':          [{'type': 'text/plain', 'value': body}],
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.sendgrid.com/v3/mail/send',
        data=payload,
        headers={
            'Authorization': f'Bearer {SENDGRID_API_KEY}',
            'Content-Type':  'application/json',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            success = resp.status in (200, 202)
            if success:
                print(f'  [notifier] {severity} email sent: {subject}')
            return success
    except urllib.error.HTTPError as e:
        print(f'  [notifier] SendGrid error {e.code}: {e.read().decode()}')
        return False
    except Exception as e:
        print(f'  [notifier] Email failed: {e}')
        return False


# ── Anomaly checks ────────────────────────────────────────────────────────────

def check_drawdown(portfolio: dict, threshold_pct: float = 10.0) -> bool:
    """Alert if portfolio is down > threshold% from starting cash."""
    starting = portfolio.get('starting_cash', 25000)
    cash     = portfolio.get('cash', starting)
    positions_value = sum(
        p.get('cost_basis', 0) for p in portfolio.get('positions', {}).values()
    )
    total = cash + positions_value
    drawdown_pct = (starting - total) / starting * 100 if starting > 0 else 0

    if drawdown_pct >= threshold_pct:
        body = (
            f'DRAWDOWN ALERT\n\n'
            f'Portfolio is down {drawdown_pct:.1f}% from starting value.\n\n'
            f'Starting cash:  ${starting:,.2f}\n'
            f'Current value:  ${total:,.2f}\n'
            f'Drawdown:       ${starting - total:,.2f} ({drawdown_pct:.1f}%)\n\n'
            f'Threshold: {threshold_pct:.0f}%\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}\n\n'
            f'Action: Review positions and consider reducing exposure.'
        )
        send_email(f'Drawdown Alert: -{drawdown_pct:.1f}%', body, 'CRITICAL')
        return True
    return False


def check_stop_loss_cascade(portfolio: dict, max_stops: int = 3) -> bool:
    """Alert if 3+ stop losses triggered today."""
    today = datetime.now().strftime('%Y-%m-%d')
    trade_log = portfolio.get('trade_log', [])
    todays_stops = [
        t for t in trade_log
        if t.get('timestamp', '').startswith(today)
        and t.get('exit_reason', '') == 'stop_loss'
    ]
    if len(todays_stops) >= max_stops:
        body = (
            f'STOP LOSS CASCADE\n\n'
            f'{len(todays_stops)} stop losses triggered today.\n\n'
            f'Symbols: {", ".join(t.get("symbol","?") for t in todays_stops)}\n'
            f'Total losses: ${sum(t.get("profit_loss",0) for t in todays_stops):,.2f}\n\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}\n\n'
            f'Action: Check for systematic issue. Consider halting new entries.'
        )
        send_email(f'Stop Loss Cascade: {len(todays_stops)} stops today', body, 'WARNING')
        return True
    return False


def check_regime_crisis(regime: str, previous_regime: str = '') -> bool:
    """Alert if regime flipped to crisis."""
    if regime == 'crisis' and previous_regime != 'crisis':
        body = (
            f'CRISIS REGIME ALERT\n\n'
            f'Regime has flipped to CRISIS.\n\n'
            f'Previous regime: {previous_regime.upper()}\n'
            f'Current regime:  CRISIS\n\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}\n\n'
            f'Bot behavior: All new entries halted. Positions being reduced.\n'
            f'Action: Monitor closely. Consider manual intervention.'
        )
        send_email('CRISIS REGIME -- All entries halted', body, 'CRITICAL')
        return True
    return False


def check_vix_spike(vix: float, threshold: float = 35.0) -> bool:
    """Alert if VIX spikes above threshold."""
    if vix >= threshold:
        body = (
            f'VIX SPIKE ALERT\n\n'
            f'VIX has spiked to {vix:.1f} (threshold: {threshold:.0f})\n\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}\n\n'
            f'Bot behavior: Position sizing reduced 50%. No new CSPs.\n'
            f'Action: Review open positions for gap risk.'
        )
        send_email(f'VIX Spike: {vix:.1f}', body, 'WARNING')
        return True
    return False


def check_bot_restart(log_entries: list) -> bool:
    """
    Detect bot restart by finding a gap > 15 min between log entries
    during market hours, or by detecting the first entry after a silence.
    main.py does not log a 'startup' event explicitly — detect via log gaps.
    """
    if len(log_entries) < 2:
        return False
    try:
        # Parse all timestamps in order
        times = []
        for entry in log_entries:
            ts_str = entry.get('timestamp', '')
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                times.append(ts)
            except Exception:
                pass

        if len(times) < 2:
            return False

        times.sort()
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(minutes=30)

        # Check for a large gap (> 15 min) followed by recent activity
        for i in range(1, len(times)):
            gap_min = (times[i] - times[i-1]).total_seconds() / 60
            if gap_min > 15 and times[i] >= recent_cutoff:
                body = (
                    f'BOT RESTART DETECTED\n\n'
                    f'Log gap of {gap_min:.0f} minutes detected, followed by new activity.\n\n'
                    f'Gap: {times[i-1].strftime("%H:%M ET")} -> {times[i].strftime("%H:%M ET")}\n\n'
                    f'This may indicate a crash or restart. Check server logs:\n'
                    f'  journalctl -u tradingbot -n 50 --no-pager\n\n'
                    f'Server: 142.93.4.251'
                )
                send_email('Bot Restart Detected -- Log Gap', body, 'CRITICAL')
                return True
    except Exception:
        pass
    return False


def check_csp_assignment(portfolio: dict) -> bool:
    """Alert if a CSP was assigned (short put exercised)."""
    today = datetime.now().strftime('%Y-%m-%d')
    trade_log = portfolio.get('trade_log', [])
    assignments = [
        t for t in trade_log
        if t.get('timestamp', '').startswith(today)
        and 'assign' in t.get('exit_reason', '').lower()
    ]
    if assignments:
        for a in assignments:
            body = (
                f'CSP ASSIGNMENT\n\n'
                f'Symbol: {a.get("symbol", "?")}\n'
                f'Quantity: {a.get("quantity", 0)} shares assigned\n'
                f'Price: ${a.get("fill_price", 0):.2f}\n\n'
                f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}\n\n'
                f'Action: Position now open. Consider selling covered call (wheel).'
            )
            send_email(f'CSP Assigned: {a.get("symbol","?")}', body, 'WARNING')
        return True
    return False


def check_cycle_errors(log_entries: list, lookback_minutes: int = 30) -> bool:
    """Alert if any error events in bot_log.jsonl in last N minutes."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    recent_errors = []
    for entry in log_entries:
        if entry.get('event') != 'error':
            continue
        ts_str = entry.get('timestamp', '')
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts > cutoff:
                recent_errors.append(entry.get('error', 'unknown error'))
        except Exception:
            pass

    if recent_errors:
        unique_errors = list(dict.fromkeys(recent_errors))  # dedupe, preserve order
        body = (
            f'CYCLE ERROR ALERT\n\n'
            f'{len(recent_errors)} error(s) in the last {lookback_minutes} minutes:\n\n'
            + '\n'.join(f'  - {e}' for e in unique_errors) +
            f'\n\nTime: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}\n\n'
            f'Check logs: ssh root@142.93.4.251 "tail -20 /root/trading-bot/bot_log.jsonl"'
        )
        send_email(f'Cycle Errors Detected: {len(recent_errors)} in {lookback_minutes}min', body, 'WARNING')
        return True
    return False


def run_anomaly_checks(portfolio: dict, regime: str = 'neutral',
                       previous_regime: str = '', vix: float = 0.0,
                       log_entries: list = None) -> dict:
    """
    Run all 7 anomaly checks. Call this every cycle from main.py.
    Returns dict of which checks triggered.
    """
    if log_entries is None:
        log_entries = []
    # Auto-load from bot_log.jsonl if no entries passed
    if not log_entries:
        try:
            import json as _json
            log_path = Path(__file__).parent / 'bot_log.jsonl'
            if log_path.exists():
                with open(log_path) as f:
                    log_entries = [_json.loads(line) for line in f if line.strip()]
        except Exception:
            pass

    triggered = {}
    triggered['drawdown']       = check_drawdown(portfolio)
    triggered['stop_cascade']   = check_stop_loss_cascade(portfolio)
    triggered['crisis_regime']  = check_regime_crisis(regime, previous_regime)
    triggered['vix_spike']      = check_vix_spike(vix) if vix > 0 else False
    triggered['bot_restart']    = check_bot_restart(log_entries)
    triggered['csp_assignment'] = check_csp_assignment(portfolio)
    triggered['cycle_errors']   = check_cycle_errors(log_entries)

    return triggered


# ── Daily summary ─────────────────────────────────────────────────────────────

def _safe_days_running(created_str: str) -> str:
    try:
        if not created_str:
            return 'unknown'
        dt = datetime.fromisoformat(created_str[:10])
        return str((datetime.now() - dt).days)
    except Exception:
        return 'unknown'


def send_daily_summary(portfolio: dict, regime_state=None,
                       macro_state=None, signals_today: list = None) -> bool:
    """
    Send end-of-day summary email after close cycle + overnight position taken.
    Call from after_hours cycle in main.py.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M ET')
    today = datetime.now().strftime('%Y-%m-%d')

    # Portfolio metrics
    starting      = portfolio.get('starting_cash', 25000)
    cash          = portfolio.get('cash', starting)
    positions     = portfolio.get('positions', {})
    trade_log     = portfolio.get('trade_log', [])
    spread_cost   = portfolio.get('total_spread_cost', 0)

    positions_value = sum(p.get('cost_basis', 0) for p in positions.values())
    total_value     = cash + positions_value
    total_pnl       = total_value - starting
    total_pnl_pct   = total_pnl / starting * 100 if starting > 0 else 0

    # Today's trades
    todays_trades = [t for t in trade_log if t.get('timestamp','').startswith(today)]
    todays_buys   = [t for t in todays_trades if t.get('type') == 'BUY']
    todays_sells  = [t for t in todays_trades if t.get('type') == 'SELL']
    todays_pnl    = sum(t.get('profit_loss', 0) for t in todays_sells)

    # Regime info — handle both RegimeState object and plain string
    if regime_state is None:
        regime, vixy = 'unknown', 0.0
    elif isinstance(regime_state, str):
        regime, vixy = regime_state, 0.0
    else:
        regime = getattr(regime_state, 'regime', 'unknown')
        vixy   = getattr(regime_state, 'vixy', 0.0)
    # Split positions by source
    standard_pos = {s: p for s, p in positions.items()
                    if p.get('source', 'standard') == 'standard'}
    politician_pos = {s: p for s, p in positions.items()
                      if p.get('source', 'standard') == 'politician'}
    wsb_pos = {s: p for s, p in positions.items()
               if p.get('source', 'standard') == 'wsb'}

    def fmt_pos(pos_dict):
        return ', '.join(
            f"{sym} x{p.get('quantity', 0)} @ ${p.get('avg_price', 0):.2f}"
            for sym, p in pos_dict.items()
        ) if pos_dict else 'None'

    positions_str = fmt_pos(standard_pos)

    # Overnight position
    overnight_syms = [sym for sym in positions if sym in ('SPY','QQQ','GLD')]
    overnight_str  = ', '.join(overnight_syms) if overnight_syms else 'None taken'

    # Macro sentinel
    macro_str = 'Unavailable'
    if macro_state:
        macro_str = (f'{macro_state.warning.upper()} '
                     f'(score: {macro_state.score:.2f}) -- {macro_state.summary}')

    body = f"""DAILY SUMMARY -- {today}
{'='*50}

PORTFOLIO
  Total value:    ${total_value:>12,.2f}
  Cash:           ${cash:>12,.2f}
  Positions:      ${positions_value:>12,.2f}
  Total P&L:      ${total_pnl:>+12,.2f} ({total_pnl_pct:+.2f}%)
  Spread costs:   ${spread_cost:>12,.4f}

TODAY'S ACTIVITY
  Buys:           {len(todays_buys)}
  Sells:          {len(todays_sells)}
  Realized P&L:   ${todays_pnl:>+,.2f}

OPEN POSITIONS (Standard)
  {positions_str}

CONGRESSIONAL TRADES (Politician-sourced)
  {fmt_pos(politician_pos)}

WSB INJECTED POSITIONS
  {fmt_pos(wsb_pos)}

OVERNIGHT POSITION
  {overnight_str}

MARKET REGIME
  Regime:         {regime.upper()}
  VIXY:           {vixy:.2f}

MACRO SENTINEL
  {macro_str}

PAPER TRADING DAY
  Started:        {portfolio.get('created','unknown')[:10]}
  Days running:   {_safe_days_running(portfolio.get('created',''))}

{'='*50}
Generated: {now}
"""

    return send_email(f'Daily Summary -- {today} -- P&L ${total_pnl:+,.0f}', body, 'INFO')


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Testing notifier...')
    if not SENDGRID_API_KEY:
        print('  No SENDGRID_API_KEY in .env -- add it to test email sending')
        print('  Required .env entries:')
        print('    SENDGRID_API_KEY=SG.xxxxxxxx')
        print('    ALERT_EMAIL_TO=you@gmail.com')
        print('    ALERT_EMAIL_FROM=bot@yourdomain.com')
    else:
        result = send_email(
            'Test Alert',
            f'Trading bot notifier is working.\nSent at {datetime.now().strftime("%Y-%m-%d %H:%M ET")}',
            'INFO'
        )
        print(f'  Test email: {"sent" if result else "failed"}')