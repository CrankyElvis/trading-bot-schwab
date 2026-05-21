# -*- coding: utf-8 -*-
"""
health_check.py  --  Server health monitoring for trading bot

Runs as a lightweight HTTP endpoint on port 8080.
UptimeRobot (free) pings it every 5 minutes.
If the bot goes silent, UptimeRobot sends an alert.

Also monitors:
  - Bot process running
  - Last cycle timestamp (alert if > 45 min stale during market hours)
  - Memory usage
  - Disk space
  - Paper portfolio integrity

Setup:
  1. Start this alongside the bot: python health_check.py &
  2. Add to systemd as healthcheck.service (see bottom of file)
  3. Sign up at uptimerobot.com (free)
  4. Add monitor: HTTP(S), URL = http://142.93.4.251:8080/health
  5. Set alert interval: 5 minutes
  6. Add your email for alerts
"""

import json
import os
import subprocess
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BOT_DIR    = Path('/root/trading-bot')
PORT       = 8080
VERSION    = '1.0'


def get_bot_status() -> dict:
    """Check if tradingbot systemd service is running."""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'tradingbot'],
            capture_output=True, text=True, timeout=5
        )
        return {
            'running': result.stdout.strip() == 'active',
            'state':   result.stdout.strip(),
        }
    except Exception as e:
        return {'running': False, 'state': f'error: {e}'}


def get_last_cycle() -> dict:
    """Check when bot last wrote a cycle event to bot_log.jsonl.
    Uses bot_log.jsonl (written every cycle) not portfolio.json
    (only written on trades — silent on all_blocked days).
    """
    try:
        log_file = BOT_DIR / 'bot_log.jsonl'
        if not log_file.exists():
            return {'stale': True, 'minutes_ago': 999, 'last_updated': 'missing'}

        # Read last line of log for most recent event
        last_line = None
        with open(log_file, 'rb') as f:
            # Efficient tail — seek to end and scan back
            f.seek(0, 2)
            size = f.tell()
            buf  = min(4096, size)
            f.seek(-buf, 2)
            chunk = f.read().decode('utf-8', errors='ignore')
            lines = [l for l in chunk.strip().splitlines() if l.strip()]
            last_line = lines[-1] if lines else None

        if not last_line:
            return {'stale': False, 'minutes_ago': 0, 'last_updated': 'no events'}

        event = json.loads(last_line)
        last_updated = event.get('timestamp', '')
        if not last_updated:
            return {'stale': False, 'minutes_ago': 0, 'last_updated': 'unknown'}

        last_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        minutes_ago = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60

        # Market hours check with DST-aware ET offset
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo('America/New_York'))
        except ImportError:
            # Fallback: approximate DST (EDT=-4 Mar-Nov, EST=-5 Nov-Mar)
            month = datetime.now().month
            offset = -4 if 3 <= month <= 11 else -5
            now_et = datetime.now(timezone(timedelta(hours=offset)))

        market_hours = (now_et.weekday() < 5 and 9 <= now_et.hour < 17)
        stale = market_hours and minutes_ago > 45

        return {
            'stale':        stale,
            'minutes_ago':  round(minutes_ago, 1),
            'last_updated': last_updated,
            'last_event':   event.get('event', 'unknown'),
            'market_hours': market_hours,
        }
    except Exception as e:
        return {'stale': False, 'minutes_ago': 0, 'last_updated': f'error: {e}'}


def get_system_stats() -> dict:
    """Get memory and disk usage."""
    try:
        # Memory
        with open('/proc/meminfo') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1])
        total_mb = meminfo.get('MemTotal', 0) / 1024
        avail_mb = meminfo.get('MemAvailable', 0) / 1024
        used_pct = round((1 - avail_mb / total_mb) * 100, 1) if total_mb > 0 else 0

        # Disk
        stat = os.statvfs('/')
        disk_total_gb = stat.f_frsize * stat.f_blocks / 1e9
        disk_free_gb  = stat.f_frsize * stat.f_bfree  / 1e9
        disk_used_pct = round((1 - disk_free_gb / disk_total_gb) * 100, 1)

        return {
            'memory_used_pct': used_pct,
            'memory_avail_mb': round(avail_mb, 0),
            'disk_used_pct':   disk_used_pct,
            'disk_free_gb':    round(disk_free_gb, 1),
        }
    except Exception as e:
        return {'error': str(e)}


def get_portfolio_summary() -> dict:
    """Quick portfolio health check."""
    try:
        with open(BOT_DIR / 'paper_portfolio.json') as f:
            p = json.load(f)
        starting = p.get('starting_cash', 25000)
        cash     = p.get('cash', starting)
        positions = p.get('positions', {})
        pos_value = sum(pos.get('cost_basis', 0) for pos in positions.values())
        total     = cash + pos_value
        pnl_pct   = (total - starting) / starting * 100 if starting > 0 else 0
        return {
            'total_value':  round(total, 2),
            'cash':         round(cash, 2),
            'positions':    len(positions),
            'pnl_pct':      round(pnl_pct, 2),
            'trades':       len(p.get('trade_log', [])),
        }
    except Exception as e:
        return {'error': str(e)}


class HealthHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Suppress request logs

    def do_GET(self):
        if self.path == '/health':
            bot      = get_bot_status()
            cycle    = get_last_cycle()
            system   = get_system_stats()
            portfolio = get_portfolio_summary()

            # Overall health
            healthy = (
                bot.get('running', False) and
                not cycle.get('stale', False) and
                system.get('memory_used_pct', 100) < 90 and
                system.get('disk_used_pct', 100) < 90
            )

            status_code = 200 if healthy else 503

            response = {
                'status':    'healthy' if healthy else 'degraded',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version':   VERSION,
                'bot':       bot,
                'last_cycle': cycle,
                'system':    system,
                'portfolio': portfolio,
            }

            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())

        elif self.path == '/':
            # Simple status page
            bot   = get_bot_status()
            cycle = get_last_cycle()
            html  = f"""<html><body>
<h2>Trading Bot Health</h2>
<p>Bot: <b>{'RUNNING' if bot.get('running') else 'DOWN'}</b></p>
<p>Last cycle: <b>{cycle.get('minutes_ago', '?')} min ago</b></p>
<p>Stale: <b>{'YES' if cycle.get('stale') else 'NO'}</b></p>
<p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
<p><a href="/health">JSON health endpoint</a></p>
</body></html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    print(f'Health check server starting on port {PORT}...')
    print(f'Endpoint: http://142.93.4.251:{PORT}/health')
    print(f'Add this URL to UptimeRobot for free monitoring')
    server = HTTPServer(('0.0.0.0', PORT), HealthHandler)
    server.serve_forever()