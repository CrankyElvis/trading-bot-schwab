"""
Daily validation cron job. v2.

Runs after market close (suggest 17:30 ET — gives Schwab time to settle fills).
1. Reconcile today's paper trades against same-day backtest
2. Re-reconcile any past day with open positions (catches multi-day CSP closes)
3. Rebuild cumulative report
4. Email the HTML cumulative view via SendGrid

v2 fixes:
- Subject line handles None realization gracefully
- Uses reconcile_with_open_followups to handle multi-day positions

Schedule in main.py SCHEDULE dict:
    "17:30": "validation_daily"

Or as standalone cron:
    30 17 * * 1-5 cd /opt/bot && python -m validation.daily_cron
"""

from __future__ import annotations

import logging
from datetime import date

from bot.config import ALERT_EMAIL
from bot.email_alerts import send_email

from validation.reconcile import reconcile_with_open_followups
from validation.cumulative import (
    load_all_daily_reports, build_cumulative, render_html,
)

log = logging.getLogger("validation.cron")


def _format_subject(cumulative) -> str:
    parts = [f"[Validation] Day {cumulative.trading_days}",
             f"agreement {cumulative.overall_agreement_rate:.0%}"]

    if cumulative.pnl_realization_ratio is not None:
        parts.append(f"realization {cumulative.pnl_realization_ratio:.0%}")
    else:
        # When backtest P&L isn't positive, realization is meaningless.
        # Use signed dollar delta instead.
        parts.append(f"delta ${cumulative.pnl_delta_total:+,.0f}")

    subject = " · ".join(parts)

    if any([cumulative.tripwire_rolling_win_rate,
            cumulative.tripwire_rolling_pnl_underperform,
            cumulative.tripwire_score_distribution_drift]):
        subject = "⚠ " + subject

    return subject


def run():
    today = date.today()
    if today.weekday() >= 5:
        log.info("Weekend — skipping validation")
        return

    # 1+2. Reconcile today, plus re-reconcile any past day with open positions
    try:
        report = reconcile_with_open_followups(today)
        log.info("Today: agreement=%.1f%%, delta=$%.2f, still_open=%d",
                 report.agreement_rate * 100,
                 report.pnl_delta,
                 report.n_still_open)
    except FileNotFoundError as e:
        log.warning("No snapshot for today: %s", e)
    except Exception:
        log.exception("Daily reconcile failed — continuing to cumulative anyway")

    # 3. Build cumulative
    daily_reports = load_all_daily_reports()
    if not daily_reports:
        log.warning("No daily reports yet — paper trading hasn't produced data")
        return

    cumulative = build_cumulative(daily_reports)
    html = render_html(cumulative)

    # 4. Email
    subject = _format_subject(cumulative)
    send_email(to=ALERT_EMAIL, subject=subject, html_body=html)
    log.info("Sent validation email — subject: %s", subject)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()