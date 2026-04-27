"""Send a daily QQQ/TQQQ signal report email.

This script computes the latest strategy state and emails a summary with:
- Latest QQQ close, SMA100, SMA190
- Distance from each SMA
- Target position signal
- Action guidance for next market open

Use --dry-run to print the message without sending.
"""

from __future__ import annotations

import argparse
import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage

import pandas as pd

from src.data_loader import download_qqq_and_tqqq_data
from src.strategy import generate_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send daily strategy email report.")
    parser.add_argument("--to", default=os.getenv("REPORT_TO"), help="Recipient email address. Defaults to REPORT_TO env var.")
    parser.add_argument("--from-email", default=None, help="From email address. Defaults to SMTP user.")
    parser.add_argument("--smtp-host", default=os.getenv("SMTP_HOST", "smtp.gmail.com"), help="SMTP host.")
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")), help="SMTP port.")
    parser.add_argument("--smtp-user", default=os.getenv("SMTP_USER"), help="SMTP username/login.")
    parser.add_argument("--smtp-password", default=os.getenv("SMTP_PASSWORD"), help="SMTP password/app password.")
    parser.add_argument("--smtp-password-file", default=None, help="Path to a file containing SMTP password.")
    parser.add_argument("--short-window", type=int, default=100, help="Short SMA window.")
    parser.add_argument("--long-window", type=int, default=190, help="Long SMA window.")
    parser.add_argument("--lookback-days", type=int, default=500, help="Calendar days of history to fetch.")
    parser.add_argument("--dry-run", action="store_true", help="Print email content without sending.")
    return parser.parse_args()


def position_text(position: int) -> str:
    return "LONG" if int(position) == 1 else "CASH"


def action_text(today_target: int, yesterday_target: int) -> str:
    if today_target != yesterday_target:
        if today_target == 1:
            return "BUY TQQQ at next market open"
        return "SELL TQQQ at next market open"
    if today_target == 1:
        return "HOLD LONG"
    return "STAY IN CASH"


def pct_distance(value: float, anchor: float) -> float:
    if anchor == 0:
        return 0.0
    return ((value / anchor) - 1.0) * 100.0


def build_report(args: argparse.Namespace) -> tuple[str, str]:
    end_date = datetime.utcnow().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=args.lookback_days)

    merged = download_qqq_and_tqqq_data(
        start_date=str(start_date),
        end_date=str(end_date),
        short_window=args.short_window,
        long_window=args.long_window,
    )
    signals = generate_signals(
        merged,
        short_window=args.short_window,
        long_window=args.long_window,
    )

    if signals.empty:
        raise ValueError("No signal data available.")

    latest = signals.iloc[-1]
    prev = signals.iloc[-2] if len(signals) > 1 else latest

    qqq_close = float(latest["QQQ_Close"])
    sma100 = float(latest["sma100"])
    sma190 = float(latest["sma190"])

    today_target = int(latest["target_position"])
    yesterday_target = int(prev["target_position"])

    dist_100 = pct_distance(qqq_close, sma100)
    dist_190 = pct_distance(qqq_close, sma190)

    signal_date = pd.Timestamp(signals.index[-1]).strftime("%Y-%m-%d")
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    subject = f"Daily QQQ/TQQQ Signal Report - {signal_date}"
    body = (
        f"Daily Strategy Report\n"
        f"Generated: {generated_at}\n"
        f"Signal Date: {signal_date}\n\n"
        f"Signal Asset (QQQ):\n"
        f"- QQQ Close: {qqq_close:.2f}\n"
        f"- SMA100: {sma100:.2f}\n"
        f"- SMA190: {sma190:.2f}\n"
        f"- Distance to SMA100: {dist_100:+.2f}%\n"
        f"- Distance to SMA190: {dist_190:+.2f}%\n\n"
        f"Positioning:\n"
        f"- Yesterday target: {position_text(yesterday_target)}\n"
        f"- Today target: {position_text(today_target)}\n"
        f"- Action: {action_text(today_target, yesterday_target)}\n\n"
        f"Execution model:\n"
        f"- Signals are based on QQQ close\n"
        f"- Trades execute at next trading day TQQQ open\n"
    )
    return subject, body


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main() -> None:
    args = parse_args()

    if not args.to:
        raise ValueError("Missing recipient: set --to or REPORT_TO.")

    if args.smtp_password_file and not args.smtp_password:
        with open(args.smtp_password_file, "r", encoding="utf-8") as f:
            args.smtp_password = f.read().strip()

    if not args.dry_run:
        if not args.smtp_user:
            raise ValueError("Missing SMTP credentials: set --smtp-user or SMTP_USER.")
        if not args.smtp_password:
            raise ValueError("Missing SMTP credentials: set --smtp-password, SMTP_PASSWORD, or --smtp-password-file.")

    from_email = args.from_email or args.smtp_user or "noreply@example.com"

    subject, body = build_report(args)

    if args.dry_run:
        print(subject)
        print("-" * 70)
        print(body)
        return

    send_email(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password,
        from_email=from_email,
        to_email=args.to,
        subject=subject,
        body=body,
    )
    print(f"Email sent to {args.to}")


if __name__ == "__main__":
    main()
