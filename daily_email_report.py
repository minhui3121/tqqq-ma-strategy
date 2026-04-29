"""Send a daily QQQ/TQQQ signal report email.

This script computes the latest strategy state and emails a summary with:
- Latest QQQ close, SMA80, SMA190
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
    parser.add_argument(
        "--to",
        action="append",
        default=None,
        help="Recipient email address. Repeat --to or use comma-separated values.",
    )
    parser.add_argument("--from-email", default=None, help="From email address. Defaults to SMTP user.")
    parser.add_argument("--smtp-host", default=os.getenv("SMTP_HOST", "smtp.gmail.com"), help="SMTP host.")
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")), help="SMTP port.")
    parser.add_argument("--smtp-user", default=os.getenv("SMTP_USER"), help="SMTP username/login.")
    parser.add_argument("--smtp-password", default=os.getenv("SMTP_PASSWORD"), help="SMTP password/app password.")
    parser.add_argument("--smtp-password-file", default=None, help="Path to a file containing SMTP password.")
    parser.add_argument("--short-window", type=int, default=80, help="Short SMA window.")
    parser.add_argument("--long-window", type=int, default=190, help="Long SMA window.")
    parser.add_argument("--lookback-days", type=int, default=500, help="Calendar days of history to fetch.")
    parser.add_argument("--dry-run", action="store_true", help="Print email content without sending.")
    return parser.parse_args()


def parse_recipients(to_args: list[str] | None, env_value: str | None) -> list[str]:
    recipients: list[str] = []

    for raw in (to_args or []):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        recipients.extend([p for p in parts if p])

    if env_value:
        parts = [p.strip() for p in env_value.replace(";", ",").split(",")]
        recipients.extend([p for p in parts if p])

    # Preserve order and remove duplicates.
    return list(dict.fromkeys(recipients))


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


def build_report(args: argparse.Namespace) -> tuple[str, str, str]:
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
    sma80 = float(latest["sma80"])
    sma190 = float(latest["sma190"])

    today_target = int(latest["target_position"])
    yesterday_target = int(prev["target_position"])

    dist_80 = pct_distance(qqq_close, sma80)
    dist_190 = pct_distance(qqq_close, sma190)

    signal_date = pd.Timestamp(signals.index[-1]).strftime("%Y-%m-%d")
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    action = action_text(today_target, yesterday_target)

    subject = f"Daily QQQ/TQQQ Signal Report - {signal_date}"
    text_body = (
        f"Daily Strategy Report\n"
        f"Generated: {generated_at}\n"
        f"Signal Date: {signal_date}\n\n"
        f"Signal Asset (QQQ):\n"
        f"- QQQ Close: {qqq_close:.2f}\n"
        f"- SMA80: {sma80:.2f}\n"
        f"- SMA190: {sma190:.2f}\n"
        f"- Distance to SMA80: {dist_80:+.2f}%\n"
        f"- Distance to SMA190: {dist_190:+.2f}%\n\n"
        f"Positioning:\n"
        f"- Yesterday target: {position_text(yesterday_target)}\n"
        f"- Today target: {position_text(today_target)}\n"
        f"- Action: {action}\n\n"
        f"Execution model:\n"
        f"- Signals are based on QQQ close\n"
        f"- Trades execute at next trading day TQQQ open\n"
    )

    action_bg = "#1f7a1f" if "BUY" in action or "HOLD" in action else "#8a1f1f"
    html_body = f"""<!doctype html>
<html>
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <style>
            body, table, td {{ -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }}
            .wrap {{ width: 100%; padding: 18px 10px; }}
            .card {{ width: 100%; max-width: 560px; background: #ffffff; border: 1px solid #d8e0ee; border-radius: 12px; overflow: hidden; }}
            .header {{ padding: 18px 20px; background: #1d2433; color: #ffffff; }}
            .title {{ font-size: 22px; font-weight: 700; line-height: 1.2; }}
            .subtitle {{ opacity: 0.9; margin-top: 5px; font-size: 13px; line-height: 1.35; }}
            .section {{ padding: 18px 20px; }}
            .section-title {{ font-size: 16px; font-weight: 700; margin-bottom: 10px; }}
            .metrics td {{ padding: 11px 10px; border-bottom: 1px solid #e6ecf7; font-size: 14px; }}
            .metrics tr:nth-child(odd) {{ background: #f8faff; }}
            .metrics td:last-child {{ text-align: right; font-weight: 700; }}
            .action {{ margin-top: 12px; padding: 12px 14px; border-radius: 9px; color: #ffffff; font-weight: 700; font-size: 14px; display: inline-block; }}
            .footer {{ padding: 14px 20px 20px 20px; background: #f8faff; color: #4a5568; font-size: 12px; line-height: 1.55; }}
            @media only screen and (max-width: 480px) {{
                .wrap {{ padding: 10px 6px; }}
                .header {{ padding: 16px 14px; }}
                .section {{ padding: 14px; }}
                .footer {{ padding: 12px 14px 16px 14px; }}
                .title {{ font-size: 20px; }}
                .subtitle {{ font-size: 12px; }}
                .section-title {{ font-size: 15px; }}
                .metrics td {{ font-size: 14px; padding: 10px 8px; }}
            }}
        </style>
    </head>
    <body style=\"margin:0;padding:0;background:#f3f6fb;font-family:Segoe UI,Arial,sans-serif;color:#1d2433;\">
        <table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\" class=\"wrap\">
            <tr>
                <td align=\"center\">
                    <table role=\"presentation\" cellspacing=\"0\" cellpadding=\"0\" class=\"card\">
                        <tr>
                            <td class=\"header\">
                                <div class=\"title\">Daily Strategy Report</div>
                                <div class=\"subtitle\">Signal Date: {signal_date}<br/>Generated: {generated_at}</div>
                            </td>
                        </tr>
                        <tr>
                            <td class=\"section\">
                                <div class=\"section-title\">Signal Asset (QQQ)</div>
                                <table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\" class=\"metrics\" style=\"border-collapse:collapse;border:1px solid #d8e0ee;border-radius:8px;overflow:hidden;\">
                                    <tr><td>QQQ Close</td><td>{qqq_close:.2f}</td></tr>
                                    <tr><td>SMA80</td><td>{sma80:.2f}</td></tr>
                                    <tr><td>SMA190</td><td>{sma190:.2f}</td></tr>
                                    <tr><td>Distance to SMA80</td><td>{dist_80:+.2f}%</td></tr>
                                    <tr><td>Distance to SMA190</td><td>{dist_190:+.2f}%</td></tr>
                                </table>
                            </td>
                        </tr>
                        <tr>
                            <td class=\"section\" style=\"padding-top:0;\">
                                <div class=\"section-title\">Positioning</div>
                                <div style=\"font-size:14px;line-height:1.75;\">Yesterday target: <b>{position_text(yesterday_target)}</b><br/>Today target: <b>{position_text(today_target)}</b></div>
                                <div class=\"action\" style=\"background:{action_bg};\">Action: {action}</div>
                            </td>
                        </tr>
                        <tr>
                            <td class=\"footer\">
                                <b>Execution model</b><br/>
                                Signals are based on QQQ close. Trades execute at next trading day TQQQ open.
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
</html>
"""

    return subject, text_body, html_body


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    to_emails: list[str],
    subject: str,
    text_body: str,
    html_body: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg, to_addrs=to_emails)


def main() -> None:
    args = parse_args()

    recipients = parse_recipients(args.to, os.getenv("REPORT_TO"))

    if not recipients:
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

    subject, text_body, html_body = build_report(args)

    if args.dry_run:
        print(subject)
        print("-" * 70)
        print(text_body)
        return

    send_email(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password,
        from_email=from_email,
        to_emails=recipients,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )
    print(f"Email sent to: {', '.join(recipients)}")


if __name__ == "__main__":
    main()
