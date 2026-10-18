"""Console + Telegram + generic webhook alerts. All non-blocking-safe."""
import logging
from typing import Optional

import requests

log = logging.getLogger("alerts")


class Alerter:
    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat: str = "",
        webhook_url: str = "",
        timeout: float = 5.0,
    ):
        self.tg_token = telegram_token
        self.tg_chat = telegram_chat
        self.webhook = webhook_url
        self.timeout = timeout

    def send(self, text: str):
        print(text)  # always echo
        self._telegram(text)
        self._webhook(text)

    def _telegram(self, text: str):
        if not (self.tg_token and self.tg_chat):
            return
        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            r = requests.post(
                url,
                json={
                    "chat_id": self.tg_chat,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=self.timeout,
            )
            if r.status_code >= 300:
                log.warning("telegram failed: %s %s", r.status_code, r.text[:200])
        except Exception as e:
            log.warning("telegram exception: %s", e)

    def _webhook(self, text: str):
        if not self.webhook:
            return
        try:
            r = requests.post(self.webhook, json={"text": text}, timeout=self.timeout)
            if r.status_code >= 300:
                log.warning("webhook failed: %s", r.status_code)
        except Exception as e:
            log.warning("webhook exception: %s", e)


def format_signal(symbol: str, interval: str, s) -> str:
    targets = " | ".join(f"{t:.5f}" for t in s.targets)
    reasons = "\n".join(f"  • {r}" for r in s.reasons)
    return (
        f"🎯 *{symbol}* `{interval}` *{s.direction.upper()}*\n"
        f"Entry: `{s.entry_price:.5f}`\n"
        f"Stop:  `{s.stop:.5f}`\n"
        f"TPs:   `{targets}`\n"
        f"Confluence: *{s.confluence_score}/5*\n"
        f"Bar: {s.entry_time}\n"
        f"Reasons:\n{reasons}"
    )
