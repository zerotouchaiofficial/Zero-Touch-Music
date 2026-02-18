"""
utils.py
Shared utilities: logging, cleanup, Discord notifications.
"""

import os
import logging
import shutil
import requests
from pathlib import Path


def setup_logging(level=logging.INFO) -> logging.Logger:
    log = logging.getLogger("yt-uploader")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(handler)
    log.setLevel(level)
    return log


def cleanup_temp_files(temp_dir: str):
    """Remove all files in the temp directory."""
    p = Path(temp_dir)
    if p.exists():
        for f in p.iterdir():
            try:
                f.unlink()
            except Exception:
                pass


def safe_filename(s: str, max_len: int = 60) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", s)[:max_len]


def send_discord_notification(title: str, description: str, color: int = 5814783):
    """
    Send Discord webhook notification.
    color: decimal (green=5763719, yellow=16776960, red=15158332)
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK", "")
    if not webhook_url:
        return  # silently skip if not configured

    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": None,
            }]
        }
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        # Don't crash pipeline if Discord fails
        print(f"Discord notification failed: {e}")
