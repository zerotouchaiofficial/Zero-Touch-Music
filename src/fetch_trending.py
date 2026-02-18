"""
fetch_trending.py
Rotates between English, Hindi, Punjabi, Haryanvi trending music.
Returns list of candidates to try.
"""

import os
import json
import random
import logging
from pathlib import Path
from googleapiclient.discovery import build

log = logging.getLogger("yt-uploader")
UPLOADED_LOG = Path("output/uploaded.json")
LANGUAGE_LOG = Path("output/last_language.json")

# Blocklist for problematic videos
BLOCKLIST = {"c5aYTMnACfk", "1FVF-9KQiPo"}

# Rotating language searches - cycles through these
SEARCH_QUERIES_BY_LANGUAGE = {
    "english": [
        "trending english songs 2025",
        "top english hits 2025",
        "viral english songs 2025",
        "best english music 2025",
        "popular english songs 2025",
    ],
    "hindi": [
        "trending hindi songs 2025",
        "top bollywood songs 2025",
        "viral hindi music 2025",
        "latest hindi hits 2025",
        "popular hindi songs 2025",
    ],
    "punjabi": [
        "trending punjabi songs 2025",
        "top punjabi hits 2025",
        "viral punjabi music 2025",
        "latest punjabi songs 2025",
        "popular punjabi songs 2025",
    ],
    "haryanvi": [
        "trending haryanvi songs 2025",
        "top haryanvi hits 2025",
        "viral haryanvi music 2025",
        "latest haryanvi songs 2025",
        "popular haryanvi songs 2025",
    ],
}

LANGUAGE_ROTATION = ["english", "hindi", "punjabi", "haryanvi"]


def _load_uploaded() -> set:
    if UPLOADED_LOG.exists():
        with open(UPLOADED_LOG) as f:
            return set(json.load(f))
    return set()


def _save_uploaded(video_ids: set):
    UPLOADED_LOG.parent.mkdir(exist_ok=True)
    with open(UPLOADED_LOG, "w") as f:
        json.dump(list(video_ids), f)


def _get_next_language() -> str:
    """Rotate through languages - returns next language to search."""
    if LANGUAGE_LOG.exists():
        try:
            with open(LANGUAGE_LOG) as f:
                data = json.load(f)
                last_lang = data.get("language", "english")
        except Exception:
            last_lang = "english"
    else:
        last_lang = "english"

    # Get next language in rotation
    try:
        idx = LANGUAGE_ROTATION.index(last_lang)
        next_lang = LANGUAGE_ROTATION[(idx + 1) % len(LANGUAGE_ROTATION)]
    except ValueError:
        next_lang = "english"

    # Save for next time
    LANGUAGE_LOG.parent.mkdir(exist_ok=True)
    with open(LANGUAGE_LOG, "w") as f:
        json.dump({"language": next_lang}, f)

    return next_lang


def get_trending_songs(max_candidates: int = 10) -> list:
    """
    Returns list of trending songs from the next language in rotation.
    """
    api_key = os.environ["YOUTUBE_API_KEY"]
    youtube = build("youtube", "v3", developerKey=api_key)
    uploaded = _load_uploaded()
    skip = uploaded | BLOCKLIST
    candidates = []

    # Get next language
    language = _get_next_language()
    queries  = SEARCH_QUERIES_BY_LANGUAGE[language]
    query    = random.choice(queries)

    log.info(f"  Language: {language.upper()} | Query: '{query}'")

    # Try chart first (filtered by region if applicable)
    region_codes = {
        "english":  "US",
        "hindi":    "IN",
        "punjabi":  "IN",
        "haryanvi": "IN",
    }
    region = region_codes.get(language, "US")

    try:
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            videoCategoryId="10",
            regionCode=region,
            maxResults=50,
        ).execute()

        for item in resp.get("items", []):
            vid_id = item["id"]
            if vid_id in skip:
                continue
            secs = _parse_duration(item["contentDetails"].get("duration", "PT0S"))
            if not (60 <= secs <= 480):
                continue
            snippet = item["snippet"]
            candidates.append({
                "video_id":   vid_id,
                "title":      _clean_title(snippet.get("title", "Unknown")),
                "artist":     _clean_artist(snippet.get("channelTitle", "Unknown")),
                "duration":   secs,
            })
            if len(candidates) >= max_candidates:
                break

    except Exception as e:
        log.warning(f"Chart lookup failed: {e}")

    # Fill remaining from search
    if len(candidates) < max_candidates:
        try:
            resp = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                videoCategoryId="10",
                order="viewCount",
                maxResults=25,
                videoDuration="medium",
                regionCode=region,
            ).execute()

            for item in resp.get("items", []):
                vid_id = item["id"]["videoId"]
                if vid_id in skip:
                    continue
                snippet = item["snippet"]
                candidates.append({
                    "video_id":   vid_id,
                    "title":      _clean_title(snippet.get("title", "Unknown")),
                    "artist":     _clean_artist(snippet.get("channelTitle", "Unknown")),
                    "duration":   180,
                })
                if len(candidates) >= max_candidates:
                    break
        except Exception as e:
            log.warning(f"Search failed: {e}")

    log.info(f"  Found {len(candidates)} {language} songs to try")
    return candidates


def mark_uploaded(video_id: str):
    """Call after successful upload."""
    uploaded = _load_uploaded()
    uploaded.add(video_id)
    _save_uploaded(uploaded)


def _parse_duration(iso: str) -> int:
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return 0
    return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)


def _clean_title(t: str) -> str:
    import re
    for p in [r"\(Official.*?\)", r"\[Official.*?\]", r"\(HD\)", r"\(4K\)", r"ft\..*", r"feat\..*", r"\(Lyric.*?\)"]:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return t.strip(" -|")


def _clean_artist(c: str) -> str:
    import re
    return re.sub(r"(VEVO|Official|Music|Channel)", "", c, flags=re.IGNORECASE).strip()
