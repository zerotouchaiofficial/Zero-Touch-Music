"""
fetch_trending.py - Returns language WITH the songs
"""

import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from googleapiclient.discovery import build

log = logging.getLogger("yt-uploader")

REPO_ROOT = Path(__file__).parent.parent
UPLOADED_LOG = REPO_ROOT / "output" / "uploaded.json"
HISTORY_LOG  = REPO_ROOT / "output" / "upload_history.txt"
LANGUAGE_LOG = REPO_ROOT / "output" / "last_language.json"

BLOCKLIST = {"c5aYTMnACfk", "1FVF-9KQiPo"}

SEARCH_QUERIES_BY_LANGUAGE = {
    "english": ["trending english songs 2025"],
    "hindi": ["trending hindi songs 2025"],
    "punjabi": ["trending punjabi songs 2025"],
    "haryanvi": ["trending haryanvi songs 2025"],
}

LANGUAGE_ROTATION = ["english", "hindi", "punjabi", "haryanvi"]


def _load_uploaded() -> set:
    if UPLOADED_LOG.exists():
        try:
            with open(UPLOADED_LOG) as f:
                data = json.load(f)
                print(f"📊 Loaded {len(data)} uploaded songs")
                return set(data)
        except Exception as e:
            print(f"⚠️ Failed to load: {e}")
    return set()


def _save_uploaded(video_ids: set):
    UPLOADED_LOG.parent.mkdir(exist_ok=True, parents=True)
    with open(UPLOADED_LOG, "w") as f:
        json.dump(sorted(list(video_ids)), f, indent=2)
    print(f"💾 Saved {len(video_ids)} IDs")


def _get_next_language() -> str:
    if LANGUAGE_LOG.exists():
        try:
            with open(LANGUAGE_LOG) as f:
                data = json.load(f)
                last_lang = data.get("language", "english")
        except Exception:
            last_lang = "english"
    else:
        last_lang = "english"

    try:
        idx = LANGUAGE_ROTATION.index(last_lang)
        next_lang = LANGUAGE_ROTATION[(idx + 1) % len(LANGUAGE_ROTATION)]
    except ValueError:
        next_lang = "english"

    LANGUAGE_LOG.parent.mkdir(exist_ok=True, parents=True)
    with open(LANGUAGE_LOG, "w") as f:
        json.dump({"language": next_lang}, f)

    return next_lang


def get_trending_songs(max_candidates: int = 10) -> tuple[list, str]:
    """
    Returns: (list of song dicts, language used)
    Now returns the ACTUAL language that was searched!
    """
    api_key = os.environ["YOUTUBE_API_KEY"]
    youtube = build("youtube", "v3", developerKey=api_key)
    
    uploaded = _load_uploaded()
    skip = uploaded | BLOCKLIST
    candidates = []

    language = _get_next_language()  # This advances the rotation
    queries = SEARCH_QUERIES_BY_LANGUAGE[language]
    query = random.choice(queries)

    print(f"🔍 Language: {language.upper()} | Query: '{query}'")

    region = "IN" if language in ["hindi", "punjabi", "haryanvi"] else "US"

    try:
        resp = youtube.videos().list(
            part="snippet,contentDetails",
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
                "video_id": vid_id,
                "title": _clean_title(snippet.get("title", "Unknown")),
                "artist": _clean_artist(snippet.get("channelTitle", "Unknown")),
                "duration": secs,
            })
            if len(candidates) >= max_candidates:
                break
    except Exception as e:
        print(f"⚠️ Chart failed: {e}")

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
                    "video_id": vid_id,
                    "title": _clean_title(snippet.get("title", "Unknown")),
                    "artist": _clean_artist(snippet.get("channelTitle", "Unknown")),
                    "duration": 180,
                })
                if len(candidates) >= max_candidates:
                    break
        except Exception:
            pass

    print(f"✅ Found {len(candidates)} {language} songs")
    return candidates, language  # Return BOTH songs AND language


def mark_uploaded(video_id: str, title: str = "", artist: str = "", youtube_url: str = "", language: str = ""):
    """Mark song as uploaded."""
    
    print(f"\n🔒 Marking {video_id} as uploaded ({language.upper()})")
    
    # Update JSON
    uploaded = _load_uploaded()
    uploaded.add(video_id)
    _save_uploaded(uploaded)
    
    # Verify
    verify = _load_uploaded()
    if video_id in verify:
        print(f"✅ Verified in uploaded.json")
    else:
        print(f"❌ ERROR: Not saved!")
    
    # Update history
    HISTORY_LOG.parent.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    entry = f"""
{'='*70}
Uploaded: {timestamp}
Language: {language.upper()}
Title:    {title}
Artist:   {artist}
Video ID: {video_id}
URL:      {youtube_url}
{'='*70}
"""
    
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    
    print(f"✅ Saved to history (total: {len(uploaded)})\n")


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
