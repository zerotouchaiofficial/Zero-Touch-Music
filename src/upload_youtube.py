"""
upload_youtube.py
Uploads with quota detection
"""

import os
import json
import time
import logging
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

log = logging.getLogger("yt-uploader")

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

CATEGORY_MUSIC = "10"
PRIVACY_STATUS = os.environ.get("VIDEO_PRIVACY", "public")
PLAYLIST_CACHE = Path("output/playlists.json")

PLAYLIST_TITLES = {
    "english":  "Slowed + Reverb | English Hits",
    "hindi":    "Slowed + Reverb | Hindi Songs",
    "punjabi":  "Slowed + Reverb | Punjabi Tracks",
    "haryanvi": "Slowed + Reverb | Haryanvi Music",
}


class QuotaExceededError(Exception):
    """Raised when daily YouTube upload quota is exceeded."""
    pass


def _get_credentials() -> Credentials:
    client_id     = os.environ["YT_CLIENT_ID"]
    client_secret = os.environ["YT_CLIENT_SECRET"]
    refresh_token = os.environ["YT_REFRESH_TOKEN"]

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )

    creds.refresh(Request())
    return creds


def _load_playlist_cache() -> dict:
    if PLAYLIST_CACHE.exists():
        try:
            with open(PLAYLIST_CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_playlist_cache(cache: dict):
    PLAYLIST_CACHE.parent.mkdir(exist_ok=True)
    with open(PLAYLIST_CACHE, "w") as f:
        json.dump(cache, f)


def _get_or_create_playlist(youtube, language: str) -> str:
    cache = _load_playlist_cache()

    if language in cache:
        try:
            youtube.playlists().list(part="snippet", id=cache[language]).execute()
            log.info(f"  Using existing {language} playlist")
            return cache[language]
        except HttpError:
            log.warning(f"  Cached playlist not found, creating new...")

    title = PLAYLIST_TITLES.get(language, f"Slowed + Reverb | {language.title()}")
    description = f"All {language.title()} songs slowed to 80% with reverb. New uploads daily!"

    try:
        response = youtube.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {"title": title, "description": description},
                "status": {"privacyStatus": "public"}
            }
        ).execute()

        playlist_id = response["id"]
        log.info(f"  ✓ Created {language} playlist")
        cache[language] = playlist_id
        _save_playlist_cache(cache)
        return playlist_id

    except HttpError as e:
        log.warning(f"  Failed to create playlist: {e}")
        return None


def _add_to_playlist(youtube, video_id: str, playlist_id: str):
    if not playlist_id:
        return

    try:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id}
                }
            }
        ).execute()
        log.info(f"  ✓ Added to playlist")
    except HttpError as e:
        log.warning(f"  Failed to add to playlist: {e}")


def upload_to_youtube(
    video_path: str,
    thumbnail_path: str,
    title: str,
    description: str,
    tags: list[str],
    language: str = "english",
) -> str:
    creds   = _get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    log.info(f"  Uploading: {Path(video_path).name} ({Path(video_path).stat().st_size / (1024*1024):.1f} MB)")

    body = {
        "snippet": {
            "title":       title[:100],
            "description": description,
            "tags":        tags,
            "categoryId":  CATEGORY_MUSIC,
        },
        "status": {
            "privacyStatus":           PRIVACY_STATUS,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    video_id = _resumable_upload(request)
    log.info(f"  ✓ Video uploaded! ID: {video_id}")

    # Set thumbnail
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
        ).execute()
        log.info("  ✓ Thumbnail set!")
    except HttpError as e:
        log.warning(f"  Thumbnail upload failed: {e}")

    # Add to playlist
    log.info(f"  Adding to {language} playlist...")
    playlist_id = _get_or_create_playlist(youtube, language)
    _add_to_playlist(youtube, video_id, playlist_id)

    return f"https://www.youtube.com/watch?v={video_id}"


def _resumable_upload(request) -> str:
    response   = None
    error      = None
    retry      = 0
    max_retry  = 10
    retry_codes = [500, 502, 503, 504]

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log.info(f"  ↑ Upload progress: {pct}%")
                
        except HttpError as e:
            # Check for quota exceeded
            if e.resp.status == 400:
                error_content = str(e.content)
                if "uploadLimitExceeded" in error_content or "quotaExceeded" in error_content:
                    raise QuotaExceededError("Daily YouTube upload quota exceeded")
            
            if e.resp.status in retry_codes:
                error = f"HTTP {e.resp.status}: {e.content}"
            else:
                raise
                
        except Exception as e:
            error = str(e)

        if error:
            retry += 1
            if retry > max_retry:
                raise RuntimeError(f"Upload failed after {max_retry} retries: {error}")
            wait = 2 ** retry
            log.warning(f"  ⚠️  Upload error. Retrying in {wait}s...")
            time.sleep(wait)
            error = None

    return response["id"]
