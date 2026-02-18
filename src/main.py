"""
main.py - Pipeline orchestrator with duplicate prevention.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from fetch_trending import get_trending_songs, mark_uploaded, get_current_language
from process_audio import process_audio, DownloadError
from create_video import create_video
from generate_thumbnail import generate_thumbnail
from upload_youtube import upload_to_youtube
from seo_generator import generate_seo_metadata
from utils import cleanup_temp_files, setup_logging, send_discord_notification

CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "LoFi Aura")
OUTPUT_DIR   = Path("output")
TEMP_DIR     = Path("temp")

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


def run_pipeline():
    log = setup_logging()
    log.info("=" * 60)
    log.info(f"🎵 YT Auto-Uploader started at {datetime.utcnow().isoformat()}")
    log.info("=" * 60)

    language = get_current_language()

    # ── Step 1: Get candidate songs (skips duplicates automatically) ──
    log.info("📡 Step 1: Fetching trending songs...")
    candidates = get_trending_songs(max_candidates=10)
    if not candidates:
        log.error("No trending songs found. Exiting.")
        send_discord_notification(
            title="⚠️ No Songs Found",
            description=f"No {language} trending songs available.",
            color=16776960
        )
        sys.exit(1)

    # ── Try each candidate until one works ───────────────────────────
    song            = None
    processed_audio = None

    for i, candidate in enumerate(candidates):
        log.info(f"\n🎵 Trying song {i+1}/{len(candidates)}: '{candidate['title']}' by {candidate['artist']} (id={candidate['video_id']})")

        try:
            log.info("🎧 Step 2: Processing audio (slowed+reverb)...")
            processed_audio = process_audio(
                video_id=candidate["video_id"],
                title=candidate["title"],
                artist=candidate["artist"],
                temp_dir=str(TEMP_DIR),
            )
            song = candidate
            log.info(f"✅ Audio processed successfully!")
            break

        except DownloadError as e:
            log.warning(f"⏭️  Download failed for '{candidate['title']}': {e} — trying next song...")
            cleanup_temp_files(str(TEMP_DIR))
            continue

        except Exception as e:
            log.warning(f"⏭️  Unexpected error for '{candidate['title']}': {e} — trying next song...")
            cleanup_temp_files(str(TEMP_DIR))
            continue

    if not song or not processed_audio:
        log.error("❌ All candidate songs failed to download. Exiting.")
        send_discord_notification(
            title="❌ All Downloads Failed",
            description=f"Tried {len(candidates)} {language} songs, all failed.",
            color=15158332
        )
        sys.exit(1)

    try:
        # ── Step 3: SEO metadata ─────────────────────────────────────
        log.info("\n📝 Step 3: Generating SEO metadata...")
        metadata = generate_seo_metadata(
            song_title=song["title"],
            artist=song["artist"],
            channel_name=CHANNEL_NAME,
            original_url=f"https://www.youtube.com/watch?v={song['video_id']}",
        )
        log.info(f"✅ Title: {metadata['title']}")

        # ── Step 4: Create video ──────────────────────────────────────
        log.info("\n🎬 Step 4: Creating video...")
        video_path = create_video(
            audio_path=processed_audio,
            song_title=song["title"],
            artist=song["artist"],
            channel_name=CHANNEL_NAME,
            output_dir=str(OUTPUT_DIR),
            temp_dir=str(TEMP_DIR),
        )
        log.info(f"✅ Video: {video_path}")

        # ── Step 5: Thumbnail ─────────────────────────────────────────
        log.info("\n🖼️  Step 5: Generating thumbnail...")
        thumbnail_path = generate_thumbnail(
            song_title=song["title"],
            artist=song["artist"],
            channel_name=CHANNEL_NAME,
            output_dir=str(OUTPUT_DIR),
        )
        log.info(f"✅ Thumbnail: {thumbnail_path}")

        # ── Step 6: Upload ────────────────────────────────────────────
        log.info("\n🚀 Step 6: Uploading to YouTube...")
        video_url = upload_to_youtube(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            language=language,
        )
        log.info(f"✅ Uploaded! → {video_url}")

        # ── Mark as uploaded (prevents duplicates forever) ────────────
        log.info("\n📝 Marking song as uploaded...")
        mark_uploaded(
            video_id=song["video_id"],
            title=song["title"],
            artist=song["artist"],
            language=language,
            youtube_url=video_url,
        )

        # ── Success notification ──────────────────────────────────────
        send_discord_notification(
            title=f"✅ Upload Complete ({language.upper()})",
            description=f"**{song['title']}** by {song['artist']}\n[Watch on YouTube]({video_url})",
            color=5763719
        )

        # ── Step 7: Cleanup ───────────────────────────────────────────
        log.info("\n🧹 Step 7: Cleaning up...")
        cleanup_temp_files(str(TEMP_DIR))

        log.info("\n🎉 Pipeline complete!")

    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}")
        log.error(traceback.format_exc())
        send_discord_notification(
            title="❌ Pipeline Error",
            description=f"Failed during processing: {str(e)[:200]}",
            color=15158332
        )
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
