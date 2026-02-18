"""
main.py - Handles YouTube quota gracefully
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
from upload_youtube import upload_to_youtube, QuotaExceededError
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

    log.info("📡 Step 1: Fetching trending songs...")
    candidates = get_trending_songs(max_candidates=10)
    if not candidates:
        log.error("No trending songs found.")
        sys.exit(1)

    song = None
    processed_audio = None

    for i, candidate in enumerate(candidates):
        log.info(f"\n🎵 Song {i+1}/{len(candidates)}: '{candidate['title']}' by {candidate['artist']}")

        try:
            log.info("🎧 Processing audio...")
            processed_audio = process_audio(
                video_id=candidate["video_id"],
                title=candidate["title"],
                artist=candidate["artist"],
                temp_dir=str(TEMP_DIR),
            )
            song = candidate
            log.info(f"✅ Audio done!")
            break

        except DownloadError as e:
            log.warning(f"⏭️  Download failed: {e}")
            cleanup_temp_files(str(TEMP_DIR))
            continue

        except Exception as e:
            log.warning(f"⏭️  Error: {e}")
            cleanup_temp_files(str(TEMP_DIR))
            continue

    if not song or not processed_audio:
        log.error("❌ All songs failed.")
        sys.exit(1)

    video_url = None
    
    try:
        log.info("\n📝 Generating SEO...")
        metadata = generate_seo_metadata(
            song_title=song["title"],
            artist=song["artist"],
            channel_name=CHANNEL_NAME,
            original_url=f"https://www.youtube.com/watch?v={song['video_id']}",
        )

        log.info("\n🎬 Creating video...")
        video_path = create_video(
            audio_path=processed_audio,
            song_title=song["title"],
            artist=song["artist"],
            channel_name=CHANNEL_NAME,
            output_dir=str(OUTPUT_DIR),
            temp_dir=str(TEMP_DIR),
        )

        log.info("\n🖼️  Creating thumbnail...")
        thumbnail_path = generate_thumbnail(
            song_title=song["title"],
            artist=song["artist"],
            channel_name=CHANNEL_NAME,
            output_dir=str(OUTPUT_DIR),
        )

        log.info("\n🚀 Uploading to YouTube...")
        video_url = upload_to_youtube(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            language=language,
        )
        log.info(f"✅ Uploaded! → {video_url}")

    except QuotaExceededError as e:
        log.warning("⏸️  YouTube quota exceeded for today!")
        log.warning("   Video files saved - will upload tomorrow")
        send_discord_notification(
            title="⏸️  Daily Quota Reached",
            description=f"YouTube upload limit hit. Video saved for tomorrow.\n**{song['title']}** by {song['artist']}",
            color=16776960  # yellow
        )
        # DON'T mark as uploaded - will retry tomorrow
        log.info("\n🧹 Cleaning up...")
        cleanup_temp_files(str(TEMP_DIR))
        sys.exit(0)  # Exit gracefully, not an error
        
    except Exception as e:
        log.error(f"❌ Upload failed: {e}")
        log.error(traceback.format_exc())
        send_discord_notification(
            title="❌ Upload Failed",
            description=str(e)[:200],
            color=15158332
        )
        sys.exit(1)
    
    # Only mark as uploaded if upload succeeded
    try:
        log.info("\n📝 Marking song as uploaded...")
        mark_uploaded(
            video_id=song["video_id"],
            title=song["title"],
            artist=song["artist"],
            youtube_url=video_url,
        )
        log.info("✅ Song marked!")
        
    except Exception as e:
        log.error(f"❌ Error marking: {e}")
    
    try:
        send_discord_notification(
            title=f"✅ Upload Complete ({language.upper()})",
            description=f"**{song['title']}** by {song['artist']}\n[Watch]({video_url})",
            color=5763719
        )
    except Exception:
        pass

    log.info("\n🧹 Cleaning up...")
    cleanup_temp_files(str(TEMP_DIR))
    log.info("\n🎉 Complete!")


if __name__ == "__main__":
    run_pipeline()
