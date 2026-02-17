"""
process_audio.py
Gets audio from SoundCloud (no bot detection, no cookies needed).
Falls back to searching multiple sources.
"""

import os
import logging
import subprocess
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pedalboard import Pedalboard, Reverb, LowShelfFilter, HighShelfFilter, Compressor
import soundfile as sf
import librosa

log = logging.getLogger("yt-uploader")

SLOW_FACTOR  = 0.80
REVERB_ROOM  = 0.75
REVERB_WET   = 0.35
TARGET_LUFS  = -14.0


class DownloadError(Exception):
    pass


def _search_soundcloud(song_title: str, artist: str) -> str | None:
    """Search SoundCloud and return the URL of the best match."""
    query = f"{artist} {song_title}".replace(" ", "+")
    search_url = f"ytsearch1:{artist} {song_title} official audio"

    # Try SoundCloud search
    sc_search = f"scsearch1:{artist} {song_title}"

    for search in [sc_search, search_url]:
        result = subprocess.run([
            "yt-dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--print", "webpage_url",
            search,
        ], capture_output=True, text=True)

        if result.returncode == 0 and result.stdout.strip():
            url = result.stdout.strip()
            log.info(f"  Found URL: {url[:60]}...")
            return url

    return None


def _download_from_url(url: str, out_path: Path) -> bool:
    """Download audio from a given URL."""
    cmd = [
        "yt-dlp",
        url,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", str(out_path),
        "--geo-bypass",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Find the actual output file
        files = list(out_path.parent.glob(f"{out_path.stem}*"))
        if files and any(f.stat().st_size > 10000 for f in files):
            log.info(f"  ✓ Downloaded successfully")
            return True

    log.warning(f"  Download failed: {result.stderr[:100].strip()}")
    return False


def process_audio(video_id: str, title: str, artist: str, temp_dir: str) -> str:
    temp    = Path(temp_dir)
    raw_out = temp / f"{video_id}_raw"   # no extension, yt-dlp adds it
    out     = temp / f"{video_id}_slowed_reverb.mp3"

    # ── Step 1: Find audio source (SoundCloud preferred) ──────────────
    log.info(f"  Searching SoundCloud for: {artist} - {title}")
    url = _search_soundcloud(title, artist)

    if not url:
        raise DownloadError(f"Could not find '{title}' by {artist} on any source")

    # ── Step 2: Download ──────────────────────────────────────────────
    log.info(f"  Downloading audio...")
    success = _download_from_url(url, raw_out)

    if not success:
        raise DownloadError(f"Download failed for '{title}'")

    # Find the downloaded file
    downloaded = [
        f for f in temp.glob(f"{video_id}_raw*")
        if not f.name.endswith(".part") and f.stat().st_size > 10000
    ]
    if not downloaded:
        raise DownloadError(f"No downloaded file found in {temp}")

    raw_file = downloaded[0]

    # Convert to mp3 if needed
    mp3_file = temp / f"{video_id}_raw.mp3"
    if raw_file.suffix.lower() != ".mp3":
        log.info(f"  Converting {raw_file.suffix} → mp3...")
        subprocess.run([
            "ffmpeg", "-i", str(raw_file),
            "-vn", "-ar", "44100", "-ac", "2", "-b:a", "320k",
            str(mp3_file), "-y", "-loglevel", "quiet"
        ], check=True)
        if raw_file != mp3_file:
            raw_file.unlink()
    else:
        mp3_file = raw_file

    log.info(f"  Audio ready: {mp3_file.stat().st_size / 1024:.0f} KB")

    # ── Step 3: Slow down ──────────────────────────────────────────────
    log.info("  Applying slowed effect (0.80x)...")
    y, sr = librosa.load(str(mp3_file), sr=44100, mono=False)

    if y.ndim == 1:
        y_slow = librosa.effects.time_stretch(y, rate=SLOW_FACTOR)
        y_slow = np.stack([y_slow, y_slow])
    else:
        left  = librosa.effects.time_stretch(y[0], rate=SLOW_FACTOR)
        right = librosa.effects.time_stretch(y[1], rate=SLOW_FACTOR)
        y_slow = np.stack([left, right])

    # ── Step 4: Effects chain ──────────────────────────────────────────
    log.info("  Applying reverb, EQ, compression...")
    board = Pedalboard([
        Compressor(threshold_db=-18, ratio=3.0, attack_ms=5.0, release_ms=100.0),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=3.0),
        HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-2.5),
        Reverb(
            room_size=REVERB_ROOM, damping=0.6,
            wet_level=REVERB_WET, dry_level=1.0 - REVERB_WET,
            width=0.9, freeze_mode=0.0,
        ),
    ])

    y_effected = board(y_slow.astype(np.float32), sr)
    y_effected = _normalize_loudness(y_effected, sr)

    wav_path = temp / f"{video_id}_processed.wav"
    sf.write(str(wav_path), y_effected.T, sr, subtype="PCM_24")

    # ── Step 5: Fade and export ────────────────────────────────────────
    log.info("  Adding fade-in/out and exporting...")
    seg = AudioSegment.from_wav(str(wav_path))
    seg = seg.fade_in(3000).fade_out(4000)
    seg.export(str(out), format="mp3", bitrate="320k",
               tags={"title": f"{title} (Slowed + Reverb)", "artist": artist})

    log.info(f"  ✓ Done: {out.name} ({out.stat().st_size / (1024*1024):.1f} MB)")
    return str(out)


def _normalize_loudness(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        import pyloudnorm as pyln
        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.T)
        if loudness > -70:
            audio = pyln.normalize.loudness(audio.T, loudness, TARGET_LUFS).T
    except ImportError:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
    return audio
