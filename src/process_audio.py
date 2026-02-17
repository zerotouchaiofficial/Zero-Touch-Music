"""
process_audio.py - Downloads + applies slowed+reverb. Raises DownloadError on failure.
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
COOKIES_PATH = Path("/tmp/yt_cookies.txt")


class DownloadError(Exception):
    """Raised when yt-dlp cannot download — signals main to try next song."""
    pass


def _try_download(url: str, out_path: Path, cookies_arg: list) -> bool:
    """
    Try downloading with multiple client strategies.
    Key fix: do NOT mix -f format selector with -x --audio-format.
    Let yt-dlp pick the best audio itself.
    """
    attempts = [
        "ios",
        "web",
        "android",
        "tv_embedded",
        "mweb",
    ]

    for client in attempts:
        log.info(f"  Trying client={client}...")

        cmd = [
            "yt-dlp",
            url,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--geo-bypass",
            # Let yt-dlp pick the best audio format automatically
            "-x",
            "--audio-quality", "0",
            "-o", str(out_path),
            "--extractor-args", f"youtube:player_client={client}",
            "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ] + cookies_arg

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            log.info(f"  ✓ Download succeeded with client={client}")
            return True

        err = result.stderr[:120].strip()
        log.warning(f"  client={client} failed: {err}")

    return False


def process_audio(video_id: str, title: str, artist: str, temp_dir: str) -> str:
    temp = Path(temp_dir)
    # Use %(ext)s so yt-dlp saves in whatever format it picks
    raw  = temp / f"{video_id}_raw.%(ext)s"
    out  = temp / f"{video_id}_slowed_reverb.mp3"

    # Load cookies
    if COOKIES_PATH.exists() and COOKIES_PATH.stat().st_size > 200:
        log.info(f"  Cookies loaded: {COOKIES_PATH.stat().st_size} bytes")
        cookies_arg = ["--cookies", str(COOKIES_PATH)]
    else:
        log.warning("  No cookies found — download may fail")
        cookies_arg = []

    log.info(f"  Downloading audio for video_id={video_id}...")
    success = _try_download(
        f"https://www.youtube.com/watch?v={video_id}", raw, cookies_arg
    )

    if not success:
        raise DownloadError(f"All clients failed for {video_id}")

    # Find whatever file yt-dlp saved (could be .webm, .m4a, .opus, .mp3 etc.)
    downloaded = [
        f for f in temp.glob(f"{video_id}_raw.*")
        if not f.name.endswith(".part")
    ]
    if not downloaded:
        raise DownloadError(f"No downloaded file found in {temp}")

    raw_file = downloaded[0]
    log.info(f"  Downloaded: {raw_file.name} ({raw_file.stat().st_size / 1024:.0f} KB)")

    # Convert to mp3 via ffmpeg regardless of source format
    mp3_file = temp / f"{video_id}_raw.mp3"
    if raw_file.suffix.lower() != ".mp3":
        log.info(f"  Converting {raw_file.suffix} → mp3...")
    subprocess.run([
        "ffmpeg", "-i", str(raw_file),
        "-vn", "-ar", "44100", "-ac", "2", "-b:a", "320k",
        str(mp3_file), "-y", "-loglevel", "quiet"
    ], check=True)
    if raw_file != mp3_file and raw_file.exists():
        raw_file.unlink()

    log.info(f"  MP3 ready: {mp3_file.stat().st_size / 1024:.0f} KB")

    # ── Slow down with librosa ────────────────────────────────────────
    log.info("  Applying slowed effect (0.80x)...")
    y, sr = librosa.load(str(mp3_file), sr=44100, mono=False)

    if y.ndim == 1:
        y_slow = librosa.effects.time_stretch(y, rate=SLOW_FACTOR)
        y_slow = np.stack([y_slow, y_slow])
    else:
        left  = librosa.effects.time_stretch(y[0], rate=SLOW_FACTOR)
        right = librosa.effects.time_stretch(y[1], rate=SLOW_FACTOR)
        y_slow = np.stack([left, right])

    # ── Apply effects chain ───────────────────────────────────────────
    log.info("  Applying reverb, EQ, compression...")
    board = Pedalboard([
        Compressor(threshold_db=-18, ratio=3.0, attack_ms=5.0, release_ms=100.0),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=3.0),
        HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-2.5),
        Reverb(
            room_size=REVERB_ROOM,
            damping=0.6,
            wet_level=REVERB_WET,
            dry_level=1.0 - REVERB_WET,
            width=0.9,
            freeze_mode=0.0,
        ),
    ])

    y_effected = board(y_slow.astype(np.float32), sr)
    y_effected = _normalize_loudness(y_effected, sr)

    # ── Save to WAV then export MP3 ───────────────────────────────────
    wav_path = temp / f"{video_id}_processed.wav"
    sf.write(str(wav_path), y_effected.T, sr, subtype="PCM_24")

    log.info("  Adding fade-in/out...")
    seg = AudioSegment.from_wav(str(wav_path))
    seg = seg.fade_in(3000).fade_out(4000)
    seg.export(str(out), format="mp3", bitrate="320k",
               tags={"title": f"{title} (Slowed + Reverb)", "artist": artist})

    log.info(f"  ✓ Processed: {out.name} ({out.stat().st_size / (1024*1024):.1f} MB)")
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
