"""
process_audio.py
Downloads the source audio with yt-dlp and applies:
  • Slowed to 0.80× speed  (dreamy tempo)
  • Reverb using Pedalboard (Spotify open-source)
  • Slight bass boost + high-frequency softening for lofi feel
  • Fade-in (3s) and fade-out (4s)
  • Exported as high-quality MP3 (320kbps)
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

SLOW_FACTOR  = 0.80   # 80% of original speed
REVERB_ROOM  = 0.75   # Room size (0-1)
REVERB_WET   = 0.35   # Wet/dry mix
TARGET_LUFS  = -14.0  # Loudness target (YouTube standard)


def process_audio(video_id: str, title: str, artist: str, temp_dir: str) -> str:
    temp    = Path(temp_dir)
    raw     = temp / f"{video_id}_raw.%(ext)s"
    out     = temp / f"{video_id}_slowed_reverb.mp3"

    # Write cookies to temp file if available
    cookies_file = None
    yt_cookies = os.environ.get("YT_COOKIES", "")
    if yt_cookies:
        cookies_file = temp / "cookies.txt"
        cookies_file.write_text(yt_cookies)

    # Download
    log.info(f"  Downloading audio for video_id={video_id}...")
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-f", "bestaudio/best",
        "-o", str(raw),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--geo-bypass",
        "--extractor-args", "youtube:player_client=web,ios",
        "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ]
    if cookies_file and cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find downloaded file (extension may vary before conversion)
    downloaded = list(temp.glob(f"{video_id}_raw.*"))
    if not downloaded:
        raise FileNotFoundError(f"Downloaded file not found in {temp}")

    raw_file = downloaded[0]

    # Convert to mp3 if needed
    if raw_file.suffix.lower() != ".mp3":
        log.info(f"  Converting {raw_file.suffix} to mp3...")
        converted = temp / f"{video_id}_raw.mp3"
        subprocess.run([
            "ffmpeg", "-i", str(raw_file),
            "-vn", "-ar", "44100", "-ac", "2", "-b:a", "320k",
            str(converted), "-y", "-loglevel", "quiet"
        ], check=True)
        raw_file.unlink()
        raw_file = converted

    log.info(f"  Downloaded: {raw_file.stat().st_size / 1024:.0f} KB")

    # Load with librosa for time-stretching
    log.info("  Applying slowed effect (0.80x)...")
    y, sr = librosa.load(str(raw_file), sr=44100, mono=False)

    if y.ndim == 1:
        y_slow = librosa.effects.time_stretch(y, rate=SLOW_FACTOR)
        y_slow = np.stack([y_slow, y_slow])
    else:
        left  = librosa.effects.time_stretch(y[0], rate=SLOW_FACTOR)
        right = librosa.effects.time_stretch(y[1], rate=SLOW_FACTOR)
        y_slow = np.stack([left, right])

    log.info("  Applying reverb, EQ, and compression...")

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

    wav_path = temp / f"{video_id}_processed.wav"
    sf.write(str(wav_path), y_effected.T, sr, subtype="PCM_24")

    log.info("  Adding fade-in/out...")
    seg = AudioSegment.from_wav(str(wav_path))
    seg = seg.fade_in(3000).fade_out(4000)

    seg.export(str(out), format="mp3", bitrate="320k",
               tags={
                   "title":  f"{title} (Slowed + Reverb)",
                   "artist": artist,
               })

    log.info(f"  Processed audio saved: {out.name} ({out.stat().st_size / (1024*1024):.1f} MB)")
    return str(out)


def _normalize_loudness(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.T)
        if loudness > -70:
            audio = pyln.normalize.loudness(audio.T, loudness, TARGET_LUFS).T
    except ImportError:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
    return audio
