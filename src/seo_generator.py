"""
seo_generator.py
Generates YouTube metadata with proper tag validation.
"""

import random
from datetime import datetime

TITLE_TEMPLATES = [
    "{title} - {artist} (Slowed + Reverb) 🌙",
    "{title} [Slowed + Reverb] | {artist} ✨",
    "{title} (Slowed to Perfection + Reverb) ~ {artist}",
    "🌊 {title} - {artist} | Slowed + Reverb",
    "{title} ♾ Slowed & Reverb | {artist} 💫",
]

DESCRIPTION_TEMPLATE = """🎵 {title} (Slowed + Reverb)
👤 Original Artist: {artist}
🎬 Channel: {channel_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ About This Edit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is a slowed + reverb version of "{title}" by {artist}.
The audio has been slowed to 80% and enhanced with warm reverb
for a dreamy, lofi aesthetic — perfect for studying, late-night
drives, or just vibing. 🌙

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📜 Credits
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎤 Original Song  : {title}
🎸 Artist         : {artist}
🔗 Original       : {original_url}
🎛️  Audio Edit     : {channel_name} (Slowed + Reverb)
📅 Uploaded       : {date}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚖️  Copyright Disclaimer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is a fan-made edit for entertainment. All rights to the
original song belong to {artist} and their label. No copyright
infringement intended. If you are the copyright owner and wish
this removed, contact us for immediate action.

Under Section 107 of the Copyright Act 1976, allowance is made
for "fair use" for transformation and commentary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔔 Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Like, Subscribe, Share
✅ Turn on notifications for daily uploads

{hashtags}
"""

BASE_TAGS = [
    "slowed and reverb",
    "slowed reverb",
    "lofi",
    "slowed songs",
    "reverb songs",
    "aesthetic music",
    "chill music",
    "study music",
    "trending songs 2026",
    "viral songs 2026",
    "slowed music",
    "dreamy music",
    "late night music",
]

HASHTAG_POOL = [
    "#slowedreverb",
    "#lofi",
    "#aesthetic",
    "#chillmusic",
    "#trending2025",
    "#viral",
    "#slowedmusic",
    "#ZeroTouchMusic",
    "#slowedandreverb",
    "#slowedreverb",
    "#slowed",
    "#reverb",
    "#slowedmusic",
    "#lofi",
    "#aesthetic",
    "#aestheticvibes",
    "#chillmusic",
    "#nightvibes",
    "#midnightvibes",
    "#sadslowed",
    "#heartbreakvibes",
    "#nostalgicvibes",
    "#HindiSlowed",
    "#PunjabiSlowed",
    "#EnglishSlowed",
    "#HaryanviSlowed",
    "#headphonesrecommended",
    "#trendingmusic",
    "#viral",
]


def generate_seo_metadata(song_title: str, artist: str,
                           channel_name: str, original_url: str) -> dict:
    # Title (max 100 chars)
    template = random.choice(TITLE_TEMPLATES)
    yt_title = template.format(title=song_title, artist=artist)
    if len(yt_title) > 100:
        yt_title = yt_title[:97] + "..."

    # Tags - YouTube limits: 500 total chars, max 30 tags, each tag max 30 chars
    song_tags = [
        song_title[:30],
        f"{song_title} slowed"[:30],
        f"{song_title} reverb"[:30],
        artist[:30],
        f"{artist} slowed"[:30],
    ]

    # Combine and filter
    all_tags = song_tags + BASE_TAGS
    all_tags = list(dict.fromkeys(all_tags))  # dedupe

    # Validate each tag: max 30 chars, alphanumeric + spaces only
    valid_tags = []
    total_chars = 0

    for tag in all_tags:
        # Clean tag
        tag = tag.strip()
        if not tag or len(tag) > 30:
            continue

        # Check if adding this tag keeps us under 500 char limit
        if total_chars + len(tag) > 450:  # leave 50 char buffer
            break

        valid_tags.append(tag)
        total_chars += len(tag)

        if len(valid_tags) >= 25:  # max 25 tags to be safe
            break

    # Hashtags for description
    hashtags = " ".join(random.sample(HASHTAG_POOL, min(6, len(HASHTAG_POOL))))

    description = DESCRIPTION_TEMPLATE.format(
        title=song_title,
        artist=artist,
        channel_name=channel_name,
        original_url=original_url,
        date=datetime.utcnow().strftime("%B %d, %Y"),
        hashtags=hashtags,
    )

    return {
        "title":       yt_title,
        "description": description,
        "tags":        valid_tags,
    }
