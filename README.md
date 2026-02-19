# 🎵 Zero Touch Music — Slowed + Reverb

> Fully automated YouTube channel that fetches trending songs, applies
> professional Slowed + Reverb audio effects, generates cinematic videos
> with animated backgrounds, creates eye-catching thumbnails, and uploads
> everything to YouTube — **3 times per day, zero human intervention.**

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎵 Trending songs | Pulls from YouTube's most-popular music chart |
| 🎧 Audio effects | Slowed to 80%, warm reverb, bass boost, compression, LUFS normalization |
| 🎬 Animated video | Animated gradient, floating particles, rotating rings, film grain |
| 🖼️ Custom thumbnail | Gradient + waveform art, bold text, auto-generated each upload |
| 📝 SEO metadata | Optimized title, 500-char description, 30+ tags |
| ⏰ Auto schedule | Runs 5× daily at peak hours via GitHub Actions |
| 🔄 No repeats | Tracks uploaded songs so nothing is re-uploaded |
| 💳 Monetization-friendly | Full credits, copyright disclaimer, niche targeting |

---

## 🚀 Quick Setup (30 minutes)

### Step 1 — Fork this repo

Click **Fork** on GitHub. Enable Actions on your fork.

---

### Step 2 — Google Cloud Console Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project: **YT-Auto-Uploader**
3. Enable these APIs:
   - **YouTube Data API v3**
4. Go to **Credentials → Create Credentials → API Key**
   - Copy this → `YOUTUBE_API_KEY` secret
5. Go to **Credentials → Create Credentials → OAuth 2.0 Client ID**
   - Application type: **Desktop App**
   - Name: `yt-uploader`
   - Download the JSON or note your Client ID + Client Secret
6. Go to **OAuth consent screen**:
   - User type: External
   - Add your Google account as a test user
   - Scopes: add `youtube.upload` and `youtube`

---

### Step 3 — Get Your Refresh Token (run once, locally)

```bash
pip install google-auth-oauthlib google-api-python-client
# Fill in CLIENT_ID and CLIENT_SECRET in get_refresh_token.py
python get_refresh_token.py
```

A browser window opens → log in with your YouTube channel account → authorize.
Copy the printed `YT_REFRESH_TOKEN`.

---

### Step 4 — Add GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

Add ALL of these:

| Secret Name | Value |
|---|---|
| `YOUTUBE_API_KEY` | Your API key from Step 2 |
| `YT_CLIENT_ID` | OAuth Client ID from Step 2 |
| `YT_CLIENT_SECRET` | OAuth Client Secret from Step 2 |
| `YT_REFRESH_TOKEN` | Token from Step 3 |
| `CHANNEL_NAME` | Your channel name (e.g. `LoFi Aura`) |
| `GH_PAT` | GitHub Personal Access Token (with `repo` scope) |

#### How to get GH_PAT:
Go to GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)** → Generate new token → check `repo` → copy it.

---

### Step 5 — Configure Channel Name

Edit `src/seo_generator.py` — optionally adjust:
- `TITLE_TEMPLATES` — your video title styles
- `BASE_TAGS` — niche tags for your channel
- `DESCRIPTION_TEMPLATE` — add your social links

---

### Step 6 — Test Run

Trigger a manual run:
```
GitHub → Actions → "Auto Upload Slowed+Reverb" → Run workflow
```

Set `privacy` to **`unlisted`** for your first test so the video isn't public yet.
Check the Actions log — it should show each step completing.

---

### Step 7 — Go Live

Change `VIDEO_PRIVACY` default to `public` (already set), and the workflow
will start uploading publicly 5× per day automatically.

---

## ⏰ Upload Schedule

| Time (UTC) | Why This Time |
|---|---|
| 06:00 | US West Coast morning, EU afternoon |
| 10:00 | EU prime time, US East morning |
| 14:00 | US East lunch, global midday |
| 18:00 | Global evening prime time |
| 22:00 | Best for lofi/slowed niche (late-night listeners) |

---

## 📈 Monetization Tips

**YouTube Partner Program requires:**
- 1,000 subscribers
- 4,000 watch hours in the last 12 months

**This system helps by:**
- ✅ Posting 5× daily (35 videos/week!) — massive watch hour accumulation
- ✅ SEO-optimized titles/descriptions target high-search keywords
- ✅ Slowed+Reverb is a highly engaged niche with loyal subscribers
- ✅ Trending songs = built-in search volume
- ✅ Custom thumbnails significantly improve click-through rate

**Additional tips:**
- Add a watermark in YouTube Studio → Branding
- Create channel art and a compelling channel description
- Pin a comment on each video with your channel link
- Reply to early comments to boost engagement signals
- Use YouTube's built-in end screens (add via Studio after upload)

---

## 🗂️ Project Structure

```
yt-auto-uploader/
├── .github/
│   └── workflows/
│       └── auto_upload.yml      ← GitHub Actions (runs 5×/day)
├── src/
│   ├── main.py                  ← Pipeline orchestrator
│   ├── fetch_trending.py        ← YouTube trending songs fetcher
│   ├── process_audio.py         ← Slowed + Reverb audio engine
│   ├── create_video.py          ← Animated video generator
│   ├── generate_thumbnail.py    ← YouTube thumbnail creator
│   ├── seo_generator.py         ← Title + description + tags
│   ├── upload_youtube.py        ← YouTube Data API uploader
│   └── utils.py                 ← Shared helpers
├── output/
│   └── uploaded.json            ← Tracks uploaded video IDs
├── get_refresh_token.py         ← One-time OAuth setup (run locally)
├── requirements.txt
└── README.md
```

---

## ⚖️ Legal Notes

- All videos include full original artist credits and copyright disclaimer
- The slowed+reverb transformation may qualify as fair use / transformative work
- If any artist or label sends a copyright claim, the video may be demonetized
  (not deleted) — this is normal and still counts toward watch hours
- Consider adding tracks from royalty-free sources (NCS, Artlist) for
  guaranteed monetization without claims

---

## 🛠️ Customization

**Change slow factor** → `process_audio.py` → `SLOW_FACTOR = 0.80`  
**Change reverb intensity** → `process_audio.py` → `REVERB_WET = 0.35`  
**Change color palettes** → `create_video.py` → `PALETTES`  
**Upload timing** → `.github/workflows/auto_upload.yml` → `cron` lines  
**Video privacy** → GitHub secret `VIDEO_PRIVACY` = `public` / `unlisted`

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| `quotaExceeded` | YouTube API has 10,000 units/day. Uploads cost ~1,600. 5 uploads = 8,000. Should be fine. |
| `Token refresh failed` | Re-run `get_refresh_token.py` and update `YT_REFRESH_TOKEN` secret |
| `yt-dlp: Video unavailable` | Song was removed from YouTube — pipeline will skip and retry next run |
| `ffmpeg not found` | Already installed in Actions — for local testing: `brew install ffmpeg` |
| Upload stuck at 0% | Check your `YT_CLIENT_ID` / `YT_CLIENT_SECRET` are correct |

---

## ⚖️ Disclaimer

- This tool is provided for educational and personal use only.
- You are responsible for complying with YouTube’s Terms of Service, copyright laws, and all applicable local regulations.
- Automated uploading of copyrighted content may lead to demonetization, copyright claims, or account penalties.
- The creator (zerotouchai.official@gmail.com) is not liable for any misuse, legal issues, or damages resulting from the use of this software.
- Always ensure you have proper rights or licenses for the content you upload.


Made with ❤️ — fully automated, zero touch needed after setup.
For queries or support, contact: zerotouchai.official@gmail.com
