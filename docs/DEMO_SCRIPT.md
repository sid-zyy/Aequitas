# Live demo script: citizen upload → officer review, then a staged fight on the phone camera

## The clip to upload: `data/footage/demo/flyover-attack.mp4`

A man is attacked outdoors, under a flyover, with a calm stretch first, so the timeline shows both "nothing found"
and the incident. It is staged stock footage: two Pexels clips from the same shoot (same place, same actors)
joined by `scripts/make_demo_clip.py`: 12 s of a man walking up to a group by a car, then the fight (33.7 s).
Checked end to end on 2026-10-16 (scratch database, RTX 4050, gemma4:e2b):

| Screen | What it shows (measured) |
|---|---|
| Queue | AI found **FIGHTING**; AI-generated **inconclusive** (see below) |
| Analysis | 00:00–00:12 **nothing found** (4 windows), 00:16–00:32 fighting (violence model 0.95–1.00, Gemma agrees on 3), event **confirmed**, analysed in **16 s** |
| Timeline bar | green, then red from 00:16 |
| Summary | "Several people in an outdoor parking lot area. At 00:16 physical movements… a physical struggle or grappling between two individuals from 00:20 to 00:24…" |

**The AI-generated check.** With no C2PA or metadata evidence only the frame detector speaks, and the screens
show words, not a percentage. The demo clip scores 0.03 on it, so the queue reads **inconclusive** (the detail
view says "Frame check: inconclusive" and why). Say it on stage: a frame detector can't prove a video is real,
so we don't show a number it can't back up. It flags "AI-like frames" only above 0.85, which in our test caught
34 of 50 AI clips (all Kling; it misses open models like CogVideoX) and 1 of 80 real clips
(results/authenticity/README.md). Until 2026-10-16 the old detector showed "100 % ?" on every stock clip.

**Backup clip:** `data/footage/demo/citizen-upload.mp4`, faculty clip **v011** (night-time restaurant brawl, 8.7 s,
2/2 windows fighting, AI-generated inconclusive, 7.5 s). Its summary is weak ("people are present in a room…"): Gemma E2B
never sees the fight on that low-detail CCTV, only the violence model does.

Clips rejected, and why:
- **The teenagers' fight (Pexels 10505848)**: fighting confirmed, but no calm part, and the people are teenagers.
- **The full 51 s "man with a chain" clip (10400886)**: a good "nothing found" clip, but the violence model gave one
  unconfirmed false alarm at 00:36 (a handshake, 0.92). Only its first 12 s are used.
- **v026** (night corridor): only 1 of 3 windows labelled fighting, and too dark for an audience.
- **The 4 Pexels college harassment clips**: no incident flagged (collar grab 0.5 on the violence model).
- **punching.mp4**: one man shadow-boxing, so not an attack.

## What to type in the citizen portal (`/report`)

- Video: `data/footage/demo/flyover-attack.mp4`
- What happened: *Two men started fighting under the flyover near our hostel. One was punching and grabbing the
  other.*
- Where: *Outer Ring Road, Marathahalli*
- Tick consent → Submit. Read out the report ID, then "Track this report" opens the tracking tab with it filled in.
- Optional: open the red **Help** button first and answer **Yes** to show the 112 / 100 / 1091 numbers.

## Officer side (`/requests`)

1. The report appears within about 3 s, with **FIGHTING** and the AI-generated check (*inconclusive*). Click **Start review**.
2. **▶ Analyse** opens the summary, player, timeline, key moment and chat. The analysis takes about 16 s, so talk
   over it.
3. On the key moment, click **Confirm**. It shows "confirmed by <officer>" and goes to the audit trail.
4. In the chat, ask **"What happens in this video?"** (~1 s, with seek buttons) and **"What happens before the
   fight?"** (walks through the calm 00:00–00:12). Optionally ask **"Who is the attacker?"**: it answers that it
   can't determine that from the analysis. That's the guardrail, so worth showing.
5. Click **Action taken**, then **Close**.

Don't ask "Does the video match the report?" on stage. The honest answer is mixed: it gives the flags, then says
the descriptions don't show punching.

## Then: the staged fight on the phone camera (Live CCTV → "Live demo")

- **Phone:** IP Webcam app, 640×480, mounted high and angled down, both people full-body in frame. Update the
  `demo-phone` address in `config/bangalore.yaml` at the venue (it changes with the network).
- **Before going on stage:** start the app with `.venv\Scripts\python -m aequitas web` and open Live CCTV about a
  minute early. The first analysis takes about 40 s while the models load.
- **During the fight:** keep it going for at least 4–6 s of continuous pushing and grappling. The model looks at
  2 s windows, and the violence model must reach 0.9 on its own to raise "fighting". Stand and talk first, so the
  audience sees "Normal activity" before the alert.
- Rehearse it once in the room. On unseen CCTV the violence model raised false alarms on 14 % of normal clips at
  0.9.
