# Live demo script: citizen upload → officer review, then a staged fight on the phone camera

Checked end to end on 2026-10-16 on a scratch database (RTX 4050, gemma4:e2b, MTP draft 3).

## The clip to upload: `data/footage/demo/citizen-upload.mp4`

This is faculty clip **v011**: a night-time restaurant brawl, 8.7 s of CCTV with the timestamp 01:34. Of everything
we tried, it was the only clip where every screen agreed:

| Screen | What it shows (measured) |
|---|---|
| Queue | AI found **FIGHTING**, AI-generated **0 %** |
| Analysis | 2/2 windows fighting (violence model 0.999 / 1.000), event confirmed, analysed in **7.5 s** (0.68 s summary + 0.38 s label per window) |
| Summary | Gemma's description, then "The violence model, which watches motion, flagged fighting at 00:00, 00:04." |
| Chat | "What happens in this video?" names the flags with seek buttons, ~1 s |

Clips rejected, and why:
- **v026** (night corridor, the closest to the story): only 1 of 3 windows labelled fighting, the other two
  suspicious activity, and the picture is too dark for an audience.
- **v001**: daytime, so it doesn't fit the story.
- **The 4 Pexels college harassment clips**: no incident flagged. The collar grab scored only 0.5 on the violence
  model, and Gemma described "two individuals standing".
- **punching.mp4**: one man shadow-boxing, so not an attack.

Gemma E2B's own text never describes these fights (CCTV is too low-detail for it; E4B was no better). The fight is
caught by the violence model, and the summary and chat now say so explicitly.

## What to type in the citizen portal (`/report`)

- Video: `data/footage/demo/citizen-upload.mp4`
- What happened: *Late at night a fight broke out inside a restaurant near our hostel. Men were punching and
  pushing each other between the tables.*
- Where: *Outer Ring Road, Marathahalli*
- Tick consent → Submit. Read out the report ID; the tracking code is shown once.
- Optional: open the red **Help** button first and answer **Yes** to show the 112 / 100 / 1091 numbers.

## Officer side (`/requests`)

1. The report appears within about 3 s, with **FIGHTING** and **0 %**. Click **Start review**.
2. **▶ Analyse** opens the summary, player, timeline, key moment and chat. The analysis takes about 8 s, so talk
   over it.
3. On the key moment, click **Confirm**. It shows "confirmed by <officer>" and goes to the audit trail.
4. In the chat, ask **"What happens in this video?"** Optionally ask **"Who is the attacker?"**: it answers that the
   analysis can't identify people. That's the guardrail, so worth showing.
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
