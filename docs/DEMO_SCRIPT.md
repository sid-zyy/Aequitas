# Live demo script: citizen upload → officer review, then a staged fight on the phone camera

## The clip to upload: `data/footage/demo/flyover-attack.mp4`

A man is attacked outdoors, under a flyover, with a calm stretch first, so the timeline shows both "nothing found"
and the incident. It is staged stock footage: two Pexels clips from the same shoot (same place, same actors)
joined by `scripts/make_demo_clip.py`: 12 s of a man walking up to a group by a car, then the fight (33.7 s).
Checked end to end on 2026-10-17 with the shipped settings (scratch database, RTX 4050, qwen3-vl:2b-instruct 4-bit):

| Screen | What it shows (measured) |
|---|---|
| Queue | AI found **FIGHTING**; AI-generated **Real** (see below) |
| Analysis | 00:00–00:08 **nothing found**; 00:12 harassment (unconfirmed: it calls the setting "a gym"); 00:16–00:32 fighting (violence model 0.95–1.00, Qwen agrees on every window), event **confirmed**, analysed in **23 s** |
| Timeline bar | green, then red from 00:12 |
| Summary | Headline "Nothing found from 00:00 to 00:12. Fighting flagged from 00:12 to 00:34 (also harassment) (not yet confirmed by an officer)." Windows read "Two men are wrestling under a bridge…" |

**The AI-generated check.** With no C2PA or metadata evidence only the frame detector speaks, and the screens
show words, not a percentage. The demo clip scores 0.03 on it, so the queue reads **Real** (the detail view says
"Frame check: no AI signs found, treated as real"). By design it prefers missing a fake to flagging a real
citizen's video: a false "AI-generated" label would discredit a genuine report. It flags "AI-like frames" only above 0.85, which in our test caught
34 of 50 AI clips (all Kling; it misses open models like CogVideoX) and 1 of 80 real clips
(results/authenticity/README.md). Until 2026-10-16 the old detector showed "100 % ?" on every stock clip.

**Backup clip:** `data/footage/demo/citizen-upload.mp4`, faculty clip **v011** (night-time restaurant brawl, 8.7 s,
2/2 windows fighting, AI-generated Real, 10 s). With Qwen its summary is good: "A group of men are fighting in a
restaurant, with one man pushing another and both falling to the ground." Gemma E2B never saw that fight.

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

1. The report appears within about 3 s, with **FIGHTING** and the AI-generated check (*Real*). Click **Start review**.
2. **▶ Analyse** opens the summary, player, timeline, key moment and chat. The analysis takes about 16 s, so talk
   over it.
3. On the key moment, click **Confirm**. It shows "confirmed by <officer>" and goes to the audit trail.
4. In the chat, ask **"What happens in this video?"** (~1 s, with seek buttons) and **"Who is the attacker?"**.
   The second answers that the analysis doesn't identify the attacker; that's the guardrail, so it's worth
   showing. "What happens before the fight?" works but with Qwen often skips the calm 00:00–00:12, so it's optional.
   Rehearse once: Qwen's chat wording varies more than Gemma's did.
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
