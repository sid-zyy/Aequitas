# Incident clips for the Bengaluru Live CCTV cameras (2026-10-17)

The Live CCTV localities showed four staged arrests and one fire. E2B misses arrest-1/2/3 in most runs. To show
other incident types, ten free Pexels clips were run through the demo pipeline. The setup was
`config/eval_incident_candidates.yaml` with E2B, the scene gate and the violence model.
`config/incident_candidates_cameras.yaml` lists the clips and `config/incident_candidates_expected.yaml` their
expected labels. A clip goes on a camera only if it gets the right label.

| Clip (Pexels id) | Expected | Result | Used |
|---|---|---|---|
| Aerial building fire with fire engines (11574594, 124 s) | fire | **fire, confirmed** | Whitefield Industrial Area (`incidents/fire-building.mp4`) |
| Firefighters hosing a burning house (6741632, 9 s) | fire | **fire, confirmed** | no: a second fire, kept as a spare |
| Smoke over a building, firefighters on a ladder (6746452) | fire | nothing | no |
| Crashed car on a street, handheld (3974558, 21 s) | accident | **accident, confirmed** | Jyoti Nivas College Road (`incidents/accident-wreck.mp4`) |
| Overhead highway truck crash (12753260) | accident | nothing | no |
| Night crash scene with police in hi-vis (11387544) | accident | arrest | no |
| Graffiti sprayed on glass (7120813, 35 s) | vandalism | **vandalism, confirmed** (plus fire when he lights the spray) | Platform Road Parking Complex (`incidents/vandalism.mp4`) |
| Purse stolen from a car (854221) | theft | nothing | no |
| Car break-in (8104041) | theft | harassment (unconfirmed) | no |
| Paramedics with a person on a stretcher (8944324) | person_down | arrest | no |

The three new clips replace arrest-1, arrest-2 and arrest-3. arrest-4 (Church Street), which E2B catches,
stays. 4 of 10 candidates passed. The pipeline does not recognise theft, and it reads people in hi-vis as police
making an arrest.
