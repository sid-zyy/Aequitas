"""Whole-recording summary: grounding of key points, long timelines, and the bodycam endpoint (fake model)."""

from aequitas.understand.recording_summary import (KeyPoint, RecordingSummary, ground, parse_mmss,
                                                   summarize_recording, timeline_lines)

WINDOWS = [{"t": 0.0, "label": "none", "summary": "An officer talks to a man near a car."},
           {"t": 4.0, "label": "none", "summary": "An officer talks to a man near a car."},
           {"t": 8.0, "label": "arrest", "summary": "The officer handcuffs the man."},
           {"t": 12.0, "label": None, "summary": None}]


class FakeLlm:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def structured(self, messages, model_cls, retries=1, max_tokens=None):
        self.prompts.append(messages[0]["content"])
        return self.answer


def test_timeline_merges_repeats_and_skips_failures():
    lines = timeline_lines(WINDOWS)
    assert lines == ["00:00 [none] An officer talks to a man near a car.", "00:08 [arrest] The officer handcuffs the man."]


def test_long_timelines_keep_every_incident():
    ws = [{"t": i * 4.0, "label": "fighting" if i % 50 == 0 else "none", "summary": f"scene {i}"} for i in range(400)]
    lines = timeline_lines(ws)
    assert len(lines) <= 80 and sum("[fighting]" in ln for ln in lines) == 8


def test_key_points_at_times_without_a_window_are_dropped():
    s = RecordingSummary(overview=" An officer stops a man and handcuffs him. ", key_points=[
        KeyPoint(time="00:09", point="Handcuffing"), KeyPoint(time="05:00", point="Invented"),
        KeyPoint(time="soon", point="Not a time"), KeyPoint(time="00:00", point="Conversation")])
    g = ground(s, WINDOWS)
    assert [k.time for k in g.key_points] == ["00:00", "00:09"] and g.overview.startswith("An officer")
    assert parse_mmss("1:02:03") == 3723
    # consecutive windows of the same scene: only the first point of a run stays
    run = ground(RecordingSummary(overview="o", key_points=[KeyPoint(time=f"00:{t:02d}", point="p")
                                                            for t in (0, 4, 8)]), WINDOWS)
    assert [k.time for k in run.key_points] == ["00:00", "00:08"]


def test_summarize_recording_passes_confirmed_moments_and_reports_dropped():
    llm = FakeLlm(RecordingSummary(overview="o", key_points=[KeyPoint(time="00:08", point="cuffs"),
                                                             KeyPoint(time="09:59", point="x")]))
    out, ms = summarize_recording(llm, WINDOWS, ["00:08 arrest"])
    assert out["key_points"] == [{"time": "00:08", "t": 8, "point": "cuffs"}] and out["dropped"] == 1
    assert "confirmed these moments: 00:08 arrest" in llm.prompts[0] and "religion" in llm.prompts[0]
    assert summarize_recording(llm, [], None) == (None, 0.0)  # nothing to summarise: no call


def test_violence_model_labels_are_marked_so_the_text_can_say_where_they_came_from():
    ws = [{"t": 0, "label": "fighting", "summary": "People stand in a room.", "violence": 0.996, "source": "clip_model"},
          {"t": 4, "label": "fighting", "summary": "Two people push each other.", "violence": 0.97, "source": "vlm"},
          {"t": 8, "label": "arrest", "summary": "A room.", "violence": 0.95, "source": "vlm+clip"}]
    assert timeline_lines(ws) == ["00:00 [fighting; violence model 1.00] People stand in a room.",
                                  "00:04 [fighting] Two people push each other.", "00:08 [arrest] A room."]


def test_headline_states_flags_review_state_and_quiet_stretches():
    from aequitas.understand.recording_summary import headline

    ws = [{"t": 0, "end": 4, "label": "none", "summary": "A man walks to a car.", "source": "vlm"},
          {"t": 4, "end": 8, "label": "none", "summary": "People stand by a car.", "source": "vlm"},
          {"t": 8, "end": 12, "label": "fighting", "summary": "Movement.", "source": "clip_model"},
          {"t": 12, "end": 16, "label": "fighting", "summary": "Two men grapple.", "source": "both"},
          {"t": 16, "end": 18, "label": "harassment", "summary": "One leans on the other.", "source": "both"}]
    ev = [{"t": 8, "review": None}]
    h = headline(ws, ev)
    assert h == ("Nothing found from 00:00 to 00:08. Fighting flagged from 00:08 to 00:18 (also harassment) "
                 "(not yet confirmed by an officer).")
    assert "confirmed by an officer)" in headline(ws, [{"t": 8, "review": "confirmed"}])
    assert "(dismissed by an officer)" in headline(ws, [{"t": 8, "review": "dismissed"}])
    only_motion = [{"t": 0, "end": 4, "label": "fighting", "summary": "People in a room.", "source": "clip_model"}]
    assert headline(only_motion, []).startswith("Fighting flagged at 00:00 (not yet confirmed by an officer; seen by "
                                                "the violence model")
    assert headline([{"t": 0, "end": 4, "label": "none", "summary": "A road."},
                     {"t": 4, "end": 8, "label": "none", "summary": "A road."}], []) == "No incident found in the 00:08 analysed."
    assert headline([], []) == ""




def test_filler_descriptions_of_incidents_are_not_shown_to_the_summary_model():
    from aequitas.understand.recording_summary import plain_incidents

    ws = [{"t": 0, "label": "fighting", "summary": "A person is performing physical movements in a confined space."},
          {"t": 4, "label": "fighting", "summary": "Two men grapple on the ground."},
          {"t": 8, "label": "none", "summary": "A person is seen in various poses."}]
    out = plain_incidents(ws)
    assert out[0]["summary"] == "The AI flagged fighting here."
    assert out[1]["summary"] == "Two men grapple on the ground." and out[2]["summary"] == ws[2]["summary"]
    assert ws[0]["summary"].startswith("A person is performing")  # the stored windows are not changed
