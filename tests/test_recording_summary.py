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


def test_motion_flags_name_what_only_the_violence_model_saw():
    from aequitas.understand.recording_summary import motion_flags

    ws = [{"t": 0, "label": "fighting", "source": "clip_model"}, {"t": 4, "label": "fighting", "source": "clip_model"},
          {"t": 8, "label": "suspicious_activity", "source": "vlm+clip"},  # Gemma's label: the clip model only agreed
          {"t": 12, "label": "none", "source": "clip_model"}]
    assert motion_flags(ws).startswith("The violence model, which watches motion, flagged fighting at 00:00, 00:04.")
    assert "suspicious" not in motion_flags(ws)
    assert motion_flags([{"t": 0, "label": "arrest", "source": "vlm"}]) == ""
