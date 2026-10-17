"""Refocus: a window only the violence clip model flagged gets a second, label-aware look by the VLM, so the
summary of a "fighting" event describes the fight, or says plainly that none is visible."""

from aequitas.understand.fusion import fuse_window
from aequitas.understand.prompts import RefocusAnswer, VlmAnswer
from aequitas.understand.stages import Understanding, needs_refocus, refocus


class FakeVlm:
    def __init__(self, ans):
        self.ans, self.calls = ans, []

    def refocus(self, grid, label, definition):
        self.calls.append((label, definition))
        return self.ans, 700.0


def first_pass(category="none"):
    s = "A group of people is gathered in an outdoor area, some standing."
    return Understanding(VlmAnswer(category=category, summary=s), s, 800.0, 380.0)


def test_refocus_only_when_the_clip_model_alone_sees_violence():
    assert needs_refocus(first_pass(), 0.98, 0.9)
    assert not needs_refocus(first_pass(), 0.85, 0.9)  # below the clip-alone bar: no fighting label anyway
    assert not needs_refocus(first_pass("fighting"), 0.98, 0.9)  # the VLM already saw it
    assert not needs_refocus(first_pass(), None, 0.9)  # no clip model
    assert not needs_refocus(first_pass(), 0.98, None)  # clip-alone labels switched off


def test_seen_fight_becomes_the_summary_and_time_is_added():
    vlm = FakeVlm(RefocusAnswer(seen=True, summary="Two men shove and punch each other between the tables."))
    u = refocus(first_pass(), vlm, b"jpeg", "fighting", "people hit each other", 0.98)
    assert u.summary.startswith("Two men shove") and u.answer.summary == u.summary
    assert u.answer.category == "none"  # the label stays the fusion rule's decision
    assert u.summarize_ms == 1500.0 and u.classify_ms == 380.0 and u.refocused == "seen"
    ans, source = fuse_window(u.answer, 0.98, 0.5, 0.9)
    assert ans.category == "fighting" and source == "clip_model"
    assert ans.summary == ("Possible fight among two men (violent motion 0.98). "
                           "Two men shove and punch each other between the tables.")


def test_not_seen_says_so_instead_of_inventing_a_fight():
    vlm = FakeVlm(RefocusAnswer(seen=False, summary="People sit and stand around tables, talking."))
    u = refocus(first_pass(), vlm, b"jpeg", "fighting", "people hit each other", 0.98)
    assert u.summary == "No fighting is clearly visible: People sit and stand around tables, talking."
    ans, _ = fuse_window(u.answer, 0.98, 0.5, 0.9)
    assert ans.summary.startswith("Possible fight (violent motion 0.98). No fighting is clearly visible")
    assert u.refocused == "not_seen"


def test_failed_second_look_keeps_the_first_summary():
    u = refocus(first_pass(), FakeVlm(None), b"jpeg", "fighting", "d", 0.98)
    assert u.summary.startswith("A group of people") and u.refocused is None and u.summarize_ms == 1500.0


def test_works_when_classification_had_failed():
    u0 = Understanding(None, "People stand around.", 800.0, None)
    u = refocus(u0, FakeVlm(RefocusAnswer(seen=True, summary="One man kicks another.")), b"j", "fighting", "d", 0.97)
    assert u.answer.summary == "One man kicks another." and u.answer.category == "none"
