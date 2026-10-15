"""Faculty dataset adapter: folder-per-label and manifest layouts, label mapping, end-to-end eval."""

import json

import cv2
import numpy as np

from aequitas.eval.dataset import clip_grid, load_dataset, map_labels, rouge_l, run_dataset_eval
from aequitas.understand.prompts import SummaryAnswer


def _video(path, n=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), i * 20 % 255, np.uint8))
    vw.release()
    return path


def test_folder_per_label_layout(tmp_path):
    _video(tmp_path / "Fight" / "a.mp4")
    _video(tmp_path / "Normal" / "b.avi")
    items = load_dataset(tmp_path)
    assert sorted((i.path.name, i.label) for i in items) == [("a.mp4", "Fight"), ("b.avi", "Normal")]
    items, cats = map_labels(items, None)
    assert cats == {"fight": "fight"} and sorted(i.label for i in items) == ["fight", "none"]


def test_csv_manifest_with_loose_column_names_and_ids_without_extension(tmp_path):
    _video(tmp_path / "clips" / "001.mp4")
    _video(tmp_path / "clips" / "002.mp4")
    (tmp_path / "labels.csv").write_text(
        "Video_ID,Class,Caption\nclips/001,Eve Teasing,A man follows a woman.\nclips/002.mp4,normal,People walk.\n",
        encoding="utf-8")
    items = load_dataset(tmp_path)  # the single manifest is found automatically
    assert [i.path.name for i in items] == ["001.mp4", "002.mp4"]
    assert items[0].summary == "A man follows a woman."
    items, cats = map_labels(items, {"Eve Teasing": "harassment"})
    assert [i.label for i in items] == ["harassment", "none"] and cats is None


def test_json_and_jsonl_manifests(tmp_path):
    _video(tmp_path / "x.mp4")
    (tmp_path / "m.json").write_text(json.dumps({"videos": [{"file": "x.mp4", "label": "fire"}]}), encoding="utf-8")
    assert load_dataset(tmp_path, tmp_path / "m.json")[0].label == "fire"
    (tmp_path / "m.jsonl").write_text(json.dumps({"path": "x.mp4", "category": "riot"}) + "\n", encoding="utf-8")
    assert load_dataset(tmp_path, tmp_path / "m.jsonl")[0].label == "riot"


def test_rouge_l():
    assert rouge_l("a man walks", "a man walks") == 1.0 and rouge_l("x", "y") == 0.0


class FakeVlm:
    def summarize(self, grid, highlighted=True, view="cctv"):
        return SummaryAnswer(summary="A man follows a woman along the road.", visible_evidence="following"), 900.0


class FakeClf:
    def classify_text(self, summary, evidence="", categories=None, previous=None):
        assert categories == {"harassment": "x"}
        return "harassment", 0.8, 30.0


def test_eval_runs_every_video_and_reports_accuracy_and_time(tmp_path):
    _video(tmp_path / "harassment" / "a.mp4")
    _video(tmp_path / "harassment" / "b.mp4")
    bad = tmp_path / "none" / "broken.mp4"
    bad.parent.mkdir()
    bad.write_bytes(b"nope")
    assert clip_grid(bad) is None
    items, _ = map_labels(load_dataset(tmp_path), None)
    seen = []
    r = run_dataset_eval(FakeVlm(), FakeClf(), items, {"harassment": "x"}, out_dir=tmp_path / "rep",
                         on_progress=lambda done, total, row: seen.append((done, total, row["video"])))
    assert [(d, t) for d, t, _ in seen] == [(1, 3), (2, 3), (3, 3)]  # once per video, unreadable ones too
    assert len({v for _, _, v in seen}) == 3
    assert r["view"] == "cctv"
    assert r["n_videos"] == 3 and r["classification"]["accuracy"] == 1.0
    assert r["total_execution_s"] >= 0 and r["per_video_s"]["p50"] >= 0
    assert any(row.get("error") == "unreadable video" for row in r["rows"])
