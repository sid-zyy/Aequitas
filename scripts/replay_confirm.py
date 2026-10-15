"""Replay the saved per-window labels of the demo runs through different confirmation rules (no GPU, no noise)."""
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
expected = yaml.safe_load((root / "config/demo_expected.yaml").read_text())
runs = ["demo", "demo_rep", "demo_ctx", "demo_cascade", "demo_gate", "demo_gate_rep"]


def windows(db):
    con = sqlite3.connect(db)
    out = {}
    for cam, s, lab in con.execute("select camera_id, start_ts, label from windows order by camera_id, start_ts"):
        out.setdefault(cam, []).append(lab)
    return out


def same_consecutive(labs, k=2):  # current rule: k windows in a row with the same label
    conf, run, prev = set(), 0, None
    for l in labs:
        if l is None:
            continue  # failure: neither opens nor closes
        if l == "none":
            run, prev = 0, None
            continue
        run = run + 1 if l == prev else 1
        prev = l
        if run >= k:
            conf.add(l)
    return conf


def same_2of3(labs):  # same label in 2 of any 3 consecutive windows
    labs = [l for l in labs if l is not None]
    conf = set()
    for i in range(len(labs)):
        c = Counter(l for l in labs[i:i + 3] if l != "none")
        conf |= {l for l, n in c.items() if n >= 2}
    return conf


def any_incident_consecutive(labs):  # 2 incident windows in a row, any labels; event label = the first one
    labs = [l for l in labs if l is not None]
    conf = set()
    for a, b in zip(labs, labs[1:]):
        if a != "none" and b != "none":
            conf.add(a)
    return conf


rules = {"current (2 in a row, same label)": same_consecutive, "2 of 3, same label": same_2of3,
         "2 in a row, any incident label": any_incident_consecutive}

for name, rule in rules.items():
    print(f"\n### {name}")
    print("run | caught | false alarms | missed")
    for r in runs:
        db = root / "data/eval" / r / "aequitas.db"
        w = windows(db)
        hit = fa = 0
        missed, fas = [], []
        req_total = 0
        for clip, exp in expected.items():
            req, opt = set(exp.get("required", [])), set(exp.get("optional", []))
            got = rule(w.get(clip, []))
            req_total += len(req)
            hit += len(req & got)
            missed += [f"{clip}" for _ in req - got]
            bad = got - req - opt
            fa += len(bad)
            fas += [f"{clip}:{b}" for b in bad]
        print(f"{r} | {hit}/{req_total} | {fa} {fas} | {missed}")


print("\n\n### clip-model vote: a window with violence >= T also votes 'fighting' (2 in a row)")
for T in (0.9, 0.95):
    print(f"\nT = {T}\nrun | caught | false alarms | missed")
    for r in runs:
        con = sqlite3.connect(root / "data/eval" / r / "aequitas.db")
        w = {}
        for cam, lab, v in con.execute("select camera_id, label, violence from windows order by camera_id, start_ts"):
            if lab is None:
                continue
            votes = set() if lab == "none" else {lab}
            if v is not None and v >= T:
                votes.add("fighting")
            w.setdefault(cam, []).append(votes)
        hit = fa = tot = 0
        missed, fas = [], []
        for clip, exp in expected.items():
            req, opt = set(exp.get("required", [])), set(exp.get("optional", []))
            seq = w.get(clip, [])
            got = set().union(*[a & b for a, b in zip(seq, seq[1:])]) if len(seq) > 1 else set()
            tot += len(req); hit += len(req & got); missed += list(req - got and [clip])
            bad = got - req - opt; fa += len(bad); fas += [f"{clip}:{b}" for b in bad]
        print(f"{r} | {hit}/{tot} | {fa} {fas} | {missed}")
    hi = con.execute("select camera_id, count(*) from windows where violence >= ? group by camera_id", (T,)).fetchall()
    print("windows >= T in last run, per camera:", hi)
