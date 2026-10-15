"""Builds Kaggle .ipynb notebooks from the `# %%`-cell .py sources in this folder.

The .py files are the source of truth (reviewable diffs, runnable locally with AEQUITAS_SMOKE=1);
the .ipynb files are what you upload to Kaggle. Run: python kaggle/build_notebooks.py"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def cells_from(source: str) -> list[dict]:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        text = [ln + "\n" for ln in lines]
        if text:
            text[-1] = text[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": text}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def build(py: Path) -> Path:
    nb = {
        "cells": cells_from(py.read_text(encoding="utf-8")),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = py.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for py in sorted(HERE.glob("*.py")):
        if py.name != Path(__file__).name:
            print("built", build(py))
