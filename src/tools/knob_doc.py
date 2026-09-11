#!/usr/bin/env python3
"""Render the knob table as markdown, so the manual cannot drift from the program.

    python3 src/tools/knob_doc.py            print it
    python3 src/tools/knob_doc.py --check    exit 1 if README.md is out of date
    python3 src/tools/knob_doc.py --write    put it back into README.md

`test_toggle.py` runs the check, so a knob added to `nr_knobs` and not to the README is
a failing test rather than a surprise for whoever reads the README next.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "layer"))
import nr_knobs  # noqa: E402

README = ROOT / "README.md"
BEGIN, END = "<!-- knobs:begin -->", "<!-- knobs:end -->"


def markdown():
    lines = [BEGIN, ""]
    for knob in nr_knobs.KNOBS:
        if knob.kind == "choice":
            span = " / ".join(f"`{name}`" for name in nr_knobs.PROFILES)
        else:
            span = (f"`{knob.low:g}` to `{knob.high:g}`, step `{knob.step:g}`, "
                    f"default `{knob.default:g}`")
        lines += [f"### `{knob.name}` — {knob.summary}", "", span, "",
                  knob.detail.replace(" -> ", " → "), ""]
    lines.append(END)
    return "\n".join(lines)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "--print"
    block = markdown()
    if what == "--print":
        print(block)
        return 0
    text = README.read_text() if README.exists() else ""
    if BEGIN not in text or END not in text:
        print(f"{README} has no {BEGIN} ... {END} block")
        return 1
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    fresh = head + block + tail
    if what == "--check":
        if fresh != text:
            print("README.md is out of date: python3 src/tools/knob_doc.py --write")
            return 1
        return 0
    if what == "--write":
        README.write_text(fresh)
        print(f"wrote the knob table into {README}")
        return 0
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
