#!/usr/bin/env python3
"""Writes the export side of the XeSS proxy (nr_xess.c) from libxess.exports.

The proxy has to export every function Intel's libxess.dll exports, under the same names and
ordinals: the game looks them up by name, and Intel's driver calls the INTC_* ones. Four are the
proxy's own (OWN below). Each of the others is a jump through a pointer that DllMain fills from
the original, so every register and stack slot reaches it as the caller left them. A jump, not
a wrapper with a prototype as in the vulkan-1.dll proxy, because most of them have no public
prototype (the INTC_* driver callbacks, xessAIL*, xessTrinity*). Not a forwarder either:
MSVC's link.exe takes `name=dll.name` in a .def for an unresolved external.

    python src/xess/gen_xess_exports.py [--check]

Writes nr_xess.def, nr_xess_thunks.asm and nr_xess_exports.h beside this file; --check only
says whether they are what libxess.exports gives.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OWN = {"xessD3D12Execute", "xessD3D12Init", "xessSetJitterScale", "xessSetVelocityScale"}
STAMP = "Written by gen_xess_exports.py from libxess.exports; not edited by hand."


def exports():
    rows = []
    for line in (HERE / "libxess.exports").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            ordinal, name = line.split()
            rows.append((int(ordinal), name))
    return rows


def render(rows):
    forwarded = [name for _, name in rows if name not in OWN]
    index = {name: i for i, name in enumerate(forwarded)}
    definition = [f"; {STAMP}", "LIBRARY libxess", "EXPORTS"]
    for ordinal, name in rows:
        target = name if name in OWN else f"{name}=nr_forward_{name}"
        definition.append(f"    {target} @{ordinal}")
    thunks = [f"; {STAMP}",
              "; Each jumps through the pointer DllMain set from Intel's libxess.dll.",
              ".data", "PUBLIC nr_xess_forward",
              f"nr_xess_forward QWORD {len(forwarded)} DUP (0)", ".code"]
    for name in forwarded:
        thunks += [f"nr_forward_{name} PROC",
                   f"\tjmp QWORD PTR [nr_xess_forward + {8 * index[name]}]",
                   f"nr_forward_{name} ENDP"]
    thunks.append("END")
    header = [f"/* {STAMP}", " * The exports nr_xess_thunks.asm jumps to, in its pointers' order. */",
              f"#define NR_XESS_FORWARDED {len(forwarded)}",
              "extern void *nr_xess_forward[NR_XESS_FORWARDED];",
              "static const char *const nr_xess_forwarded[NR_XESS_FORWARDED] = {"]
    header += [f'\t"{name}",' for name in forwarded]
    header.append("};")
    return {"nr_xess.def": definition, "nr_xess_thunks.asm": thunks, "nr_xess_exports.h": header}


def main():
    files = render(exports())
    stale = []
    for name, lines in files.items():
        text = "\n".join(lines) + "\n"
        path = HERE / name
        if "--check" in sys.argv:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
    if stale:
        print("stale: " + ", ".join(stale))
        return 1
    print("up to date" if "--check" in sys.argv else "wrote " + ", ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
