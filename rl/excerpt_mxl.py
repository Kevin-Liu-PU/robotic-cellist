#!/usr/bin/env python3
"""
Cut a contiguous run of PLAYED measures out of a .mxl and write it as a new .mxl.

    python rl/excerpt_mxl.py maysong.mxl --measures 0-11 --out maysong_m0-11.mxl

Measure indices are the ones the training parser uses (play order after repeat
expansion, 0-based: the `measure` field on ScoreNote), so a window found with
load_piece maps straight onto the file this writes. What the cut preserves:

  * attributes (divisions / key / time / clef) in force at the cut, copied into
    the first kept measure if it does not carry its own;
  * the dynamic marking and the tempo (<sound tempo> / metronome) in force at
    the cut, re-stated at the start of the excerpt;
  * slurs, ties and wedges that lie inside the window.

What it drops: slurs / ties / wedges that cross either edge (unmatched ends),
repeat barlines and endings (the window is already in play order), and every
other part. The output is verified by re-parsing it with load_piece and
comparing stroke count, duration, fill and dynamics against the same window
taken from the source.

 (excerpt-by-fill piece selection).
"""
from __future__ import annotations

import argparse
import copy
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "BaselineControls"))
import play_midi_pieces as PMP  # noqa: E402

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container>
  <rootfiles>
    <rootfile full-path="score.xml" media-type="application/vnd.recordare.musicxml+xml"/>
  </rootfiles>
</container>
"""


def read_score(path: Path) -> tuple[ET.ElementTree, str]:
    """Return the parsed score and the raw XML text (for the header)."""
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            root = "score.xml"
            try:
                cont = ET.fromstring(z.read("META-INF/container.xml"))
                rf = cont.find(".//rootfile")
                if rf is not None and rf.get("full-path"):
                    root = rf.get("full-path")
            except KeyError:
                pass
            text = z.read(root).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return ET.ElementTree(ET.fromstring(text)), text


def _drop_unmatched(measures: list[ET.Element], tag: str, start=("start",), stop="stop"):
    """Remove <tag type=start/stop> pairs that do not both fall inside `measures`.

    Slurs/wedges are keyed by their `number`; ties have none (key None)."""
    open_by_key: dict = {}
    to_remove: list[tuple[ET.Element, ET.Element]] = []
    for m in measures:
        for parent in m.iter():
            for el in list(parent):
                if el.tag != tag:
                    continue
                key = el.get("number")
                t = el.get("type")
                if t in start:
                    if key in open_by_key:           # nested start without stop
                        to_remove.append(open_by_key.pop(key))
                    open_by_key[key] = (parent, el)
                elif t == stop:
                    if key in open_by_key:
                        open_by_key.pop(key)         # matched inside the window
                    else:
                        to_remove.append((parent, el))   # stop whose start is before the cut
    to_remove.extend(open_by_key.values())            # starts whose stop is after the cut
    for parent, el in to_remove:
        parent.remove(el)
    return len(to_remove)


def excerpt(tree: ET.ElementTree, first: int, last: int) -> tuple[ET.Element, dict]:
    root = tree.getroot()
    part = root.find("part")
    if part is None:
        raise SystemExit("no <part> in score")
    written = part.findall("measure")
    play_order, notes = PMP.expand_repeat_order(written)
    if not (0 <= first <= last < len(play_order)):
        raise SystemExit(f"--measures {first}-{last} outside play order 0..{len(play_order) - 1}")

    # State in force at the cut, scanning play order up to `first`.
    attrs_in_force: dict[str, ET.Element] = {}
    dyn_in_force: ET.Element | None = None
    tempo_in_force: ET.Element | None = None
    for src in play_order[:first]:
        m = written[src]
        for a in m.findall("attributes"):
            for child in a:
                attrs_in_force[child.tag] = child
        for d in m.findall("direction"):
            if d.find(".//dynamics") is not None:
                dyn_in_force = d
            snd = d.find(".//sound")
            if (snd is not None and snd.get("tempo")) or d.find(".//metronome") is not None:
                tempo_in_force = d

    kept = [copy.deepcopy(written[src]) for src in play_order[first:last + 1]]

    # Repeat barlines / endings would be re-expanded by the parser: strip them.
    n_barline = 0
    for m in kept:
        for b in list(m.findall("barline")):
            if b.find("repeat") is not None or b.find("ending") is not None:
                m.remove(b)
                n_barline += 1

    # Attributes: the first kept measure needs divisions/key/time/clef.
    m0 = kept[0]
    a0 = m0.find("attributes")
    if a0 is None:
        a0 = ET.Element("attributes")
        m0.insert(0, a0)
    have = {c.tag for c in a0}
    order = ["divisions", "key", "time", "staves", "clef", "transpose"]
    for tag in order:
        if tag not in have and tag in attrs_in_force:
            pos = 0
            for c in a0:
                if order.index(c.tag) < order.index(tag) if c.tag in order else False:
                    pos = list(a0).index(c) + 1
            a0.insert(pos, copy.deepcopy(attrs_in_force[tag]))

    # Dynamic / tempo in force: re-state before the first note unless the
    # measure already carries one before its first note.
    inserted = []
    first_note_idx = next((i for i, c in enumerate(m0) if c.tag == "note"), len(m0))
    has_dyn = any(c.tag == "direction" and c.find(".//dynamics") is not None for c in list(m0)[:first_note_idx])
    has_tempo = any(c.tag == "direction" and ((c.find(".//sound") is not None and c.find(".//sound").get("tempo"))
                                              or c.find(".//metronome") is not None) for c in list(m0)[:first_note_idx])
    at = list(m0).index(a0) + 1
    if tempo_in_force is not None and not has_tempo:
        m0.insert(at, copy.deepcopy(tempo_in_force)); inserted.append("tempo"); at += 1
    if dyn_in_force is not None and not has_dyn:
        m0.insert(at, copy.deepcopy(dyn_in_force)); inserted.append("dynamic")

    dropped = {
        "slur": _drop_unmatched(kept, "slur"),
        "tied": _drop_unmatched(kept, "tied"),
        "tie": _drop_unmatched(kept, "tie"),
        "wedge": _drop_unmatched(kept, "wedge", start=("crescendo", "diminuendo"), stop="stop"),
        "repeat_barlines": n_barline,
    }

    # Rebuild the part: only the kept measures, renumbered 1..n.
    for m in list(part):
        part.remove(m)
    for i, m in enumerate(kept):
        m.set("number", str(i + 1))
        part.append(m)
    # Other parts go (the parser reads the first part only anyway).
    for p in root.findall("part")[1:]:
        root.remove(p)
    pl = root.find("part-list")
    if pl is not None:
        keep_id = part.get("id")
        for sp in list(pl.findall("score-part")):
            if sp.get("id") != keep_id:
                pl.remove(sp)

    info = {"play_order_len": len(play_order), "repeat_notes": notes,
            "inserted": inserted, "dropped": dropped}
    return root, info


def write_mxl(root: ET.Element, header: str, out: Path, comment: str):
    body = ET.tostring(root, encoding="unicode")
    # Keep the original XML declaration and DOCTYPE, if any.
    head_lines = []
    for line in header.splitlines():
        if line.startswith("<?xml") or line.startswith("<!DOCTYPE"):
            head_lines.append(line)
        elif line.strip().startswith("<score-partwise") or line.strip().startswith("<score-timewise"):
            break
    text = "\n".join(head_lines) + f"\n<!-- {comment} -->\n" + body + "\n"
    if out.suffix.lower() == ".mxl":
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("META-INF/container.xml", CONTAINER)
            z.writestr("score.xml", text)
    else:
        out.write_text(text, encoding="utf-8")


def window_stats(path: str, first: int | None = None, last: int | None = None) -> dict:
    """Stroke count / duration / fill / dynamics of a played-measure window,
    through the same load_piece path training uses."""
    import numpy as np
    import piece_env as PE
    notes, meta, strokes = PE.load_piece(path)
    sel = [s for s in strokes if first is None or first <= notes[s.note_index].measure <= last]
    d = np.array([s.duration for s in sel])
    dyn: dict[str, int] = {}
    for s in sel:
        k = notes[s.note_index].dynamic
        dyn[k] = dyn.get(k, 0) + 1
    return {"strokes": len(sel), "seconds": round(float(d.sum()), 2),
            "fill": round(float(np.minimum(1.0, d / PE.CLASSIFIER_WINDOW_SEC).mean()), 3),
            "share_ge_0.3s": round(float((d >= 0.3).mean()), 3),
            "median_s": round(float(np.median(d)), 3), "dynamics": dict(sorted(dyn.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--measures", required=True, help="first-last, played-measure indices (0-based, inclusive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    first, last = (int(x) for x in args.measures.split("-"))
    src, out = Path(args.src), Path(args.out)

    tree, header = read_score(src)
    root, info = excerpt(tree, first, last)
    comment = (f"excerpt of {src.name}, played measures {first}-{last} (0-based, after repeat expansion); "
               f"generated by rl/excerpt_mxl.py")
    write_mxl(root, header, out, comment)
    print(f"wrote {out}  (source play order {info['play_order_len']} measures; kept {last - first + 1})")
    print(f"  re-stated at the cut: {info['inserted'] or 'nothing'};  dropped: {info['dropped']}")
    for n in info["repeat_notes"]:
        print(f"  note: {n}")
    if args.no_verify:
        return
    want = window_stats(str(src), first, last)
    got = window_stats(str(out))
    print(f"  source window : {want}")
    print(f"  excerpt file  : {got}")
    ok = all(want[k] == got[k] for k in ("strokes", "seconds", "fill", "dynamics"))
    print("  VERIFY", "OK: excerpt parses to the same strokes as the source window" if ok
          else "MISMATCH: check the cut (dynamics / slur at the edge?)")
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
