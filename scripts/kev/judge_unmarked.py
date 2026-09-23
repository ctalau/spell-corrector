#!/usr/bin/env python3
"""Stage 2 of the unmarked-markup audit: kev decides what each regex candidate really is.

One `choice` question per candidate, over the four things a span of prose in the guide can be: a UI label
(<uicontrol>), a file or path (<filepath>), code (<codeph>), or plain prose. The regex only nominates; the finding is
kept when kev's argmax is a markup element. The same question is asked of spans the writers *did* mark up
(find_unmarked.py --gold), shown as plain text, which is how accuracy per element is measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

KEV = Path(os.environ.get("KEV_DIR", "/home/user/jaredpalmer/kev"))
sys.path.insert(0, str(KEV))

# Fixed before the first run; not tuned on the validation rows.
INSTRUCTIONS = ("This is a paragraph from a software user guide written in DITA. The span \"{match}\" is marked "
                "[[like this]] in the text. Which DITA inline element should wrap that span?")
OPTIONS = {
    "uicontrol": "the label of a user-interface control in the application: a button, menu, menu item, tab, check box, "
                 "option, field, view, pane, dialog box, wizard page or preferences page the reader clicks, selects or sees",
    "filepath": "the name of a file or folder, or a path to one",
    "codeph": "code or markup: an XML element or attribute name, a function, method or class, a variable, a parameter "
              "or property name, a command-line option, or a literal value the reader types",
    "plain": "no element, ordinary prose: a product, technology, standard, format or document type named in passing, "
             "or a normal English word or phrase",
}
# One yes/no question per element, asked of every span. The choice above spreads one unit of probability over four
# answers and in the first run it called 3 in 4 real <uicontrol> spans "plain"; separate yes/no questions let each
# element be thresholded on its own.
NOUL = {
    "uicontrol": "Is \"{match}\" the name of a specific control or area of the application's user interface, such as a "
                 "button, menu, menu item, tab, check box, option, field, view, pane, toolbar, dialog box, wizard page "
                 "or preferences page?",
    "filepath": "Is \"{match}\" the name of a file or folder, or a path to one?",
    "codeph": "Is \"{match}\" literal code: an element, attribute, property, parameter, variable, function, method or "
              "class name, a command-line option, or a value to be typed exactly as written?",
}
WINDOW = 700  # characters of context kept around the marked span


def clip(context: str) -> str:
    i = context.find("[[")
    j = context.find("]]", i) + 2
    if len(context) <= WINDOW: return context
    pad = max(0, (WINDOW - (j - i)) // 2)
    a, b = max(0, i - pad), min(len(context), j + pad)
    return ("…" if a else "") + context[a:b] + ("…" if b < len(context) else "")


def main() -> int:
    import torch  # here, not at the top, so judge_unmarked_jev.py can import the questions without torch

    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from kev.api import SystemOneRequest, to_answers, to_record
    from kev.checkpoint import Checkpoint, LoadOptions

    torch.set_num_threads(os.cpu_count() or 4)
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit: rows = rows[: a.limit]
    done = set()
    out = Path(a.out)
    if out.exists():  # resumable
        done = {(r["file"], r["line"], r["context"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}

    t0 = time.perf_counter()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    tok, model = Checkpoint(a.run).load(a.device, LoadOptions(dtype=dtype))
    print(f"loaded {a.run} in {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if (r["file"], r["line"], r["context"]) in done: continue
            qs = {"tag": {"type": "choice", "instructions": INSTRUCTIONS.format(match=r["match"]), "criteria": OPTIONS}}
            qs.update({f"is_{el}": {"type": "noul", "instructions": q.format(match=r["match"])} for el, q in NOUL.items()})
            req = SystemOneRequest(state=clip(r["context"]), questions=qs)
            rec, meta = to_record(req)
            enc = model.encode(tok, rec, max_state=1024, max_branch=2048)
            s = time.perf_counter()
            with torch.no_grad():
                probs = model.probs(enc)
            ms = 1000 * (time.perf_counter() - s)
            answers = to_answers([p.float().tolist() for p in probs], meta)
            ans = answers["tag"]
            r = {**r, "kev": ans["choice"], "kev_confidence": ans["confidence"], "kev_probs": ans["probabilities"],
                 **{f"p_{el}": answers[f"is_{el}"]["noul"] for el in NOUL}, "latency_ms": round(ms, 1)}
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 25 == 0: print(f"{i + 1}/{len(rows)} {ms:.0f}ms", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
