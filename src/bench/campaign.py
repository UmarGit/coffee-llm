"""Bookkeeping for scripts/campaign.sh: one directory, one set of flags.

    python -m bench.campaign manifest <dir> <name> <models> [bench flags]
    python -m bench.campaign size     <dir>             # the training size to export
    python -m bench.campaign summary  <dir>             # checks, written to bundle/summary.txt

`manifest` writes <dir>/campaign.json, or refuses if the directory already holds
a campaign started with different flags: run_id does not encode the training
mix, so mixing two campaigns in one directory would silently skip cells.
"""

import json
import os
import sys
import time
from pathlib import Path

STAGES = ("prepare", "check", "smoke", "sweep", "select", "grid", "latency",
          "determinism", "test", "fewshot", "grammar", "report")


def _cfg(model, flags):
    from . import config
    from .__main__ import _cfg, parser

    config.select(model, "pizza")
    return _cfg(parser().parse_args(["grid", *flags]))


def _version():
    """push_code.sh writes VERSION, because the server has no .git."""
    from .config import REPO, git_commit

    p = REPO / "VERSION"
    return p.read_text().strip() if p.exists() else git_commit()


def _gpus():
    import subprocess

    try:
        return subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
    except Exception:
        return []


def manifest(root, name, models, flags):
    root = Path(root)
    path = root / "campaign.json"
    models = models.split(",")
    if path.exists():
        old = json.loads(path.read_text())
        if old["flags"] != flags or old["models"] != models:
            sys.exit(f"{path} was started with flags {old['flags']} and models {old['models']};\n"
                     f"this command has {flags} and {models}. Use another campaign name.")
        print(f"resuming campaign {old['name']} (started {old['started']})")
        return 0
    cfg = _cfg(models[0], flags)
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name, "flags": flags, "models": models,
        "arms": list(cfg.arms), "sizes": list(cfg.sizes), "seeds": list(cfg.seeds),
        "negatives": cfg.negatives, "corrections": cfg.corrections,
        "version": _version(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpus": _gpus(), "gpu_ids": os.environ.get("GPU_IDS") or None, "eval_limit": int(os.environ.get("BENCH_EVAL_LIMIT") or 0),
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path}")
    return 0


def size(root):
    print(max(json.loads((Path(root) / "campaign.json").read_text())["sizes"]))
    return 0


def summary(root):
    from .config import run_id
    from .tasks.pizza import negatives

    root = Path(root)
    m = json.loads((root / "campaign.json").read_text())
    lines, failed = [], []

    def check(ok, what):
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failed.append(what)

    lines += [f"campaign {m['name']}  version {m['version']}  started {m['started']}",
              f"flags    {' '.join(m['flags']) or '(none)'}", ""]
    for model in m["models"]:
        lines.append(f"{model}:")
        stages = root / "stages"
        for s in STAGES:
            if (stages / f"{model}.{s}.done").exists():
                continue
            check(False, f"{model} stage {s} not done")
        base = root / model / "pizza"
        lrs = json.loads((base / "lrs.json").read_text()) if (base / "lrs.json").exists() else {}
        want = [(a, n, s) for a in m["arms"] for n in m["sizes"] for s in m["seeds"]]
        done, capped, no_neg, no_fix = 0, [], [], []
        for a, n, s in want:
            if a not in lrs:
                continue
            p = base / run_id(a, n, s, lrs[a]) / "metrics.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            if not r.get("complete"):
                continue
            done += 1
            if any(r[k].get("hit_cap_rate", 0) > 0 for k in ("dev", "test")):
                capped.append(r["run_id"])
            if m["negatives"] and "negatives" not in r:
                no_neg.append(r["run_id"])
            if "corrections" not in r:
                no_fix.append(r["run_id"])
        check(done == len(want), f"{model} runs complete {done}/{len(want)}")
        check(not capped, f"{model} token cap hits {len(capped)} {capped[:3]}")
        if m["negatives"]:
            check(not no_neg, f"{model} negatives block missing in {len(no_neg)}")
        check(not no_fix, f"{model} corrections block missing in {len(no_fix)}")
        lines.append("")

    hits = negatives.collisions() + negatives.cue_collisions()
    check(not hits, f"vocabulary collisions {len(hits)}")
    exports = sorted((root / "models").glob("**/*.json"))
    check(bool(exports), f"exports {len(exports)}")
    for e in exports:
        x = json.loads(e.read_text())
        for q, f in x["files"].items():
            v = f.get("verify") or {}
            agree = v.get("agreement_with_saved")
            check(agree is not None and agree >= 0.95,
                  f"export {x['run_id']} {q} agreement {agree} em {v.get('em_gguf')}")
    lines += ["", "FAILED" if failed else "ALL CHECKS PASSED"]

    text = "\n".join(lines) + "\n"
    out = root / "bundle" / "summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    return 1 if failed else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd == "manifest":
        return manifest(rest[0], rest[1], rest[2], rest[3:])
    if cmd == "size":
        return size(rest[0])
    if cmd == "summary":
        return summary(rest[0])
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
