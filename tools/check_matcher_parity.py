#!/usr/bin/env python3
"""Prove the TypeScript matcher agrees with the official Python one.

    uv run --extra pizza python tools/check_matcher_parity.py

The site scores orders in the browser, where the dataset's own matcher cannot
run. site/src/lib/matcher.ts is a port of it, and a port is only worth
anything if it is checked, so this feeds every saved prediction through both
and fails on a single disagreement.

Pairs come from real runs rather than hand written cases, so they include the
malformed outputs, the near misses and the sibling reorderings that are exactly
where a reimplementation drifts.
"""
import json
import subprocess
import sys
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bench.tasks.pizza import metrics

TS = REPO / "site" / "src" / "lib" / "matcher.ts"


def pairs():
    """(predicted EXR or None, gold EXR) for every saved prediction."""
    out = []
    for f in sorted(glob(str(REPO / "runs/*/pizza/*/test_predictions.jsonl"))):
        arm = Path(f).parent.name.split("-")[0]
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # `parsed` is already EXR, or null where the output did not decode.
            out.append((r["parsed"], r["gold"], arm, r["correct"]))
    return out


def python_side(rows):
    return [
        bool(p is not None and metrics.exact_match(p, g)) for p, g, _, _ in rows
    ]


def node_side(rows):
    """Run the port under node, reading pairs on stdin as JSON lines."""
    driver = f"""
import {{ isUnorderedExactMatch }} from {json.dumps(str(TS))};
let buf = "";
process.stdin.on("data", (d) => (buf += d));
process.stdin.on("end", () => {{
  const out = [];
  for (const line of buf.split("\\n")) {{
    if (!line.trim()) continue;
    const [p, g] = JSON.parse(line);
    out.push(p === null ? false : isUnorderedExactMatch(p, g));
  }}
  process.stdout.write(JSON.stringify(out));
}});
"""
    drv = REPO / "site" / ".parity-driver.mts"
    drv.write_text(driver)
    payload = "\n".join(json.dumps([p, g]) for p, g, _, _ in rows)
    try:
        res = subprocess.run(
            ["node", "--experimental-strip-types", "--no-warnings", str(drv)],
            input=payload, capture_output=True, text=True, timeout=1800,
        )
    finally:
        drv.unlink(missing_ok=True)
    if res.returncode != 0:
        sys.exit(f"node failed:\n{res.stderr[:4000]}")
    return json.loads(res.stdout)


def main():
    rows = pairs()
    if not rows:
        sys.exit("no saved predictions found; run the grid first")
    print(f"comparing {len(rows):,} prediction pairs")
    py = python_side(rows)
    ts = node_side(rows)
    if len(py) != len(ts):
        sys.exit(f"length mismatch: python {len(py)} vs node {len(ts)}")

    bad = [(i, rows[i], py[i], ts[i]) for i in range(len(py)) if py[i] != ts[i]]
    # The recorded `correct` flag is a third opinion: if the port and the
    # official matcher agree with each other but not with what was saved at run
    # time, something has changed underneath the runs.
    drift = [
        (i, rows[i]) for i in range(len(py))
        if py[i] == ts[i] and py[i] != rows[i][3]
    ]
    print(f"agree: {len(py) - len(bad):,}/{len(py):,}")
    if drift:
        print(f"WARNING {len(drift)} rows disagree with the saved `correct` flag")
        for i, r in drift[:3]:
            print(f"  {r[2]} gold={r[1][:60]} pred={str(r[0])[:60]}")
    if bad:
        print(f"\n{len(bad)} DISAGREEMENTS between python and the port:")
        for i, r, p, t in bad[:10]:
            print(f"  [{r[2]}] python={p} node={t}")
            print(f"     pred: {r[0]}")
            print(f"     gold: {r[1]}")
        sys.exit(1)
    print("parity: 100%")


if __name__ == "__main__":
    main()
