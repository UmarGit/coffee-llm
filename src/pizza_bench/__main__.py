"""python -m pizza_bench <stage> [options]

Stages: prepare check smoke sweep grid test grammar fewshot latency report
Every stage is resumable: runs with a complete metrics.json are skipped.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from . import data, evaluate, formats, metrics, report
from .config import (
    BASE_MODEL, EOS, PROMPT, REPO, REPORTS, RUNS, Config, caps_path, describe_env,
    is_complete, load_caps, run_dir, run_id,
)

PIZZA_URL = "https://github.com/amazon-science/pizza-semantic-parsing-dataset.git"


def _splits():
    """Evaluation splits only. Training rows are streamed per run by size and seed."""
    return {s: data.load(s) for s in ("dev", "test")}


def _cfg(args):
    cfg = Config()
    if getattr(args, "arms", None):
        cfg = cfg.with_(arms=tuple(args.arms.split(",")))
    if getattr(args, "sizes", None):
        cfg = cfg.with_(sizes=tuple(int(x) for x in args.sizes.split(",")))
    if getattr(args, "seeds", None):
        cfg = cfg.with_(seeds=tuple(int(x) for x in args.seeds.split(",")))
    if getattr(args, "gen_batch_size", None):
        cfg = cfg.with_(gen_batch_size=args.gen_batch_size)
    if getattr(args, "shard", None):
        i, n = (int(x) for x in args.shard.split("/"))
        cfg = cfg.with_(shard=(i, n))
    try:
        cfg = cfg.with_(caps=load_caps())
    except FileNotFoundError:
        pass
    lr_file = RUNS / "lrs.json"
    if lr_file.exists():
        cfg = cfg.with_(lrs=json.loads(lr_file.read_text()))
    return cfg


# ---------- stages ----------

def stage_prepare(args):
    if not (REPO / "PROTOCOL.md").exists():
        sys.exit("PROTOCOL.md is missing; it must exist before any run. Sync the repo first.")
    target = Path(formats.PIZZA_DIR)
    if not (target / "utils/catalogs").is_dir():
        print(f"cloning PIZZA into {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", PIZZA_URL, str(target)])
    splits = _splits()
    print({k: len(v) for k, v in splits.items()})

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    caps = {}
    for arm in Config().arms:
        longest = 0
        for rows in splits.values():
            texts = [formats.encode(formats.parse_exr(r["exr"]), arm) + EOS for r in rows]
            longest = max(longest, max(len(x) for x in
                                       tok(texts, add_special_tokens=False)["input_ids"]))
        caps[arm] = math.ceil(longest * 1.25)
        print(f"  {arm:11} longest target {longest:4}  cap {caps[arm]:4}")
    RUNS.mkdir(parents=True, exist_ok=True)
    caps_path().write_text(json.dumps(caps, indent=2, sort_keys=True))
    print("wrote", caps_path())
    print("PROTOCOL.md present")
    return 0


def _preflight_gpu():
    """Catch an unusable GPU toolchain here rather than minutes into training.

    Triton JIT-compiles a CUDA shim at the first kernel launch, which needs the Python
    development headers. Without them training dies partway through the first step.
    """
    problems = []
    try:
        import torch
    except ImportError:
        print("  torch not installed, skipping GPU preflight (expected off the server)")
        return problems
    print(f"  torch {torch.__version__}  cuda available {torch.cuda.is_available()}  "
          f"devices {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("  no CUDA device, skipping the Triton preflight")
        return problems

    import sysconfig
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    print(f"  python headers {header} {'present' if header.exists() else 'MISSING'}")
    if not header.exists():
        problems.append(
            f"Python development headers missing ({header}). Triton cannot compile its CUDA\n"
            f"    shim without them and training will fail at the first kernel launch. Fix with\n"
            f"    one of:\n"
            f"      sudo apt-get install -y python3.12-dev\n"
            f"      uv python install 3.12 && uv venv --python 3.12 --clear && \\\n"
            f"        uv sync --group gpu --extra pizza     # uv Pythons ship headers, no sudo")

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _noop(x_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)

        x = torch.zeros(64, device="cuda")
        _noop[(1,)](x, 64, BLOCK=64)
        torch.cuda.synchronize()
        print(f"  triton {triton.__version__} compiled and ran a kernel: ok")
    except ImportError:
        print("  triton not installed, skipping kernel compile")
    except Exception as e:
        problems.append(f"Triton could not compile a kernel: {type(e).__name__}: "
                        f"{str(e)[:200]}")
    return problems


def _stale_dirs():
    """Directories in runs/ that are not runs, so the report never trips over them."""
    known = {"logs", "tmp", "fewshot"}
    out = []
    for d in sorted(RUNS.glob("*/")):
        if d.name in known or (d / "metrics.json").exists():
            continue
        out.append(d.name)
    return out


def stage_check(args):
    cfg = _cfg(args)
    splits = _splits()
    failures = []

    failures += _preflight_gpu()
    stale = _stale_dirs()
    if stale:
        print(f"  note: {len(stale)} directories in runs/ are not runs and will be ignored "
              f"by the report: {stale[:6]}")

    sample = data.train_subset(min(50_000, data.pool_size(cfg.pool)), 0, cfg.pool)
    for name, rows in (("dev", splits["dev"]), ("test", splits["test"]),
                       ("train50k", sample)):
        for arm in cfg.arms:
            bad = 0
            for r in rows:
                node = formats.parse_exr(r["exr"])
                try:
                    if formats.to_exr(formats.decode(formats.encode(node, arm), arm)) \
                            != formats.to_exr(node):
                        bad += 1
                except Exception:
                    bad += 1
            print(f"  round-trip {name:9} {arm:11} failures {bad}")
            if bad:
                failures.append(f"round-trip {name}/{arm}: {bad}")

    for size in cfg.sizes:
        for seed in cfg.seeds:
            sub = data.train_subset(size, seed, cfg.pool)
            for split in ("dev", "test"):
                ov = data.overlap(sub, splits[split])
                print(f"  overlap n={size:<6} seed={seed} vs {split:4} "
                      f"exact {ov['exact']} normalized {ov['normalized']}")
                if ov["exact"] or ov["normalized"]:
                    failures.append(f"overlap n={size} seed={seed} {split}")
    print(f"  duplicate sources in pool: {data.duplicate_sources(splits['train'])}")

    caps = cfg.caps or load_caps()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    for arm in cfg.arms:
        longest = 0
        for rows in splits.values():
            texts = [formats.encode(formats.parse_exr(r["exr"]), arm) + EOS for r in rows]
            longest = max(longest, max(len(x) for x in
                                       tok(texts, add_special_tokens=False)["input_ids"]))
        ok = caps[arm] >= longest
        print(f"  cap {arm:11} {caps[arm]:4} >= longest {longest:4}  {'ok' if ok else 'TOO SMALL'}")
        if not ok:
            failures.append(f"cap too small for {arm}")

    eos_id = tok.convert_tokens_to_ids(EOS)
    print(f"  eos token {EOS} -> id {eos_id}")
    if eos_id is None or eos_id == tok.unk_token_id:
        failures.append("EOS token has no id")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


def _train(stage, cfg, splits, arm, size, seed, lr, use_unsloth=True, tag=""):
    from .train import train_one
    return train_one(stage, arm, size, seed, lr, splits, cfg, use_unsloth=use_unsloth, tag=tag)


def stage_smoke(args):
    cfg = _cfg(args).with_(arms=("dsl",), sizes=(200,), seeds=(0,), max_steps=100,
                           min_steps=0, eval_every=50, patience=10, gen_batch_size=16)
    splits = _splits()
    splits["dev"] = splits["dev"][:50]
    splits["test"] = splits["test"][:50]
    m = _train("smoke", cfg, splits, "dsl", 200, 0, 2e-4, use_unsloth=not args.no_unsloth)
    print(json.dumps({k: m[k] for k in ("run_id", "best_step", "examples_seen")}, indent=2))
    print("dev EM", m["dev"]["em"], "test EM", m["test"]["em"],
          "hit_cap", m["dev"].get("hit_cap_rate"))
    return stage_report(args)


def _sweep_jobs(cfg):
    return [(arm, lr, seed) for arm in cfg.arms
            for lr in cfg.lr_candidates for seed in cfg.lr_select_seeds]


def _reduce_lrs(cfg):
    """Write lrs.json once every sweep run exists. Idempotent, so re-running `sweep`
    after the shards finish does no training and just performs the selection."""
    missing, scores = [], {}
    for arm, lr, seed in _sweep_jobs(cfg):
        rid = run_id(arm, cfg.lr_select_size, seed, lr)
        if not is_complete(rid):
            missing.append(rid)
            continue
        m = json.loads((run_dir(rid) / "metrics.json").read_text())
        scores.setdefault(arm, {}).setdefault(lr, []).append(m["dev"]["em"])
    if missing:
        print(f"\n{len(missing)} sweep runs still missing, so no LR has been selected yet.")
        print("Run `python -m pizza_bench sweep` again once every shard has finished;")
        print("completed runs are skipped, so it only performs the selection.")
        for rid in missing[:8]:
            print("  missing", rid)
        return None
    chosen = {}
    for arm in cfg.arms:
        means = {lr: statistics.fmean(v) for lr, v in scores[arm].items()}
        chosen[arm] = max(means, key=means.get)
        detail = "  ".join(f"{lr:g}={means[lr]:.4f}" for lr in sorted(means))
        print(f"  {arm:11} chose {chosen[arm]:g}   ({detail})")
    (RUNS / "lrs.json").write_text(json.dumps(chosen, indent=2, sort_keys=True))
    print("wrote", RUNS / "lrs.json")
    return chosen


def stage_sweep(args):
    cfg = _cfg(args)
    splits = _splits()
    jobs = _sweep_jobs(cfg)
    i, total = cfg.shard
    mine = [j for k, j in enumerate(jobs) if k % total == i]
    todo = [j for j in mine
            if not is_complete(run_id(j[0], cfg.lr_select_size, j[2], j[1]))]
    print(f"{len(todo)} of {len(mine)} sweep runs to go in this shard "
          f"({len(jobs)} across all shards)")
    if args.dry_run:
        for j in todo:
            print("  would run", j)
        return 0
    for arm, lr, seed in todo:
        m = _train("sweep", cfg, splits, arm, cfg.lr_select_size, seed, lr,
                   use_unsloth=not args.no_unsloth)
        print(f"  {m['run_id']}: dev {m['dev']['em']:.4f} ({m['wall_time_s']:.0f}s)",
              flush=True)
    _reduce_lrs(cfg)
    return 0


def stage_grid(args):
    cfg = _cfg(args)
    if not cfg.lrs:
        sys.exit("no runs/lrs.json; run the sweep stage first")
    splits = _splits()
    jobs = [(a, n, s) for a in cfg.arms for n in cfg.sizes for s in cfg.seeds]
    i, total = cfg.shard
    jobs = [j for k, j in enumerate(jobs) if k % total == i]
    todo = [j for j in jobs if not is_complete(run_id(j[0], j[1], j[2], cfg.lrs[j[0]]))]
    print(f"{len(todo)} of {len(jobs)} runs to go in this shard")
    if args.dry_run:
        for j in todo:
            print("  would run", j)
        return 0
    for arm, size, seed in todo:
        m = _train("grid", cfg, splits, arm, size, seed, cfg.lrs[arm],
                   use_unsloth=not args.no_unsloth)
        print(f"{m['run_id']}: dev {m['dev']['em']:.4f} test {m['test']['em']:.4f} "
              f"tokens {m['test']['mean_generated_tokens']:.1f} ({m['wall_time_s']:.0f}s)",
              flush=True)
    return 0


def stage_determinism(args):
    """Rerun one already-completed configuration and report the EM difference, so
    run-to-run variance is measured rather than assumed."""
    cfg = _cfg(args)
    if not cfg.lrs:
        sys.exit("no runs/lrs.json; run the sweep stage first")
    splits = _splits()
    arm, size, seed = cfg.determinism_arm, max(cfg.sizes), cfg.seeds[0]
    lr = cfg.lrs[arm]
    first_id = run_id(arm, size, seed, lr)
    if not is_complete(first_id):
        sys.exit(f"{first_id} has not run yet; run the grid stage first")
    first = json.loads((run_dir(first_id) / "metrics.json").read_text())
    second = _train("determinism", cfg, splits, arm, size, seed, lr,
                    use_unsloth=not args.no_unsloth, tag="rep2")
    out = {"arm": arm, "size": size, "seed": seed, "lr": lr,
           "run_a": first_id, "run_b": second["run_id"]}
    for split in ("dev", "test"):
        a, b = first[split]["em"], second[split]["em"]
        out[f"{split}_em_a"] = a
        out[f"{split}_em_b"] = b
        out[f"{split}_abs_diff"] = abs(a - b)
    out["best_step_a"] = first["best_step"]
    out["best_step_b"] = second["best_step"]
    (RUNS / "determinism.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    print("\nThis is the floor on any difference between arms. Differences smaller than")
    print("test_em_abs_diff are not distinguishable from rerun noise.")
    return 0


def stage_test(args):
    """Aggregation only. Grid runs already evaluated test; nothing is retrained here."""
    runs = [r for r in report.load_runs("grid")]
    if not runs:
        sys.exit("no completed grid runs")
    print(report.render(report.summarize(runs, "test"),
                        ["arm", "size", "n_seeds", "mean", "std", "per_seed"]))
    print()
    print(report.render(report.matched_pairs(runs, "test"),
                        ["a", "b", "size", "seed", "mean_diff", "ci_lo", "ci_hi",
                         "within_noise"]))
    return 0


def fewshot_prompt(shots, arm, src):
    """Lay the shots out as real chat turns, identical to the SFT training format.

    Putting all the examples inside one user turn makes a chat-tuned base model answer
    conversationally ("Here is the breakdown of your order"), because it never sees an
    assistant turn containing the target format. Chaining complete turns fixes that and
    keeps this condition comparable with the fine-tuned runs, which use the same template.
    """
    parts = []
    for r in shots:
        target = formats.encode(formats.parse_exr(r["exr"]), arm)
        parts.append(PROMPT.format(r["src"]) + target + EOS + "\n")
    parts.append(PROMPT.format(src))
    return "".join(parts)


def stage_fewshot(args):
    cfg = _cfg(args)
    splits = _splits()
    from .train import assert_single_gpu, build
    assert_single_gpu()
    import torch
    model, tok = build(cfg, 0, use_unsloth=not args.no_unsloth)
    if next(model.parameters()).device.type != "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    shots = data.train_subset(5, 0, cfg.pool)
    out_dir = RUNS / "fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = splits["dev"][:args.limit]
    failed, summary = [], {}
    for arm in cfg.arms:
        prompts = [fewshot_prompt(shots, arm, r["src"]) for r in rows]
        texts, counts = evaluate.generate_from_prompts(model, tok, prompts, cfg.caps[arm],
                                                       cfg.gen_batch_size)
        res, preds, em = metrics.evaluate(texts, [r["exr"] for r in rows], arm,
                                          n_tokens=counts, cap=cfg.caps[arm])
        evaluate.write_predictions(out_dir / f"{arm}_dev_predictions.jsonl", rows,
                                   texts, preds, em, counts)
        summary[arm] = {k: res[k] for k in ("em", "parse_validity", "em_given_valid",
                                            "mean_generated_tokens", "hit_cap_rate")}
        print(f"  {arm:11} EM {res['em']:.4f} validity {res['parse_validity']:.4f} "
              f"tokens {res['mean_generated_tokens']:.1f}")
        if res["parse_validity"] == 0.0:
            failed.append(arm)
            for t in texts[:3]:
                print(f"      raw: {t[:110]!r}")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"complete": True, "stage": "fewshot", "n_shots": len(shots),
         "n": len(rows), "arms": summary}, indent=2, sort_keys=True))
    if failed:
        print("\nParse validity is 0 for:", failed)
        print("Raw outputs are in runs/fewshot/. Per PROTOCOL.md section 7 this is reported")
        print("as a pipeline fault, not recorded as a score of 0.0.")
        return 1
    return 0


def stage_grammar(args):
    cfg = _cfg(args)
    from . import grammar
    path = RUNS / "pizza_dsl.gbnf"
    grammar.write_gbnf(path)
    print("wrote", path)
    print("Constrained decoding runs through llama.cpp or xgrammar; see scripts/pizza/.")
    return 0


def stage_latency(args):
    cfg = _cfg(args)
    if not cfg.lrs:
        sys.exit("no runs/lrs.json")
    splits = _splits()
    from . import latency
    from .train import assert_single_gpu
    assert_single_gpu()
    jobs = [(arm, run_id(arm, max(cfg.sizes), s, cfg.lrs[arm]))
            for arm in cfg.arms for s in cfg.seeds]
    i, total = cfg.shard
    jobs = [j for k, j in enumerate(jobs) if k % total == i]
    done = latency.done_run_ids()
    todo = [j for j in jobs if j[1] not in done]
    print(f"{len(todo)} of {len(jobs)} latency runs to go in this shard "
          f"({len(done)} already measured)")
    if args.dry_run:
        for j in todo:
            print("  would measure", j[1])
        return 0
    for arm in cfg.arms:
        ids = [rid for a, rid in todo if a == arm]
        if ids:
            latency.run(cfg, splits, arm, ids, limit=args.limit,
                        use_unsloth=not args.no_unsloth)
    print("results in", latency.results_path())
    return 0


def stage_report(args):
    res = report.build(args.out, split="test")
    print(json.dumps(res, indent=2))
    return 0 if "error" not in res else 1


STAGES = {"prepare": stage_prepare, "check": stage_check, "smoke": stage_smoke,
          "sweep": stage_sweep, "grid": stage_grid, "determinism": stage_determinism,
          "test": stage_test, "grammar": stage_grammar, "fewshot": stage_fewshot,
          "latency": stage_latency, "report": stage_report}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pizza_bench")
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--arms")
    ap.add_argument("--sizes")
    ap.add_argument("--seeds")
    ap.add_argument("--shard", help="i/n, one process per GPU")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-unsloth", action="store_true")
    ap.add_argument("--gen-batch-size", type=int, default=None,
                    help="lower this if evaluation runs out of GPU memory (default 32)")
    args = ap.parse_args(argv)
    return STAGES[args.stage](args) or 0


if __name__ == "__main__":
    sys.exit(main())
