"""python -m bench <stage> [--model qwen|lfm2] [--task pizza|coffee] [options]

Stages: prepare check smoke sweep grid test grammar fewshot latency report
Every stage is resumable: runs with a complete metrics.json are skipped, and
results live under runs/<model>/<task>/ so no two combinations collide.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from . import config as cfgmod
from . import evaluate, report
from .config import (
    REPO,
    Config,
    caps_path,
    is_complete,
    load_caps,
    load_lrs,
    lrs_path,
    model_task_dir,
    run_dir,
    run_id,
)
from .config import model as active_model
from .config import task as active_task


def _splits():
    """Evaluation splits only. Training rows are drawn per run by size and seed."""
    return active_task().load_splits()


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(active_model().hf_id)


def _target_texts(rows, arm):
    """Completions as the model will see them, for measuring the token cap."""
    t, eos = active_task(), active_model().eos
    return [t.target(r, arm) + eos for r in rows]


def _cfg(args):
    cfg = Config.for_active()
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
    lrs = load_lrs()
    if lrs:
        cfg = cfg.with_(lrs=lrs)
    return cfg


# ---------- stages ----------


def stage_prepare(args):
    if not (REPO / "PROTOCOL.md").exists():
        sys.exit(
            "PROTOCOL.md is missing; it must exist before any run. Sync the repo first."
        )
    t, m = active_task(), active_model()
    print(f"model {m.label} ({m.hf_id})")
    print(f"task  {t.label}" + (f"  [{t.licence}]" if t.licence else ""))
    if t.prepare:
        t.prepare()
    splits = _splits()
    print({k: len(v) for k, v in splits.items()})

    # The cap is a property of (model, task, arm): a different tokenizer fragments
    # the same target differently, so caps are never shared between models.
    tok = _tokenizer()
    caps = {}
    for arm in t.arms:
        longest = 0
        for rows in splits.values():
            lens = [
                len(x)
                for x in tok(_target_texts(rows, arm), add_special_tokens=False)[
                    "input_ids"
                ]
            ]
            longest = max(longest, max(lens))
        caps[arm] = math.ceil(longest * 1.25)
        print(f"  {arm:11} longest target {longest:4}  cap {caps[arm]:4}")
    model_task_dir().mkdir(parents=True, exist_ok=True)
    caps_path().write_text(json.dumps(caps, indent=2, sort_keys=True))
    print("wrote", caps_path())
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
    print(
        f"  torch {torch.__version__}  cuda available {torch.cuda.is_available()}  "
        f"devices {torch.cuda.device_count()}"
    )
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
            f"        uv sync --group gpu --extra pizza     # uv Pythons ship headers, no sudo"
        )

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
        problems.append(
            f"Triton could not compile a kernel: {type(e).__name__}: {str(e)[:200]}"
        )
    return problems


def _stale_dirs():
    """Directories in runs/ that are not runs, so the report never trips over them."""
    known = {"logs", "tmp", "fewshot"}
    out = []
    for d in sorted(model_task_dir().glob("*/")):
        if d.name in known or (d / "metrics.json").exists():
            continue
        out.append(d.name)
    return out


def stage_check(args):
    cfg = _cfg(args)
    t, m = active_task(), active_model()
    splits = _splits()
    failures = []
    print(f"model {m.label}   task {t.label}")

    failures += _preflight_gpu()
    stale = _stale_dirs()
    if stale:
        print(
            f"  note: {len(stale)} directories are not runs and will be ignored "
            f"by the report: {stale[:6]}"
        )

    # Whatever this task must prove before any GPU time is spent.
    if t.checks:
        failures += t.checks(cfg, splits)

    caps = cfg.caps or load_caps()
    tok = _tokenizer()
    for arm in cfg.arms:
        longest = 0
        for rows in splits.values():
            lens = [
                len(x)
                for x in tok(_target_texts(rows, arm), add_special_tokens=False)[
                    "input_ids"
                ]
            ]
            longest = max(longest, max(lens))
        ok = caps.get(arm, 0) >= longest
        print(
            f"  cap {arm:11} {caps.get(arm, 0):4} >= longest {longest:4}  "
            f"{'ok' if ok else 'TOO SMALL'}"
        )
        if not ok:
            failures.append(f"cap too small for {arm}")

    eos = m.eos
    eos_id = tok.convert_tokens_to_ids(eos)
    print(f"  eos token {eos} -> id {eos_id}")
    if eos_id is None or eos_id == tok.unk_token_id:
        failures.append("EOS token has no id")

    # The prompt is derived from the tokenizer for any model that does not pin one,
    # so print what it resolved to rather than trusting it.
    try:
        print(f"  prompt {evaluate.prompt_for(tok)!r}")
    except Exception as e:
        failures.append(f"prompt could not be resolved: {e}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


def _train(stage, cfg, splits, arm, size, seed, lr, use_unsloth=True, tag=""):
    from .train import train_one

    return train_one(
        stage, arm, size, seed, lr, splits, cfg, use_unsloth=use_unsloth, tag=tag
    )


def stage_smoke(args):
    cfg = _cfg(args).with_(
        arms=("dsl",),
        sizes=(200,),
        seeds=(0,),
        max_steps=100,
        min_steps=0,
        eval_every=50,
        patience=10,
        gen_batch_size=16,
    )
    splits = _splits()
    splits["dev"] = splits["dev"][:50]
    splits["test"] = splits["test"][:50]
    m = _train(
        "smoke", cfg, splits, "dsl", 200, 0, 2e-4, use_unsloth=not args.no_unsloth
    )
    print(
        json.dumps(
            {k: m[k] for k in ("run_id", "best_step", "examples_seen")}, indent=2
        )
    )
    print(
        "dev EM",
        m["dev"]["em"],
        "test EM",
        m["test"]["em"],
        "hit_cap",
        m["dev"].get("hit_cap_rate"),
    )
    return stage_report(args)


def _sweep_jobs(cfg):
    return [
        (arm, lr, seed)
        for arm in cfg.arms
        for lr in cfg.lr_candidates
        for seed in cfg.lr_select_seeds
    ]


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
        print(
            f"\n{len(missing)} sweep runs still missing, so no LR has been selected yet."
        )
        print("Run `python -m bench sweep` again once every shard has finished;")
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
    lrs_path().parent.mkdir(parents=True, exist_ok=True)
    lrs_path().write_text(json.dumps(chosen, indent=2, sort_keys=True))
    print("wrote", lrs_path())
    return chosen


def stage_sweep(args):
    cfg = _cfg(args)
    splits = _splits()
    jobs = _sweep_jobs(cfg)
    i, total = cfg.shard
    mine = [j for k, j in enumerate(jobs) if k % total == i]
    todo = [
        j for j in mine if not is_complete(run_id(j[0], cfg.lr_select_size, j[2], j[1]))
    ]
    print(
        f"{len(todo)} of {len(mine)} sweep runs to go in this shard "
        f"({len(jobs)} across all shards)"
    )
    if args.dry_run:
        for j in todo:
            print("  would run", j)
        return 0
    for arm, lr, seed in todo:
        m = _train(
            "sweep",
            cfg,
            splits,
            arm,
            cfg.lr_select_size,
            seed,
            lr,
            use_unsloth=not args.no_unsloth,
        )
        print(
            f"  {m['run_id']}: dev {m['dev']['em']:.4f} ({m['wall_time_s']:.0f}s)",
            flush=True,
        )
    _reduce_lrs(cfg)
    return 0


def stage_grid(args):
    cfg = _cfg(args)
    if not cfg.lrs:
        sys.exit(f"no {lrs_path()}; run the sweep stage first")
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
        m = _train(
            "grid",
            cfg,
            splits,
            arm,
            size,
            seed,
            cfg.lrs[arm],
            use_unsloth=not args.no_unsloth,
        )
        print(
            f"{m['run_id']}: dev {m['dev']['em']:.4f} test {m['test']['em']:.4f} "
            f"tokens {m['test']['mean_generated_tokens']:.1f} ({m['wall_time_s']:.0f}s)",
            flush=True,
        )
    return 0


def stage_determinism(args):
    """Rerun one already-completed configuration and report the EM difference, so
    run-to-run variance is measured rather than assumed."""
    cfg = _cfg(args)
    if not cfg.lrs:
        sys.exit(f"no {lrs_path()}; run the sweep stage first")
    splits = _splits()
    arm, size, seed = cfg.determinism_arm, max(cfg.sizes), cfg.seeds[0]
    lr = cfg.lrs[arm]
    first_id = run_id(arm, size, seed, lr)
    if not is_complete(first_id):
        sys.exit(f"{first_id} has not run yet; run the grid stage first")
    first = json.loads((run_dir(first_id) / "metrics.json").read_text())
    second = _train(
        "determinism",
        cfg,
        splits,
        arm,
        size,
        seed,
        lr,
        use_unsloth=not args.no_unsloth,
        tag="rep2",
    )
    out = {
        "arm": arm,
        "size": size,
        "seed": seed,
        "lr": lr,
        "run_a": first_id,
        "run_b": second["run_id"],
    }
    for split in ("dev", "test"):
        a, b = first[split]["em"], second[split]["em"]
        out[f"{split}_em_a"] = a
        out[f"{split}_em_b"] = b
        out[f"{split}_abs_diff"] = abs(a - b)
    out["best_step_a"] = first["best_step"]
    out["best_step_b"] = second["best_step"]
    (model_task_dir() / "determinism.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    print(
        "\nThis is the floor on any difference between arms. Differences smaller than"
    )
    print("test_em_abs_diff are not distinguishable from rerun noise.")
    return 0


def stage_test(args):
    """Aggregation only. Grid runs already evaluated test; nothing is retrained here."""
    runs = [r for r in report.load_runs("grid")]
    if not runs:
        sys.exit("no completed grid runs")
    print(
        report.render(
            report.summarize(runs, "test"),
            ["arm", "size", "n_seeds", "mean", "std", "per_seed"],
        )
    )
    print()
    print(
        report.render(
            report.matched_pairs(runs, "test"),
            ["a", "b", "size", "seed", "mean_diff", "ci_lo", "ci_hi", "within_noise"],
        )
    )
    return 0


def fewshot_prompt(shots, arm, src, fmt):
    """Lay the shots out as real chat turns, identical to the SFT training format.

    Putting all the examples inside one user turn makes a chat-tuned base model answer
    conversationally ("Here is the breakdown of your order"), because it never sees an
    assistant turn containing the target format. Chaining complete turns fixes that and
    keeps this condition comparable with the fine-tuned runs, which use the same template.
    """
    t, eos = active_task(), active_model().eos
    parts = []
    for r in shots:
        parts.append(fmt.format(t.source(r)) + t.target(r, arm) + eos + "\n")
    parts.append(fmt.format(src))
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
    t = active_task()
    shots = t.train_rows(5, 0)
    fmt = evaluate.prompt_for(tok)
    out_dir = model_task_dir() / "fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = splits["dev"][: args.limit]
    failed, summary = [], {}
    for arm in cfg.arms:
        prompts = [fewshot_prompt(shots, arm, t.source(r), fmt) for r in rows]
        texts, counts = evaluate.generate_from_prompts(
            model, tok, prompts, cfg.caps[arm], cfg.gen_batch_size
        )
        res, preds, em = t.score(texts, rows, arm, n_tokens=counts, cap=cfg.caps[arm])
        evaluate.write_predictions(
            out_dir / f"{arm}_dev_predictions.jsonl", rows, texts, preds, em, counts
        )
        summary[arm] = {
            k: res[k]
            for k in (
                "em",
                "parse_validity",
                "em_given_valid",
                "mean_generated_tokens",
                "hit_cap_rate",
            )
        }
        print(
            f"  {arm:11} EM {res['em']:.4f} validity {res['parse_validity']:.4f} "
            f"tokens {res['mean_generated_tokens']:.1f}"
        )
        if res["parse_validity"] == 0.0:
            failed.append(arm)
            for t in texts[:3]:
                print(f"      raw: {t[:110]!r}")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "complete": True,
                "stage": "fewshot",
                "n_shots": len(shots),
                "n": len(rows),
                "arms": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failed:
        print("\nParse validity is 0 for:", failed)
        print(
            "Raw outputs are in runs/fewshot/. Per PROTOCOL.md section 7 this is reported"
        )
        print("as a pipeline fault, not recorded as a score of 0.0.")
        return 1
    return 0


def stage_grammar(args):
    cfg = _cfg(args)
    from . import grammar

    path = model_task_dir() / f"{active_task().name}_dsl.gbnf"
    t = active_task()
    if not t.write_grammar:
        sys.exit(f"task {t.name} has no grammar")
    t.write_grammar(path)
    print("wrote", path)
    print(
        "Constrained decoding runs through llama.cpp or xgrammar; see scripts/pizza/."
    )
    return 0


def stage_latency(args):
    cfg = _cfg(args)
    if not cfg.lrs:
        sys.exit(f"no {lrs_path()}")
    splits = _splits()
    from . import latency
    from .train import assert_single_gpu

    assert_single_gpu()
    jobs = [
        (arm, run_id(arm, max(cfg.sizes), s, cfg.lrs[arm]))
        for arm in cfg.arms
        for s in cfg.seeds
    ]
    i, total = cfg.shard
    jobs = [j for k, j in enumerate(jobs) if k % total == i]
    done = latency.done_run_ids()
    todo = [j for j in jobs if j[1] not in done]
    print(
        f"{len(todo)} of {len(jobs)} latency runs to go in this shard "
        f"({len(done)} already measured)"
    )
    if args.dry_run:
        for j in todo:
            print("  would measure", j[1])
        return 0
    for arm in cfg.arms:
        ids = [rid for a, rid in todo if a == arm]
        if ids:
            latency.run(
                cfg, splits, arm, ids, limit=args.limit, use_unsloth=not args.no_unsloth
            )
    print("results in", latency.results_path())
    return 0


def stage_report(args):
    res = report.build(args.out, split="test")
    print(json.dumps(res, indent=2))
    return 0 if "error" not in res else 1


STAGES = {
    "prepare": stage_prepare,
    "check": stage_check,
    "smoke": stage_smoke,
    "sweep": stage_sweep,
    "grid": stage_grid,
    "determinism": stage_determinism,
    "test": stage_test,
    "grammar": stage_grammar,
    "fewshot": stage_fewshot,
    "latency": stage_latency,
    "report": stage_report,
}


def main(argv=None):
    from . import models as modelmod
    from . import tasks as taskmod

    ap = argparse.ArgumentParser(prog="bench")
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument(
        "--model",
        default=cfgmod.DEFAULT_MODEL,
        choices=sorted(modelmod.MODELS),
        help="which base model to run (default: %(default)s)",
    )
    ap.add_argument(
        "--task",
        default=cfgmod.DEFAULT_TASK,
        choices=sorted(taskmod.NAMES),
        help="which task to run (default: %(default)s)",
    )
    ap.add_argument("--arms")
    ap.add_argument("--sizes")
    ap.add_argument("--seeds")
    ap.add_argument("--shard", help="i/n, one process per GPU")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-unsloth", action="store_true")
    ap.add_argument(
        "--gen-batch-size",
        type=int,
        default=None,
        help="lower this if evaluation runs out of GPU memory (default 32)",
    )
    args = ap.parse_args(argv)
    # One selection point for the whole process. Everything downstream reads it
    # back through config.model() and config.task().
    cfgmod.select(args.model, args.task)
    return STAGES[args.stage](args) or 0


if __name__ == "__main__":
    sys.exit(main())
