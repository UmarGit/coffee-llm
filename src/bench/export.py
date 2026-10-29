"""Merge one run's adapter into its base model and export it as GGUF for llama.cpp.

    python -m bench export --model qwen --task pizza --arm words
    python -m bench export --model qwen --task pizza --arm words --fresh

Without --fresh the existing run is exported, so the file reproduces the outputs the
paper and the site report. With --fresh a new run is trained first, under its own
tag, so the paper's run is never overwritten.

The GGUF is then checked, not trusted: llama-server runs it greedily on the first
--verify test orders, and the result is compared against that run's saved
predictions. A quantization or template bug shows up here as a drop in agreement
rather than as a quietly worse model on the website.

PIZZA is CC-BY-NC-4.0. Everything this writes goes under models/, which is
gitignored, and the manifest records the licence so the file never travels without it.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

from .config import REPO, is_complete, run_dir, run_id
from .config import model as active_model
from .config import task as active_task

LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"
QUANTS = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M", "f16": None}


def _sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], **kw)


def llama_cpp(path=None):
    """A llama.cpp checkout with llama-quantize and llama-server built.

    Cloned and built on first use, with CUDA when nvcc is present, so the
    verification pass runs on the GPU too."""
    root = Path(path or os.environ.get("LLAMA_CPP") or REPO.parent / "llama.cpp")
    if not (root / "convert_hf_to_gguf.py").exists():
        print(f"llama.cpp not found at {root}, cloning", flush=True)
        try:
            # A server without internet access hangs here rather than failing, so
            # bound it and say what to do instead.
            subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_URL, str(root)],
                           check=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            shutil.rmtree(root, ignore_errors=True)
            raise RuntimeError(
                f"could not clone llama.cpp into {root} ({type(e).__name__}). This machine "
                "probably cannot reach GitHub. From the Mac, send a copy instead:\n\n"
                "    ./scripts/push_llama_cpp.sh\n\n"
                "then rerun the export (it is built here on first use)."
            ) from None
    bins = root / "build" / "bin"
    if not ((bins / "llama-quantize").exists() and (bins / "llama-server").exists()):
        cuda = "ON" if shutil.which("nvcc") else "OFF"
        _sh(["cmake", "-S", root, "-B", root / "build", f"-DGGML_CUDA={cuda}",
             "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"])
        _sh(["cmake", "--build", root / "build", "-j", str(os.cpu_count() or 8),
             "--target", "llama-quantize", "llama-server"])
    return root


def merge(adapter_dir, out_dir, use_unsloth=True):
    """Adapter + base -> one 16-bit Hugging Face checkpoint in out_dir."""
    from .latency import load_adapter

    model, tok = load_adapter(adapter_dir, use_unsloth=use_unsloth)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(str(out_dir), tok, save_method="merged_16bit")
    else:
        model.merge_and_unload().save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def convert(root, merged_dir, out_dir, name, quants):
    """merged checkpoint -> f16 GGUF -> each requested quantization."""
    f16 = out_dir / f"{name}-f16.gguf"
    env = {**os.environ, "PYTHONPATH": f"{root / 'gguf-py'}:{os.environ.get('PYTHONPATH', '')}"}
    _sh([sys.executable, root / "convert_hf_to_gguf.py", merged_dir,
         "--outtype", "f16", "--outfile", f16], env=env)
    files = {}
    for q in quants:
        if QUANTS[q] is None:
            files[q] = f16
            continue
        dst = out_dir / f"{name}-{q}.gguf"
        _sh([root / "build" / "bin" / "llama-quantize", f16, dst, QUANTS[q]])
        files[q] = dst
    if "f16" not in quants:
        f16.unlink()
    return files


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def verify(root, gguf, rid, arm, cap, n, fmt):
    """Greedy llama-server outputs vs the run's saved test predictions."""
    saved = [json.loads(line) for line in (run_dir(rid) / "test_predictions.jsonl").open()][:n]
    port = 8090
    srv = subprocess.Popen(
        [str(root / "build" / "bin" / "llama-server"), "-m", str(gguf), "--port", str(port),
         "-c", "1024", "-ngl", "99", "--no-webui"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not come up")

        outputs, ms = [], []
        for row in saved:
            t0 = time.perf_counter()
            res = _post(f"http://127.0.0.1:{port}/completion", {
                "prompt": fmt.format(row["src"]), "n_predict": cap, "temperature": 0,
                "top_k": 1, "stop": [active_model().eos], "cache_prompt": False,
            })
            ms.append((time.perf_counter() - t0) * 1000)
            outputs.append(res["content"].split(active_model().eos)[0].strip())
    finally:
        srv.terminate()
        srv.wait(timeout=30)

    t = active_task()
    metrics, _, correct = t.score(outputs, t.rows_from_predictions(saved), arm)
    agree = sum(o == s["raw"] for o, s in zip(outputs, saved)) / len(saved)
    saved_em = sum(bool(s["correct"]) for s in saved) / len(saved)
    diffs = [
        {"src": s["src"], "saved": s["raw"], "gguf": o}
        for o, s in zip(outputs, saved) if o != s["raw"]
    ][:10]
    ms.sort()
    return {
        "n": len(saved),
        "agreement_with_saved": round(agree, 4),
        "em_gguf": round(metrics["em"], 4),
        "em_saved_same_rows": round(saved_em, 4),
        "parse_validity_gguf": round(metrics["parse_validity"], 4),
        "median_ms_server": round(ms[len(ms) // 2], 1),
        "first_disagreements": diffs,
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(cfg, splits, arm, size, seed, lr, fresh=False, quants=("q8_0",), n_verify=200,
           llama_cpp_path=None, use_unsloth=True, tag=""):
    # --fresh trains under a dated tag; --tag names an already trained run, so an
    # export that failed after training can be rerun without training again.
    tag = tag or (f"fresh-{date.today():%Y%m%d}" if fresh else "")
    rid = run_id(arm, size, seed, lr, tag)
    if fresh or not is_complete(rid):
        from .train import train_one

        print(f"training {rid}", flush=True)
        train_one("export", arm, size, seed, lr, splits, cfg, use_unsloth=use_unsloth, tag=tag)
    adapter = run_dir(rid) / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"{adapter} has no adapter; was the run saved with one?")

    root = llama_cpp(llama_cpp_path)
    m, t = active_model(), active_task()
    # The exact prompt evaluation used, resolved the same way, so the GGUF is
    # verified under the contract the saved predictions were made with.
    from transformers import AutoTokenizer

    from .evaluate import prompt_for

    fmt = prompt_for(AutoTokenizer.from_pretrained(m.hf_id))
    out = REPO / "models" / t.name
    out.mkdir(parents=True, exist_ok=True)
    name = f"{t.name}-{m.name}-{rid}"
    merged = out / f"{name}-merged"

    print(f"merging {adapter}", flush=True)
    merge(adapter, merged, use_unsloth)
    files = convert(root, merged, out, name, quants)
    shutil.rmtree(merged)

    manifest = {
        "run_id": rid,
        "model": m.name,
        "base": m.hf_id,
        "task": t.name,
        "arm": arm,
        "prompt": fmt.replace("{{", "{").replace("}}", "}"),
        "stop": m.eos,
        "cap": cfg.caps[arm],
        "licence": getattr(t, "licence", None),
        "files": {},
    }
    for q, path in files.items():
        print(f"verifying {path.name} on {n_verify} test orders", flush=True)
        check = verify(root, path, rid, arm, cfg.caps[arm], n_verify, fmt) if n_verify else None
        manifest["files"][q] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "verify": check,
        }
        if check:
            print(f"  agreement {check['agreement_with_saved']:.1%}  "
                  f"em {check['em_gguf']:.3f} (saved {check['em_saved_same_rows']:.3f})  "
                  f"median {check['median_ms_server']} ms", flush=True)
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}/{name}.json")
    return manifest
