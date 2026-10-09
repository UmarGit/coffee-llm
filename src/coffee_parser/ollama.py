"""Ollama client.

The model is called in RAW mode with the exact prompt it was trained on: Ollama's
own template handling is bypassed, because in non-raw mode the model returns empty
output. Do not change PROMPT or STOPS without retraining.
"""

import json
import time
import urllib.request

PROMPT = (
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
STOPS = ["<|im_end|>", "<|endoftext|>", "</think>"]

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "coffee"


def clean(out):
    """Trim anything the server returned past a stop string."""
    for s in STOPS:
        out = out.split(s)[0]
    return out.strip()


def generate(text, model=DEFAULT_MODEL, host=DEFAULT_HOST, num_predict=64):
    """Parse one order. Returns (output, elapsed_ms, output_token_count)."""
    body = json.dumps(
        {
            "model": model,
            "prompt": PROMPT.format(text),
            "raw": True,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "num_predict": num_predict, "stop": STOPS},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        resp = json.loads(r.read())
    return (
        clean(resp["response"]),
        (time.perf_counter() - t0) * 1000,
        resp.get("eval_count", 0),
    )
