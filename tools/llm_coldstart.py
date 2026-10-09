"""Measure the LLM server's cold start and generation speed.

Usage (make sure no LLM container is running first: `modal container list`):
    python tools/llm_coldstart.py [label]

1. Sends a request to the cold endpoint (which starts a container) and polls
   /v1/models until it answers 200: the cold-start time.
2. Sends a few streaming chat requests, like the voice bot does, and reports
   time to first token and generation speed.

Uses only the standard library, so it runs without the project's venv.
"""

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://nazariusgrentina--multilingual-llm-qwenllm.us-east.modal.direct/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
POLL_SECONDS = 2
COLD_START_TIMEOUT = 15 * 60

PROMPTS = [
    "Hey, how are you?",
    "What is a good way to spend a sunny afternoon?",
    "Explain how a rainbow forms.",
    "Give me two tips for staying focused while working from home.",
    "What should I cook for dinner tonight?",
]
SYSTEM = (
    "You are a helpful voice assistant. Your replies are spoken aloud, so "
    "answer in English in one or two short sentences of plain text: no "
    "emojis, markdown, or lists."
)


def status(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def wait_until_ready():
    start = time.time()
    while time.time() - start < COLD_START_TIMEOUT:
        if status(f"{BASE_URL}/models") == 200:
            return time.time() - start
        time.sleep(POLL_SECONDS)
    raise TimeoutError("LLM did not become ready")


def stream_chat(prompt):
    body = json.dumps({
        "model": MODEL,
        "stream": True,
        "max_tokens": 200,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    first = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
            if delta:
                tokens += 1  # vLLM streams one token per chunk
                if first is None:
                    first = time.time() - start
    total = time.time() - start
    decode_rate = (tokens - 1) / (total - first) if tokens > 1 else 0.0
    return first, decode_rate, tokens


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    cold = wait_until_ready()
    print(f"[{label}] cold start (request -> /v1/models 200): {cold:.0f} s")

    stream_chat("Warm-up.")  # first request after boot can include one-off work
    ttfts, rates = [], []
    for p in PROMPTS:
        ttft, rate, n = stream_chat(p)
        ttfts.append(ttft)
        rates.append(rate)
        print(f"  ttft={ttft:.3f}s  {rate:5.1f} tok/s  ({n} tokens)  {p}")
    print(
        f"[{label}] median ttft={statistics.median(ttfts):.3f}s  "
        f"median decode={statistics.median(rates):.1f} tok/s"
    )


if __name__ == "__main__":
    main()
