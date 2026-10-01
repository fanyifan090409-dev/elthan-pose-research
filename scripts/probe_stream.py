import json
import time

import requests

BASE_TEXT = (
    "用户正在做深蹲训练，请以健身教练的口吻给出即时语音纠错建议。"
    "注意膝盖是否内扣、重心是否后移、背部是否挺直。"
    "反馈要简短有力，先说问题再说纠正动作，最后给一句鼓励。"
)


def make_prompt(chars):
    return (BASE_TEXT * (chars // len(BASE_TEXT) + 1))[:chars]


def probe(url, prompt, max_tokens, label):
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    r = requests.post(url.rstrip("/") + "/chat/completions", json=payload, timeout=600)
    n_chunks = 0
    first_content = None
    first_reasoning = None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            print(f"[{label}] [DONE] at {time.time() - t0:.3f}s, chunks={n_chunks}")
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        n_chunks += 1
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {}) or {}
            rc = delta.get("reasoning_content")
            c = delta.get("content")
            if first_reasoning is None and rc:
                first_reasoning = time.time() - t0
            if first_content is None and c:
                first_content = time.time() - t0
            if n_chunks <= 3 or (rc and len(rc) < 200):
                print(
                    f"[{label}] t={time.time() - t0:.3f}s "
                    f"reasoning={len(rc) if rc else 0} content={len(c) if c else 0} "
                    f"finish={chunk['choices'][0].get('finish_reason')}"
                )
        if chunk.get("usage"):
            print(f"[{label}] usage at {time.time() - t0:.3f}s: {chunk['usage']}")
    print(
        f"[{label}] first_reasoning={first_reasoning} first_content={first_content}"
    )


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1"
    probe(url, make_prompt(1500), 256, "A-1500chars")
    probe(url, make_prompt(1500) + "/no_think", 256, "A-no_think")
    probe(url, make_prompt(6000), 64, "D-6000chars")
    probe(url, "数三个数", 16, "short-16")
