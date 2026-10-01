import sys
import time

sys.path.insert(0, "/home/ubuntu/isef/scripts")
import requests

import bench_speed as bs

BASE = bs.BASE_TEXT


def make_prompt(chars):
    return (BASE * (chars // len(BASE) + 1))[:chars]


def miniprobe(label, url):
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [{"role": "user", "content": make_prompt(1500)}],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    r = requests.post(url + "/chat/completions", json=payload, stream=True, timeout=600)
    n = 0
    first = None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        n += 1
        if first is None:
            first = time.time()
    print(
        "%s first=%.3f n=%d end=%.3f"
        % (label, (first - t0) if first else -1, n, time.time() - t0)
    )


class Args:
    url = "http://localhost:8001/v1"
    model = "qwen3-vl-8b"
    rounds = 1
    warmup = 0


for i in range(3):
    miniprobe("mini-sgl-%d" % i, "http://localhost:8001/v1")
    out = bs.group_a(Args())
    print("bench-sgl-%d" % i, out)
for i in range(2):
    miniprobe("mini-vllm-%d" % i, "http://localhost:8000/v1")
    a = Args()
    a.url = "http://localhost:8000/v1"
    print("bench-vllm-%d" % i, bs.group_a(a))
