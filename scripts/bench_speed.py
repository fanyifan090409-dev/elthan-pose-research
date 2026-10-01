import argparse
import concurrent.futures
import json
import time

import requests

BASE_TEXT = (
    "用户正在做深蹲训练，请以健身教练的口吻给出即时语音纠错建议。"
    "注意膝盖是否内扣、重心是否后移、背部是否挺直。"
    "反馈要简短有力，先说问题再说纠正动作，最后给一句鼓励。"
)

QUESTIONS = [
    "第一轮：膝盖内扣怎么纠正？",
    "第二轮：重心靠前了，给个提示。",
    "第三轮：这组做完了，报个数。",
    "第四轮：弓背了，纠正一下。",
    "第五轮：给一句鼓励的话。",
]


def make_prompt(chars):
    return (BASE_TEXT * (chars // len(BASE_TEXT) + 1))[:chars]


def chat(url, model, prompt, max_tokens, temperature, stream=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    r = requests.post(url.rstrip("/") + "/chat/completions", json=payload, timeout=600, stream=stream)
    r.raise_for_status()
    return r


def stream_collect(r, t0):
    first_data = None
    first_content = None
    usage = {}
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if first_data is None:
            first_data = time.time()
        if first_content is None and chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {}) or {}
            if delta.get("content") or delta.get("reasoning_content"):
                first_content = time.time()
        if chunk.get("usage"):
            usage = chunk["usage"]
    return first_data, first_content, usage


def p5095(xs):
    xs = sorted(xs)
    def pct(p):
        k = (len(xs) - 1) * p
        lo = min(int(k), len(xs) - 1)
        hi = min(int(k) + 1, len(xs) - 1)
        return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)
    return pct(0.5), pct(0.95)


def warmup(args):
    for i in range(args.warmup):
        chat(args.url, args.model, make_prompt(1500), 32, 0.7, stream=True).close()
        time.sleep(1)


def group_a(args):
    out = []
    prompt = make_prompt(1500)
    for i in range(args.rounds):
        t0 = time.time()
        r = chat(args.url, args.model, prompt, 256, 0.7, stream=True)
        first_data, first_content, usage = stream_collect(r, t0)
        t1 = time.time()
        n_out = usage.get("completion_tokens") or 0
        n_in = usage.get("prompt_tokens") or 0
        out.append({
            "round": i,
            "total_s": round(t1 - t0, 3),
            "ttft_data_s": round(first_data - t0, 3) if first_data else None,
            "ttft_content_s": round(first_content - t0, 3) if first_content else None,
            "decode_tok_s": round(n_out / (t1 - first_data), 2) if first_data and t1 > first_data else None,
            "prompt_tokens": n_in,
            "completion_tokens": n_out,
        })
    return out


def group_b(args):
    out = []
    prompt = make_prompt(1500)
    for c in args.concurrency:
        for i in range(args.rounds):
            def one(_):
                t0 = time.time()
                r = chat(args.url, args.model, prompt, 256, 0.7)
                j = r.json()
                u = j.get("usage", {})
                return time.time() - t0, u.get("completion_tokens") or 0, u.get("prompt_tokens") or 0

            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
                results = list(ex.map(one, range(c)))
            wall = time.time() - t0
            n_out = sum(x[1] for x in results)
            n_in = sum(x[2] for x in results)
            p50, p95 = p5095([x[0] for x in results])
            out.append({
                "concurrency": c,
                "round": i,
                "wall_s": round(wall, 3),
                "agg_tok_s": round(n_out / wall, 2) if wall else None,
                "latency_p50_s": p50,
                "latency_p95_s": p95,
                "total_prompt_tokens": n_in,
                "total_completion_tokens": n_out,
            })
    return out


def group_c(args):
    out = []
    for chars in args.input_sizes:
        prompt = make_prompt(chars)
        ttft_data = []
        ttft_content = []
        n_in = 0
        for i in range(5):
            t0 = time.time()
            r = chat(args.url, args.model, prompt, 1, 0.7, stream=True)
            fd, fc, usage = stream_collect(r, t0)
            if fd:
                ttft_data.append(fd - t0)
            if fc:
                ttft_content.append(fc - t0)
            n_in = usage.get("prompt_tokens") or n_in
        p50, p95 = p5095(ttft_data) if ttft_data else (None, None)
        c50, c95 = p5095(ttft_content) if ttft_content else (None, None)
        out.append({
            "input_chars": chars,
            "prompt_tokens": n_in,
            "ttft_data_p50_s": p50,
            "ttft_data_p95_s": p95,
            "ttft_content_p50_s": c50,
            "ttft_content_p95_s": c95,
            "samples": [round(t, 3) for t in ttft_data],
        })
    return out


def group_d(args):
    out = []
    prefix = make_prompt(6000)
    for rnd in range(args.rounds):
        turns = []
        for q in QUESTIONS:
            t0 = time.time()
            r = chat(args.url, args.model, prefix + "\n" + q, 256, 0.7, stream=True)
            first_data, first_content, usage = stream_collect(r, t0)
            t1 = time.time()
            n_out = usage.get("completion_tokens") or 0
            turns.append({
                "turn": q[:2],
                "ttft_data_s": round(first_data - t0, 3) if first_data else None,
                "ttft_content_s": round(first_content - t0, 3) if first_content else None,
                "total_s": round(t1 - t0, 3),
                "decode_tok_s": round(n_out / (t1 - first_data), 2) if first_data and t1 > first_data else None,
                "completion_tokens": n_out,
                "prompt_tokens": usage.get("prompt_tokens") or 0,
            })
        out.append({"round": rnd, "turns": turns})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--group", default="all", choices=["a", "b", "c", "d", "all"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--input-sizes", type=int, nargs="+", default=[6000, 24000, 48000])
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args()

    warmup(args)
    result = {"url": args.url, "model": args.model}
    if args.group in ("a", "all"):
        result["A_单流解码"] = group_a(args)
    if args.group in ("b", "all"):
        result["B_并发吞吐"] = group_b(args)
    if args.group in ("c", "all"):
        result["C_TTFT"] = group_c(args)
    if args.group in ("d", "all"):
        result["D_多轮共享前缀"] = group_d(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
