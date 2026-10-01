import argparse
import json

import requests

COACH = [
    "用户深蹲时膝盖明显内扣，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户弓步时前膝超过脚尖，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户俯卧撑时腰部塌陷，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户硬拉时背部弓起，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户跑步时摆臂幅度过大，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户平板支撑时髋部下沉，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户仰卧起坐时颈部代偿发力，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
    "用户波比跳落地没有缓冲，请用健身教练口吻给出即时语音纠错提示，不超过50字。",
]

JSON_PROMPTS = [
    ("训练数据：动作=深蹲，次数=12，膝盖内扣=有，重心后移=无。输出JSON，字段：exercise、reps、issues、advice。只输出JSON。",
     ["exercise", "reps", "issues", "advice"]),
    ("训练数据：动作=硬拉，重量=60kg，次数=8，组数=3。输出JSON，字段：exercise、weight、reps、sets、notes。只输出JSON。",
     ["exercise", "weight", "reps", "sets", "notes"]),
    ("训练数据：动作=俯卧撑，次数=15，塌腰=有，组数=2。输出JSON，字段：exercise、reps、issues、sets。只输出JSON。",
     ["exercise", "reps", "issues", "sets"]),
    ("训练数据：动作=平板支撑，时长=60秒，髋部下沉=无。输出JSON，字段：exercise、duration、issues。只输出JSON。",
     ["exercise", "duration", "issues"]),
    ("训练数据：动作=跑步，时长=30分钟，配速=5分30秒。输出JSON，字段：exercise、duration、pace、issues。只输出JSON。",
     ["exercise", "duration", "pace", "issues"]),
    ("训练数据：动作=弓步，次数=10，膝盖过脚尖=有，组数=3。输出JSON，字段：exercise、reps、issues、sets。只输出JSON。",
     ["exercise", "reps", "issues", "sets"]),
    ("训练数据：动作=仰卧起坐，次数=20，颈部代偿=有。输出JSON，字段：exercise、reps、issues、advice。只输出JSON。",
     ["exercise", "reps", "issues", "advice"]),
    ("训练数据：动作=波比跳，次数=10，落地缓冲=无。输出JSON，字段：exercise、reps、issues、advice。只输出JSON。",
     ["exercise", "reps", "issues", "advice"]),
]

CODE = [
    "用Python写一个函数fib(n)，返回第n个斐波那契数。只输出代码。",
    "用Python写一个函数is_prime(n)，判断n是否为素数。只输出代码。",
    "用Python写一个函数binary_search(arr, x)，返回x在有序数组arr中的下标，不存在返回-1。只输出代码。",
    "用Python写一个函数merge_sorted(a, b)，合并两个升序列表。只输出代码。",
    "用Python写一个函数reverse_string(s)，反转字符串。只输出代码。",
    "用Python写一个函数fizzbuzz(n)，返回1到n的FizzBuzz列表。只输出代码。",
    "用Python写一个函数is_balanced(s)，判断字符串括号是否匹配。只输出代码。",
    "用Python写一个函数dedup(lst)，去除列表重复元素并保持原顺序。只输出代码。",
]


def extract_code(text):
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("python"):
                part = part[3:].strip()
            return part
    return text.strip()


def check_coach(text):
    n = len(text)
    return {"length": n, "ok": 10 <= n <= 150}


def check_json(text, required):
    t = text.strip()
    if t.startswith("```"):
        t = extract_code(t)
    try:
        obj = json.loads(t)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    missing = [k for k in required if k not in obj]
    return {"ok": not missing, "missing_keys": missing, "keys": sorted(obj.keys())}


def check_code(text):
    code = extract_code(text)
    try:
        compile(code, "<gen>", "exec")
        return {"ok": True, "code": code}
    except SyntaxError as e:
        return {"ok": False, "error": str(e)[:200], "code": code}


def run(args):
    results = []
    for i, p in enumerate(COACH):
        r = chat(args.url, args.model, p)
        results.append({
            "group": "coach", "idx": i, "prompt": p,
            "output": r["out"], "tokens": r["tokens"],
            "check": check_coach(r["out"]),
        })
    for i, (p, keys) in enumerate(JSON_PROMPTS):
        r = chat(args.url, args.model, p)
        results.append({
            "group": "json", "idx": i, "prompt": p,
            "output": r["out"], "tokens": r["tokens"],
            "check": check_json(r["out"], keys),
        })
    for i, p in enumerate(CODE):
        r = chat(args.url, args.model, p)
        results.append({
            "group": "code", "idx": i, "prompt": p,
            "output": r["out"], "tokens": r["tokens"],
            "check": check_code(r["out"]),
        })
    return results


def chat(url, model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0,
    }
    r = requests.post(url.rstrip("/") + "/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    j = r.json()
    return {
        "out": j["choices"][0]["message"]["content"],
        "tokens": j.get("usage", {}),
    }


def compare(path_a, path_b):
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    same = 0
    diff = []
    for x, y in zip(a, b):
        if x["output"].strip() == y["output"].strip():
            same += 1
        else:
            diff.append({"group": x["group"], "idx": x["idx"], "prompt": x["prompt"]})
    by_group = {}
    for d in diff:
        by_group.setdefault(d["group"], 0)
        by_group[d["group"]] += 1
    print(json.dumps({
        "total": len(a),
        "exact_match": same,
        "diverged": len(diff),
        "diverged_by_group": by_group,
        "diverged_list": diff,
    }, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
