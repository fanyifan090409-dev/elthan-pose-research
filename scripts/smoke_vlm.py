import base64, json, sys, time, requests

PORT = sys.argv[3] if len(sys.argv) > 3 else "8000"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl-8b"
IMG = sys.argv[2] if len(sys.argv) > 2 else ""

SYSTEM = ("你是一名专业健身姿态矫正教练。只根据画面中可见的内容进行分析，"
          "输出三段：【问题】【风险】【矫正动作】。"
          "画面信息不足时如实说明，严禁虚构看不见的姿态。")

content = [{"type": "text", "text": "请分析图中动作的姿态并给出矫正反馈。"}]
if IMG:
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    mime = "image/png" if IMG.endswith(".png") else "image/jpeg"
    content.insert(0, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

payload = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": content}], "max_tokens": 300, "temperature": 0.2, "stream": True}

t0 = time.time(); ttft = None; parts = []
r = requests.post(URL, json=payload, stream=True, timeout=180)
r.raise_for_status()
for line in r.iter_lines():
    if not line:
        continue
    s = line.decode()
    if not s.startswith("data: "):
        continue
    d = s[6:]
    if d.strip() == "[DONE]":
        break
    try:
        delta = json.loads(d)["choices"][0]["delta"].get("content")
    except Exception:
        continue
    if delta:
        if ttft is None:
            ttft = time.time() - t0
        parts.append(delta)
total = time.time() - t0
text = "".join(parts)
print(f"TTFT={ttft:.3f}s TOTAL={total:.2f}s CHARS={len(text)}")
print(text)
