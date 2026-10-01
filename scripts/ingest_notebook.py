# -*- coding: utf-8 -*-
"""重建 notebook 向量集合：全量摄入 /home/ubuntu/isef/lab-notebook/*.md
（实验记录本：全部实验数据、方案与结论）。幂等：先删旧集合再重建。
由 @reboot 后的 boot_all 与每 30 分钟的 cron 触发，flock 防并发。

只有笔记内容真的变了才重建：每 30 分钟无条件 DELETE+rebuild 会在磁盘上留下
孤儿 HNSW 段（曾累积约 350 个 / 154M），并且一旦进程在重建中途被杀，
就会留下缺 hnsw.bin 的半写段，查询直接 500。
"""
import glob
import hashlib
import json
import os
import urllib.error
import urllib.request

ROOT = "/home/ubuntu/isef/lab-notebook"
RAG = "http://127.0.0.1:8100"
STAMP = "/home/ubuntu/isef/rag/.notebook-ingest-hash"


def post(path, payload, method=None):
    req = urllib.request.Request(
        RAG + path, data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"}, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def current_hash() -> str:
    h = hashlib.sha256()
    for fn in sorted(glob.glob(os.path.join(ROOT, "*.md"))):
        h.update(os.path.basename(fn).encode())
        h.update(str(os.path.getmtime(fn)).encode())
        with open(fn, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


digest = current_hash()
unchanged = False
if os.path.exists(STAMP):
    try:
        with open(STAMP, encoding="utf-8") as f:
            unchanged = f.read().strip() == digest
    except Exception as e:
        print("读取指纹失败，按有变化处理:", e)
if unchanged:
    print("笔记无变化，跳过重建")
    raise SystemExit(0)

try:
    urllib.request.urlopen(urllib.request.Request(
        RAG + "/collection/notebook", method="DELETE"), timeout=60).read()
    print("已删除旧 notebook 集合，准备重建")
except urllib.error.HTTPError as e:
    print("删除跳过（可能本就不存在）:", e.code)

total = 0
for fn in sorted(glob.glob(os.path.join(ROOT, "*.md"))):
    with open(fn, encoding="utf-8") as f:
        text = f.read()
    r = post("/ingest-text", {"collection": "notebook",
                              "source": os.path.basename(fn), "text": text})
    total += r["chunks"]
    print(f"{os.path.basename(fn)}: +{r['chunks']}块 (total={r['total']})")

# 全部入库成功才落指纹，中途失败下次仍会重建
with open(STAMP, "w", encoding="utf-8") as f:
    f.write(digest)
print("notebook 入库完成，共", total, "块")
