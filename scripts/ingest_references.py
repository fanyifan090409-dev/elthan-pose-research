# -*- coding: utf-8 -*-
"""重建 references 向量集合：遍历 /home/ubuntu/isef/references 分类目录，
00-总索引 不入库；元数据（类别/年份/权威）由 server 端从文档头部解析。"""
import json
import os
import urllib.error
import urllib.request

ROOT = "/home/ubuntu/isef/references"
RAG = "http://127.0.0.1:8100"


def post(path, payload, method=None):
    req = urllib.request.Request(
        RAG + path, data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"}, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


try:
    urllib.request.urlopen(urllib.request.Request(
        RAG + "/collection/references", method="DELETE"), timeout=60).read()
    print("已删除旧 references 集合，准备重建")
except urllib.error.HTTPError as e:
    print("删除跳过（可能本就不存在）:", e.code)

total = 0
for cat in sorted(os.listdir(ROOT)):
    cd = os.path.join(ROOT, cat)
    if not os.path.isdir(cd) or cat.startswith("00"):
        continue
    for fn in sorted(os.listdir(cd)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(cd, fn), encoding="utf-8") as f:
            text = f.read()
        r = post("/ingest-text", {"collection": "references",
                                  "source": fn, "text": text})
        total += r["chunks"]
        print(f"{cat}/{fn}: +{r['chunks']}块 (total={r['total']})")
print("references 入库完成，共", total, "块")
