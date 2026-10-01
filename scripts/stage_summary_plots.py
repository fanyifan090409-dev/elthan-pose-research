# -*- coding: utf-8 -*-
# stage_summary_plots.py — 阶段总结图：谱系曲线(损失/学习率/mAP) + 数据组成 + 消融柱图
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

RUNS = "/home/ubuntu/isef/runs"

# 谱系 run 目录（按训练时间顺序）
LINEAGE = [
    ("官方预训练", None),
    ("round3", f"{RUNS}/pose-round3"),
    ("round4", f"{RUNS}/pose-round4"),
    ("round5(bad-lr)", f"{RUNS}/pose-round5"),
    ("round5-2", f"{RUNS}/pose-round5-2"),
    ("round5-3", f"{RUNS}/pose-round5-3"),
    ("去CrowdPose", f"{RUNS}/pose-round5-abl-nocrowd-2"),
    ("去OCHuman", f"{RUNS}/pose-round5-abl-noochu"),
    ("去OTP", f"{RUNS}/pose-round5-abl-nootp"),
]

# 三基准 Pose mAP50-95（官方预训练/round3/round4/round5-2/round5-3 已实测；消融三组跑完后更新）
BENCH = {
    "官方预训练": {"COCO": 0.390, "CrowdPose": 0.474, "OCHuman": 0.272},
    "round3":      {"COCO": 0.502, "CrowdPose": 0.465, "OCHuman": 0.272},
    "round4":      {"COCO": 0.510, "CrowdPose": 0.543, "OCHuman": 0.331},
    "round5-2":    {"COCO": 0.507, "CrowdPose": 0.532, "OCHuman": 0.395},
    "round5-3":    {"COCO": 0.505, "CrowdPose": 0.536, "OCHuman": 0.399},
    "去CrowdPose": {"COCO": 0.50546, "CrowdPose": 0.46115, "OCHuman": 0.37712},
    "去OCHuman":   {"COCO": 0.50234, "CrowdPose": 0.53266, "OCHuman": 0.33120},
    "去OTP":       {"COCO": 0.50533, "CrowdPose": 0.53038, "OCHuman": 0.39349},
}

# 训练数据组成（图数）
DATA_COMP = {
    "round3":   {"COCO": 53044, "MPII": 14679, "CrowdPose": 0, "OCHuman": 0, "OTP": 0},
    "round4":   {"COCO": 53044, "MPII": 14679, "CrowdPose": 11555, "OCHuman": 0, "OTP": 0},
    "round5":   {"COCO": 53044, "MPII": 14679, "CrowdPose": 11555, "OCHuman": 2837, "OTP": 4300},
}

# 中文字体探测（Ubuntu 训练机）
def setup_font():
    for name in ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Zen Hei", "AR PL UMing CN"]:
        for f in fm.fontManager.ttflist:
            if f.name == name:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

def load_results(path):
    """读 results.csv，按表头动态取列；返回 dict of list"""
    fp = os.path.join(path, "results.csv")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    out = {}
    for col in reader.fieldnames:
        col = col.strip().lstrip("\ufeff")
        try:
            out[col] = [float(r[col]) for r in rows]
        except (ValueError, KeyError):
            pass
    return out

def col(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None

setup_font()
runs_data = {label: load_results(p) for label, p in LINEAGE if p}

# ---------------- 图1：谱系曲线 2×2 ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
colors = {"round3": "#1f77b4", "round4": "#ff7f0e", "round5(bad-lr)": "#d62728",
          "round5-2": "#2ca02c", "round5-3": "#9467bd",
          "去CrowdPose": "#8c564b", "去OCHuman": "#e377c2", "去OTP": "#7f7f7f"}
styles = {"round5(bad-lr)": "--"}

# (a) COCO val Pose mAP50-95
ax = axes[0, 0]
for label, d in runs_data.items():
    if d is None:
        continue
    m = col(d, "metrics/mAP50-95(P)")
    if m:
        ax.plot(range(1, len(m) + 1), m, label=label, color=colors[label],
                linestyle=styles.get(label, "-"), linewidth=1.6)
ax.axhline(0.390, color="gray", linestyle=":", linewidth=1, label="官方预训练 0.390")
ax.axhline(0.510, color="gray", linestyle="--", linewidth=1, label="round4 best 0.510")
ax.set_xlabel("epoch")
ax.set_ylabel("COCO val Pose mAP50-95")
ax.set_title("(a) COCO 验证集 Pose mAP50-95 谱系")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# (b) pose_loss（train 实线 / val 虚线）
ax = axes[0, 1]
for label, d in runs_data.items():
    if d is None:
        continue
    tr = col(d, "train/pose_loss")
    va = col(d, "val/pose_loss")
    if tr:
        ax.plot(range(1, len(tr) + 1), tr, label=label, color=colors[label],
                linestyle=styles.get(label, "-"), linewidth=1.4, alpha=0.85)
    if va:
        ax.plot(range(1, len(va) + 1), va, color=colors[label],
                linestyle=":", linewidth=1.0, alpha=0.6)
ax.set_xlabel("epoch")
ax.set_ylabel("pose_loss")
ax.set_title("(b) 姿态损失曲线（实线=train，点线=val）")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# (c) 学习率曲线
ax = axes[1, 0]
for label, d in runs_data.items():
    if d is None:
        continue
    lr = col(d, "lr/pg0")
    if lr:
        ax.plot(range(1, len(lr) + 1), lr, label=label, color=colors[label],
                linestyle=styles.get(label, "-"), linewidth=1.4)
ax.set_xlabel("epoch")
ax.set_ylabel("lr (pg0)")
ax.set_yscale("log")
ax.set_title("(c) 学习率曲线（warmup 3 epoch + 余弦退火）")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3, which="both")

# (d) 训练数据组成（堆叠柱）
ax = axes[1, 1]
labels = list(DATA_COMP.keys())
keys = ["COCO", "MPII", "CrowdPose", "OCHuman", "OTP"]
kcolors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
bottom = [0] * len(labels)
for i, k in enumerate(keys):
    vals = [DATA_COMP[l][k] for l in labels]
    ax.bar(labels, vals, bottom=bottom, label=k, color=kcolors[i], width=0.5)
    bottom = [b + v for b, v in zip(bottom, vals)]
for j, b in enumerate(bottom):
    ax.text(j, b + 900, f"{b:,}", ha="center", fontsize=9)
ax.set_ylabel("训练图像数")
ax.set_title("(d) 各轮训练数据组成")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("YOLO11s-pose 训练谱系（2026-09-26 ~ 09-28）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{RUNS}/bench/stage_lineage.png", dpi=150)
print("saved", f"{RUNS}/bench/stage_lineage.png")

# ---------------- 图2：三基准 × 检查点 消融柱图 ----------------
fig2, ax2 = plt.subplots(figsize=(12, 5.5))
groups = [g for g in ["官方预训练", "round3", "round4", "round5-2", "round5-3",
                      "去CrowdPose", "去OCHuman", "去OTP"]]
sets = ["COCO", "CrowdPose", "OCHuman"]
scolors = ["#4c72b0", "#c44e52", "#55a868"]
w = 0.26
for i, s in enumerate(sets):
    vals, xs = [], []
    for gi, g in enumerate(groups):
        b = BENCH.get(g)
        if b and b.get(s) is not None:
            vals.append(b[s])
            xs.append(gi + (i - 1) * w)
    bars = ax2.bar(xs, vals, width=w, label=s, color=scolors[i])
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", fontsize=8)
ax2.set_xticks(range(len(groups)))
ax2.set_xticklabels(groups, fontsize=9)
ax2.set_ylabel("Pose mAP50-95")
ax2.set_title("三基准 Pose mAP50-95 × 检查点（消融对比）")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig(f"{RUNS}/bench/stage_ablation.png", dpi=150)
print("saved", f"{RUNS}/bench/stage_ablation.png")
