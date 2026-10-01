"""round3→round5 训练谱系可视化：COCO val 曲线、val loss、遮挡域基准柱状图。

输出 runs/bench/round5_lineage.png，并打印各轮 best 值便于记录。
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUNS = [
    ("round3", "/home/ubuntu/isef/runs/pose-round3/results.csv"),
    ("round4", "/home/ubuntu/isef/runs/pose-round4/results.csv"),
    ("round5(bad-lr)", "/home/ubuntu/isef/runs/pose-round5/results.csv"),
    ("round5-2", "/home/ubuntu/isef/runs/pose-round5-2/results.csv"),
]
OUT = "/home/ubuntu/isef/runs/bench/round5_lineage.png"

# 中文字体探测，缺失则退英文标签
FONT = None
for cand in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]:
    if Path(cand).exists():
        FONT = cand
        break
ZH = FONT is not None
if ZH:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()

COLORS = {
    "round3": "#9aa0a6",
    "round4": "#4285f4",
    "round5(bad-lr)": "#ea4335",
    "round5-2": "#34a853",
}


def load(p):
    rows = list(csv.DictReader(open(p)))
    ep = [int(r["epoch"]) for r in rows]
    m = [float(r["metrics/mAP50-95(P)"]) for r in rows]
    vloss = [float(r["val/pose_loss"]) for r in rows]
    return ep, m, vloss


fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
fig.suptitle(
    "pose-round3 → round5 谱系（86k 训练图，val 恒定 4,924 图）",
    fontsize=13,
)

for name, path in RUNS:
    ep, m, vloss = load(path)
    c = COLORS[name]
    axes[0].plot(ep, m, marker="o", ms=3.5, lw=1.6, color=c, label=f"{name}  best={max(m):.4f}")
    axes[1].plot(ep, vloss, marker=".", ms=2, lw=1.1, color=c, alpha=0.85, label=name)

# 子图 1：COCO val Pose mAP50-95
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("COCO val Pose mAP50-95")
axes[0].set_title("COCO val（同 val 集可比）")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].axhline(0.510, color="#4285f4", ls="--", lw=1, alpha=0.6)
axes[0].axhline(0.390, color="#9aa0a6", ls=":", lw=1, alpha=0.8)
axes[0].text(61, 0.393, "官方预训练 0.390", fontsize=8, color="#5f6368")

# 子图 2：val pose_loss（过拟合监测）
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("val pose_loss")
axes[1].set_title("val loss（未分叉 = 重复训练仍有价值）")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# 子图 3：遮挡域基准柱状图（含官方预训练基线）
groups = ["官方预训练", "round3", "round4", "round5-2"]
crowd = [0.474, 0.465, 0.543, 0.532]
ochu = [0.272, None, 0.331, 0.395]
x0 = range(len(groups))
bars1 = axes[2].bar([i - 0.2 for i in x0], crowd, width=0.38, color="#4285f4", label="CrowdPose val 遮挡代理")
bars2 = axes[2].bar([i + 0.2 for i, v in enumerate(ochu) if v is not None],
                    [v for v in ochu if v is not None], width=0.38, color="#34a853",
                    label="OCHuman test（首建基线）")
for b in list(bars1) + list(bars2):
    axes[2].annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8)
axes[2].set_xticks(list(x0))
axes[2].set_xticklabels(groups)
axes[2].set_ylabel("Pose mAP50-95")
axes[2].set_title("遮挡域基准（独立 val，含预训练基线）")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
for name, path in RUNS:
    ep, m, _ = load(path)
    print(f"{name}: best epoch {ep[m.index(max(m))]} = {max(m):.5f}")
