#!/usr/bin/env python3
"""results.csv 训练曲线绘图：按表头取列（禁固定列号，新 loss 列插入不再错位）。

用法:
  plot_results_csv.py results.csv [out.png] [--epoch-col epoch] [--cols col1,col2,...]

默认 3 面板:
  ① loss 类列（表头含 loss）
  ② metrics 类列（表头含 metrics/mAP/precision/recall）
  ③ lr 类列（表头含 lr）

表头取列规则: 列名匹配按小写包含；空白行（warmup 期）自动跳过。
"""
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="按表头绘制训练 CSV 曲线")
    ap.add_argument("csv_path", help="results.csv 路径")
    ap.add_argument("out_png", nargs="?", default=None, help="输出 PNG（默认与 csv 同名）")
    ap.add_argument("--epoch-col", default="epoch", help="epoch 列名（默认 epoch）")
    ap.add_argument("--cols", default=None, help="手动指定列（逗号分隔），覆盖自动分面板")
    return ap.parse_args()


def numeric(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "").strip()
        if v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def main():
    args = parse_args()
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get(args.epoch_col, "").strip() != ""]
    if not rows:
        sys.exit(f"无有效数据行: {args.csv_path}")
    header = [k.strip() for k in rows[0].keys()]

    def match(sub):
        return [k for k in header if sub in k.lower()]

    epoch_key = args.epoch_col if args.epoch_col in header else next(
        (k for k in header if "epoch" in k.lower()), None)
    if epoch_key is None:
        sys.exit(f"找不到 epoch 列: {header}")

    if args.cols:
        panels = [("指定列", [c.strip() for c in args.cols.split(",")])]
    else:
        metrics_cols = []
        for sub in ("metrics", "map", "precision", "recall"):
            for k in match(sub):
                if k not in metrics_cols:
                    metrics_cols.append(k)
        panels = [
            ("loss", match("loss")),
            ("metrics", metrics_cols),
            ("lr", match("lr")),
        ]
    panels = [(t, c) for t, c in panels if c]

    x = numeric(rows, epoch_key)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)), squeeze=False)
    summary = []
    for (title, cols), ax in zip(panels, axes[:, 0]):
        for col in cols:
            y = numeric(rows, col)
            if not y:
                continue
            n = min(len(x), len(y))
            ax.plot(x[:n], y[:n], label=col, linewidth=1.2)
            summary.append(f"{col}: 末值={y[-1]:.6g} (n={n})")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(epoch_key)
    fig.tight_layout()
    out = args.out_png or args.csv_path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"已保存: {out}")
    for s in summary:
        print(f"  {s}")


if __name__ == "__main__":
    main()
