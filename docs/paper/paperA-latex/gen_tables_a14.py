#!/usr/bin/env python3
"""表 A14 生成器：a14_fdr.json → a14_en.tex + a14_zh.tex 附录表片段。

Δ 约定（与 p1_data.py 一致，与主协议 A1-A12 相反）：
Δ = round5-3 − YOLO11s-pose。F1 为 AP（Δ>0 表示 round5-3 占优），
F2/F3 为 RMSE/抖动（Δ>0 表示 YOLO11s-pose 占优）。表头与表注自述约定。
"""
import json
import sys

DATASETS_EN = {"coco-val": "COCO val", "crowdpose-val": "CrowdPose val",
               "ochuman-test": "OCHuman test"}
JOINTS_EN = {"left_elbow": "left elbow", "right_elbow": "right elbow",
             "left_knee": "left knee", "right_knee": "right knee"}
JOINTS_ZH = {"left_elbow": "左肘", "right_elbow": "右肘",
             "left_knee": "左膝", "right_knee": "右膝"}
M3_EN = {"disp_mean": r"disp mean", "jitter_mean": r"jitter mean",
         "disp_p95": r"disp P95", "jitter_p95": r"jitter P95",
         "jump_0.05": r"jump rate $\tau{=}0.05$",
         "jump_0.1": r"jump rate $\tau{=}0.10$",
         "jump_0.2": r"jump rate $\tau{=}0.20$"}
M3_ZH = {"disp_mean": "位移均值", "jitter_mean": "抖动均值",
         "disp_p95": "位移 P95", "jitter_p95": "抖动 P95",
         "jump_0.05": r"跳变率 $\tau{=}0.05$",
         "jump_0.1": r"跳变率 $\tau{=}0.10$",
         "jump_0.2": r"跳变率 $\tau{=}0.20$"}
FAM_EN = {"F1_M1_full_AP": r"\textbf{F1} \ M1 full AP",
          "F2_M2_rmse_cells": r"\textbf{F2} \ M2 per-cell RMSE ($^{\circ}$)",
          "F3_M3_temporal": r"\textbf{F3} \ M3 temporal (PoseTrack val)"}
FAM_ZH = {"F1_M1_full_AP": r"\textbf{F1} \ M1 全量 AP",
          "F2_M2_rmse_cells": r"\textbf{F2} \ M2 逐格 RMSE（度）",
          "F3_M3_temporal": r"\textbf{F3} \ M3 时序指标（PoseTrack val）"}
ORDER = ["F1_M1_full_AP", "F2_M2_rmse_cells", "F3_M3_temporal"]


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def parse_label(label, lang):
    parts = label.split(" | ") if " | " in label else label.split()
    fam = None
    if label.startswith("M1 "):
        fam, ds = "F1_M1_full_AP", DATASETS_EN[label[3:]]
        cmp_txt = ds
    elif label.startswith("M2 "):
        _, ds, joint, vbin = label.split()
        fam = "F2_M2_rmse_cells"
        cmp_txt = (f"{DATASETS_EN[ds]} \\ {JOINTS_EN[joint]} \\ $v{{=}}{vbin[1]}$"
                   if lang == "en" else
                   f"{DATASETS_EN[ds]} \\ {JOINTS_ZH[joint]} \\ $v{{=}}{vbin[1]}$")
    elif label.startswith("M3 "):
        _, ds, m = label.split()
        fam = "F3_M3_temporal"
        cmp_txt = M3_EN[m] if lang == "en" else M3_ZH[m]
    else:
        raise ValueError(f"未知标签: {label}")
    return fam, cmp_txt


def build_rows(data, lang):
    rows = []
    for fam in ORDER:
        fam_data = data[fam]
        n, n_raw, n_bh = fam_data["n"], fam_data["n_sig_raw"], fam_data["n_sig_bh"]
        header = (FAM_EN[fam] if lang == "en" else FAM_ZH[fam]) + \
                 f" ($n{{=}}{n}$; raw sig {n_raw}; BH sig {n_bh})"
        rows.append(("family", header, None))
        for label, it in fam_data["rows"].items():
            f, cmp_txt = parse_label(label, lang)
            assert f == fam, f"{label} 不属于 {fam}"
            rows.append(("row", cmp_txt, it))
    return rows


def emit(data, lang):
    rows = build_rows(data, lang)
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    if lang == "en":
        out.append(r"\caption{Benjamini--Hochberg FDR sensitivity analysis over the "
                   r"core comparison families (round5-3 vs.\ YOLO11s-pose). "
                   r"$\Delta=$ round5-3 $-$ YOLO11s-pose: $\Delta>0$ favors round5-3 "
                   r"for AP (F1) and favors YOLO11s-pose for RMSE and jitter (F2, F3).}")
    else:
        out.append(r"\caption{核心比较族的 Benjamini--Hochberg FDR 敏感性分析"
                   r"（round5-3 对 YOLO11s-pose）。$\Delta=$ round5-3 $-$ YOLO11s-pose："
                   r"$\Delta>0$ 对 AP（F1）表示 round5-3 占优，对 RMSE 与抖动（F2、F3）"
                   r"表示 YOLO11s-pose 占优。}")
    out.append(r"\label{tab:a14}")
    out.append(r"\renewcommand{\arraystretch}{1.15}")
    out.append(r"\resizebox{\columnwidth}{!}{%")
    out.append(r"\begin{tabular}{@{}lllrrr@{}}")
    out.append(r"\toprule")
    out.append(r"Family & Comparison & $\Delta$ & 95\% CI & $p$ & $p_{\mathrm{BH}}$ \\")
    out.append(r"\midrule")
    for kind, txt, it in rows:
        if kind == "family":
            out.append(r"\multicolumn{6}{@{}l}{" + txt + r"} \\")
        else:
            d = fmt(it["point_delta"])
            lo, hi = fmt(it["ci_lo"]), fmt(it["ci_hi"])
            p = fmt(it["p"])
            padj = fmt(it["p_adj"])
            star = r"$^{*}$" if it["sig_adj"] else ""
            out.append(f"& {txt} & {d} & [{lo}, {hi}] & {p} & {padj}{star} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\vspace{0.4em}")
    if lang == "en":
        out.append(r"\footnotesize $^{*}$ BH-adjusted $p<0.05$ within the family. "
                   r"$p$: two-sided bootstrap ($2\min(P(\Delta\le0),P(\Delta\ge0))$), "
                   r"10,000 resamples, seed 42; significance by whether the 95\% CI "
                   r"excludes 0. F1: paired image universe (union of GT and predicted "
                   r"images of both models); F2: image-level independent resampling; "
                   r"F3: track-segment-level independent resampling. Raw sig counts "
                   r"CI-excludes-0 comparisons before BH adjustment.")
    else:
        out.append(r"\footnotesize $^{*}$ 族内 BH 校正后 $p<0.05$。$p$：双侧 bootstrap"
                   r"（$2\min(P(\Delta\le0),P(\Delta\ge0))$），10,000 次重采样，seed 42；"
                   r"显著性以 95\% CI 是否含 0 判定。F1：配对图像域（两模型 GT 与预测图"
                   r"并集）；F2：图像级独立重采样；F3：轨迹段级独立重采样。raw sig 为 "
                   r"BH 校正前 CI 不含 0 的比较数。")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "a14_fdr.json"))
    open("a14_en.tex", "w").write(emit(data, "en"))
    open("a14_zh.tex", "w").write(emit(data, "zh"))
    n_total = sum(data[f]["n"] for f in ORDER)
    print(f"A14 片段已生成: a14_en.tex / a14_zh.tex（合计 {n_total} 行比较）")


if __name__ == "__main__":
    main()
