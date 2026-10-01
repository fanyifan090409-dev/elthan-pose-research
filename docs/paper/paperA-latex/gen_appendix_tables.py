#!/usr/bin/env python3
"""从 summary.json + extra_ci.json 生成论文 A 附录 A 的 bootstrap CI 表（LaTeX）。

输出 appendix_tables.tex，由 main.tex 在 \\appendices 段 \\input。
每表固定比较清单，不做筛选；sig = 95% CI 不含 0。
"""
import argparse
import json

JOINTS = (("left_elbow", "Left elbow"), ("right_elbow", "Right elbow"),
          ("left_knee", "Left knee"), ("right_knee", "Right knee"))
LAYERS = (("full", "full"), ("v2", r"$v{=}2$"), ("v1", r"$v{=}1$"),
          ("v0", r"$v{=}0$"))
DS = (("coco-val", "COCO val"), ("crowdpose-val", "CrowdPose val"),
      ("ochuman-test", "OCHuman test"))
TAUS = (("10", r"$10^{\circ}$"), ("15", r"$15^{\circ}$"), ("17", r"$17^{\circ}$"))
M3 = (("disp_mean", r"$\bar d$"), ("jitter_mean", r"$\bar j$"),
      ("disp_p95", r"$d_{95}$"), ("jitter_p95", r"$j_{95}$"),
      ("jump_0.05", r"jump$_{0.05}$"), ("jump_0.1", r"jump$_{0.10}$"),
      ("jump_0.2", r"jump$_{0.20}$"))
TAUS_MD = ("10°", "15°", "17°")


def f(x, nd=4):
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}"


def ci(x, nd=4):
    return f"{f(x['ci_lo'], nd)}, {f(x['ci_hi'], nd)}"


def sig_mark(x):
    return r"$^{*}$" if x.get("sig") else ""


def pt_ci(x, nd=4):
    return f"{f(x['pt'], nd)} [{ci(x, nd)}]"


def table(label, caption, note, tabular):
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"{tabular}"
        "}\n"
        f"\\vspace{{0.4em}}\\footnotesize {note}\n"
        "\\end{table}\n\n"
    )


def gen_m1(summary):
    """表 A1：M1 逐层 AP 与模型差 CI（图像级配对）。"""
    rows = []
    for ds_key, ds_name in DS:
        b = summary["m1"][ds_key]["bootstrap"]
        a = summary["m1"][ds_key]["yolo11s-pose"]
        bb = summary["m1"][ds_key]["round5-3"]
        for l_key, l_name in LAYERS:
            ap_a = a["full"]["ap"] if l_key == "full" else a["layers"][l_key]["ap"]
            ap_b = bb["full"]["ap"] if l_key == "full" else bb["layers"][l_key]["ap"]
            x = b[l_key]
            rows.append(
                f"{ds_name} & {l_name} & {f(ap_b)} & {f(ap_a)} & "
                f"{f(x['point_delta'])} [{ci(x)}]{sig_mark(x)} \\\\"
            )
    body = "\n".join(rows)
    return table(
        "tab:a1",
        "M1 per-layer AP comparison with bootstrap CIs. "
        r"$\Delta=$ YOLO11s-pose $-$ round5-3; $^{*}$ CI excludes 0.",
        "Unit: image (both models share the same image universe; paired resampling). "
        "10,000 resamples, seed 42, 95\\% percentile CI.",
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Layer} & \\textbf{AP r5-3} & "
        "\\textbf{AP YOLO} & \\textbf{$\\Delta$ AP [95\\% CI]} \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
    )


def gen_delta_occ(extra):
    """表 A2：M1 Δ_occ 逐模型与模型差 CI（图像级配对）。"""
    rows = []
    for ds_key, ds_name in DS:
        x = extra["m1_delta_occ"][ds_key]
        rows.append(
            f"{ds_name} & {pt_ci(x['b'])} & {pt_ci(x['a'])} & "
            f"{pt_ci(x['diff'])}{sig_mark(x['diff'])} \\\\"
        )
    body = "\n".join(rows)
    return table(
        "tab:a2",
        r"M1 $\Delta_{\mathrm{occ}}$ (mAP($v{=}1$) $-$ mAP(full)) per model, and the "
        "model difference, with bootstrap CIs.",
        r"Unit: image (paired); $\Delta\Delta_{\mathrm{occ}}$ is computed element-wise "
        "on the paired resample. 10,000 resamples, seed 42, 95\\% percentile CI.",
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "\\textbf{Dataset} & "
        r"$\Delta_{\mathrm{occ}}$ r5-3 [95\% CI] & "
        r"$\Delta_{\mathrm{occ}}$ YOLO [95\% CI] & "
        r"$\Delta\Delta_{\mathrm{occ}}$ [95\% CI] \\"
        "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
    )


def gen_m2(summary):
    """表 A3：M2 角度 RMSE 逐关节/层与模型差 CI（图像级，逐模型独立重采样）。"""
    rows = []
    for ds_key, ds_name in DS:
        b = summary["m2"][ds_key]["bootstrap"]
        for j_key, j_name in JOINTS:
            for l_key, l_name in (("v2", r"$v{=}2$"), ("v1", r"$v{=}1$")):
                x = b[j_key][l_key]
                rows.append(
                    f"{ds_name} & {j_name} & {l_name} & {f(x['rmse_b'], 1)} & "
                    f"{f(x['rmse_a'], 1)} & {f(x['point_delta'], 1)} "
                    f"[{ci(x, 1)}]{sig_mark(x)} & {x['n_img_b']} / {x['n_img_a']} \\\\"
                )
    body = "\n".join(rows)
    return table(
        "tab:a3",
        r"M2 joint-angle RMSE ($^{\circ}$, sides pooled) with bootstrap CIs. "
        r"$\Delta=$ YOLO11s-pose $-$ round5-3 (negative $=$ YOLO lower error); "
        r"$^{*}$ CI excludes 0.",
        "Unit: image, resampled independently per model (each model's image set "
        "contains images with at least one valid instance of that joint and layer); "
        r"last column $n_{\mathrm{img}}$: round5-3 / YOLO11s-pose. 10,000 resamples, "
        "seed 42, 95\\% percentile CI.",
        "\\begin{tabular}{lllcccc}\n\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Joint} & \\textbf{Layer} & \\textbf{RMSE r5-3} & "
        "\\textbf{RMSE YOLO} & \\textbf{$\\Delta$ RMSE [95\\% CI]} "
        "& \\textbf{$n_{\\mathrm{img}}$} \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
    )


def gen_kappa(extra):
    """表 A4：M2 κ/sens/spec 逐 τ 与模型差 CI（图像级，逐模型独立重采样）。"""
    rows = []
    for ds_key, ds_name in DS:
        for t_key, t_name in TAUS:
            x = extra["m2_kappa"][ds_key][t_key]
            rows.append(
                f"{ds_name} & {t_name} & {pt_ci(x['b']['kappa'], 3)} & "
                f"{pt_ci(x['a']['kappa'], 3)} & "
                f"{pt_ci(x['diff_kappa'], 3)}{sig_mark(x['diff_kappa'])} & "
                f"{pt_ci(x['diff_sens'], 3)} & {pt_ci(x['diff_spec'], 3)} \\\\"
            )
    body = "\n".join(rows)
    return table(
        "tab:a4",
        r"Knee-valgus detection consistency ($\kappa$, sensitivity, specificity) with "
        r"bootstrap CIs. $\Delta=$ YOLO11s-pose $-$ round5-3; $^{*}$ CI excludes 0.",
        "Unit: image (binarized $\\phi$ instances pooled per image), resampled "
        "independently per model. 10,000 resamples, seed 42, 95\\% percentile CI.",
        "\\begin{tabular}{llccc cc}\n\\toprule\n"
        "\\textbf{Dataset} & \\textbf{$\\tau_\\varphi$} & "
        "\\textbf{$\\kappa$ r5-3 [95\\% CI]} & \\textbf{$\\kappa$ YOLO [95\\% CI]} & "
        "\\multicolumn{3}{c}{\\textbf{Model difference [95\\% CI]}} \\\\\n"
        " & & & & $\\Delta\\kappa$ & $\\Delta$ sens. & $\\Delta$ spec. \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
    )


def gen_m3(extra):
    """表 A5：M3 逐域值与域差 CI、模型差 CI（轨迹段级）。"""
    wm = extra["m3_domains"]["within_model"]
    bm = extra["m3_domains"]["between_models"]
    rows = []
    for m_key, m_name in M3:
        r5 = wm["round5-3"]
        yo = wm["yolo11s-pose"]
        raw_r5 = r5["per_domain"]["raw"][m_key]["pt"]
        raw_yo = yo["per_domain"]["raw"][m_key]["pt"]
        d_raw = bm["raw"][m_key]
        px_r5 = r5["px-raw"][m_key]
        px_yo = yo["px-raw"][m_key]
        q_r5 = r5["q-raw"][m_key]
        q_yo = yo["q-raw"][m_key]
        rows.append(
            f"{m_name} & {f(raw_r5)} & {f(raw_yo)} & "
            f"{pt_ci(d_raw)}{sig_mark(d_raw)} & "
            f"{pt_ci(px_r5)}{sig_mark(px_r5)} & {pt_ci(px_yo)}{sig_mark(px_yo)} & "
            f"{pt_ci(q_r5)}{sig_mark(q_r5)} & {pt_ci(q_yo)}{sig_mark(q_yo)} \\\\"
        )
    body = "\n".join(rows)
    return table(
        "tab:a5",
        r"M3 temporal metrics: per-domain point values, within-model domain "
        r"differences (px$-$raw, q$-$raw), and between-model difference on the raw "
        r"domain, all with bootstrap CIs. $\Delta$ (model) $=$ YOLO11s-pose $-$ "
        r"round5-3; $^{*}$ CI excludes 0.",
        "Unit: track segment. Domain differences are paired (same resample index "
        "across domains, segments fixed by the raw matching); between-model "
        "differences resample each model's segments independently. 10,000 resamples, "
        "seed 42 (1042 for the YOLO11s-pose between-model distribution), 95\\% "
        "percentile CI.",
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        "\\textbf{Metric} & \\textbf{raw r5-3} & \\textbf{raw YOLO} & "
        "\\textbf{$\\Delta$ raw} & \\textbf{px$-$raw r5-3} & \\textbf{px$-$raw YOLO} & "
        "\\textbf{q$-$raw r5-3} & \\textbf{q$-$raw YOLO} \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
    )


def md_ci(x, nd=4, star=True):
    mark = '*' if (star and x.get('sig')) else ''
    return f"{f(x['pt'], nd)} [{f(x['ci_lo'], nd)}, {f(x['ci_hi'], nd)}]{mark}"


def gen_md(summary, extra):
    """附录 A 的 Markdown 版本（英文稿 md 直接粘贴用）。"""
    lines = [
        "## Appendix A: Bootstrap Confidence Intervals",
        "",
        "All intervals are 95% percentile bootstrap CIs from 10,000 resamples with seed 42. "
        "The resampling unit is the image for M1 and M2 (the unit over which each metric is "
        "aggregated before the model comparison) and the track segment for M3. Paired "
        "comparisons share the same resample index across the two quantities; between-model "
        "comparisons resample the two models' units independently. Sign convention: "
        "Δ = YOLO11s-pose − round5-3, so Δ<0 favors round5-3 for AP and κ, and favors "
        "YOLO11s-pose for RMSE. * marks intervals that exclude 0. Every entry of the fixed "
        "comparison list is tabulated; none is selected.",
        "",
        "**Table A1. M1 per-layer AP comparison with bootstrap CIs.** Unit: image (both "
        "models share the same image universe; paired resampling).",
        "",
        "| Dataset | Layer | AP r5-3 | AP YOLO | Δ AP [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for ds_key, ds_name in DS:
        b = summary["m1"][ds_key]["bootstrap"]
        a = summary["m1"][ds_key]["yolo11s-pose"]
        bb = summary["m1"][ds_key]["round5-3"]
        for l_key, l_name in LAYERS:
            ap_a = a["full"]["ap"] if l_key == "full" else a["layers"][l_key]["ap"]
            ap_b = bb["full"]["ap"] if l_key == "full" else bb["layers"][l_key]["ap"]
            x = b[l_key]
            lines.append(
                f"| {ds_name} | {l_key} | {f(ap_b)} | {f(ap_a)} | "
                f"{f(x['point_delta'])} [{f(x['ci_lo'])}, {f(x['ci_hi'])}]{'*' if x.get('sig') else ''} |"
            )
    lines += [
        "",
        "**Table A2. M1 Δ_occ (mAP(v=1) − mAP(full)) per model, and the model difference, "
        "with bootstrap CIs.** Unit: image (paired); ΔΔ_occ is computed element-wise on "
        "the paired resample.",
        "",
        "| Dataset | Δ_occ r5-3 [95% CI] | Δ_occ YOLO [95% CI] | ΔΔ_occ [95% CI] |",
        "|---|---|---|---|",
    ]
    for ds_key, ds_name in DS:
        x = extra["m1_delta_occ"][ds_key]
        lines.append(
            f"| {ds_name} | {md_ci(x['b'])} | {md_ci(x['a'])} | {md_ci(x['diff'])} |"
        )
    lines += [
        "",
        "**Table A3. M2 joint-angle RMSE (°, sides pooled) with bootstrap CIs.** Unit: "
        "image, resampled independently per model; last column n_img: round5-3 / "
        "YOLO11s-pose. Δ = YOLO11s-pose − round5-3 (negative = YOLO lower error).",
        "",
        "| Dataset | Joint | Layer | RMSE r5-3 | RMSE YOLO | Δ RMSE [95% CI] | n_img |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds_key, ds_name in DS:
        b = summary["m2"][ds_key]["bootstrap"]
        for j_key, j_name in JOINTS:
            for l_key, l_name in (("v2", "v=2"), ("v1", "v=1")):
                x = b[j_key][l_key]
                lines.append(
                    f"| {ds_name} | {j_name} | {l_name} | {f(x['rmse_b'], 1)} | "
                    f"{f(x['rmse_a'], 1)} | {f(x['point_delta'], 1)} "
                    f"[{f(x['ci_lo'], 1)}, {f(x['ci_hi'], 1)}]{'*' if x.get('sig') else ''} | "
                    f"{x['n_img_b']} / {x['n_img_a']} |"
                )
    lines += [
        "",
        "**Table A4. Knee-valgus detection consistency (κ, sensitivity, specificity) with "
        "bootstrap CIs.** Unit: image (binarized φ instances pooled per image), resampled "
        "independently per model. Δ = YOLO11s-pose − round5-3.",
        "",
        "| Dataset | τ_φ | κ r5-3 [95% CI] | κ YOLO [95% CI] | Δκ [95% CI] | Δ sens. [95% CI] | Δ spec. [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds_key, ds_name in DS:
        for t_key, t_name in zip(("10", "15", "17"), TAUS_MD):
            x = extra["m2_kappa"][ds_key][t_key]
            lines.append(
                f"| {ds_name} | {t_name} | {md_ci(x['b']['kappa'], 3, star=False)} | "
                f"{md_ci(x['a']['kappa'], 3, star=False)} | {md_ci(x['diff_kappa'], 3)} | "
                f"{md_ci(x['diff_sens'], 3)} | {md_ci(x['diff_spec'], 3)} |"
            )
    lines += [
        "",
        "**Table A5. M3 temporal metrics: per-domain point values, within-model domain "
        "differences (px−raw, q−raw), and between-model difference on the raw domain, all "
        "with bootstrap CIs.** Unit: track segment; domain differences are paired (same "
        "resample index), between-model differences resample independently. Δ (model) = "
        "YOLO11s-pose − round5-3.",
        "",
        "| Metric | raw r5-3 | raw YOLO | Δ raw [95% CI] | px−raw r5-3 [95% CI] | px−raw YOLO [95% CI] | q−raw r5-3 [95% CI] | q−raw YOLO [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    wm = extra["m3_domains"]["within_model"]
    bm = extra["m3_domains"]["between_models"]
    for m_key, _ in M3:
        r5 = wm["round5-3"]
        yo = wm["yolo11s-pose"]
        raw_r5 = r5["per_domain"]["raw"][m_key]["pt"]
        raw_yo = yo["per_domain"]["raw"][m_key]["pt"]
        lines.append(
            f"| {m_key} | {f(raw_r5)} | {f(raw_yo)} | {md_ci(bm['raw'][m_key])} | "
            f"{md_ci(r5['px-raw'][m_key])} | {md_ci(yo['px-raw'][m_key])} | "
            f"{md_ci(r5['q-raw'][m_key])} | {md_ci(yo['q-raw'][m_key])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--extra-ci", required=True)
    ap.add_argument("--out", default="appendix_tables.tex")
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    summary = json.load(open(args.summary))
    extra = json.load(open(args.extra_ci))

    parts = [
        gen_m1(summary),
        gen_delta_occ(extra),
        gen_m2(summary),
        gen_kappa(extra),
        gen_m3(extra),
    ]
    with open(args.out, "w") as fh:
        fh.write("% Auto-generated by gen_appendix_tables.py. Do not edit by hand.\n\n")
        fh.write("\n".join(parts))
    print(f"wrote {args.out}")

    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(gen_md(summary, extra))
        print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
