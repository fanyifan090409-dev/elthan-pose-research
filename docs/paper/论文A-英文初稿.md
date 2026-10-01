# Uncovering the Blind Spots of OKS-mAP: A Unified Protocol for Occlusion, Joint-Angle, and Temporal Evaluation of 2D Pose Estimators

> English draft of Paper A (2026-10-01), translated from the Chinese manuscript. Author: Yifan Fan (Independent Researcher).
> All numbers are locked values from the G3 baseline experiments (summary.json, schema_version=1). The Chinese draft remains the source of truth until author sign-off.

## Abstract

The dominant evaluation protocol for 2D human pose estimation — mean average precision based on Object Keypoint Similarity (OKS-mAP) — is insensitive to three properties that matter for downstream applications: it does not distinguish visible, occluded, and missing keypoints; it does not measure joint-angle error; and it is silent on the temporal stability of outputs produced by tracking-free, frame-by-frame inference. We propose a unified supplementary evaluation protocol with three modules: (M1) OKS-mAP stratified by ground-truth visibility labels (v), together with Δ_occ = mAP(v=1) − mAP(full), which quantifies how much the aggregate metric masks occlusion-specific differences; (M2) joint-angle error (RMSE/MAE/P95) against angles derived from ground-truth keypoints, plus knee-valgus detection consistency (Cohen's κ); and (M3) frame-to-frame displacement, median-filter residual jitter, and jump rate under tracking-free per-frame inference, before and after OneEuro smoothing. We executed the protocol on four public datasets (COCO val2017, CrowdPose, OCHuman, PoseTrack18) for two representative models (the official YOLO11s-pose and a model fine-tuned with occlusion-domain data), with all metrics produced by an open-source harness in a single inference pass. Δ_occ is negative for every model–dataset combination (−0.24 to −0.41): the aggregate mAP systematically overestimates occluded-scene performance. Joint-angle error is systematically amplified in occluded layers, and on the occlusion-domain datasets the angle metrics reveal significant differences between the two models that the aggregate mAP cannot localize. The two models show no significant difference in raw temporal jitter, while pixel-domain OneEuro smoothing reduces the jump rate by 9–10% at zero accuracy cost (|ΔAP| ≤ 0.0002). Correlation analysis shows that stratified mAP is highly consistent with full mAP (Kendall τ = 1.0) — its value lies in magnitude diagnosis rather than re-ranking — while the angle metrics, particularly knee angle, carry partially independent information (ρ = 0.72, n = 6, exploratory). The protocol and harness are open-sourced and positioned as a complement to, not a replacement for, OKS-mAP.

**Keywords**: human pose estimation; evaluation protocol; occlusion; joint angle; temporal stability; OKS mAP

## 1 Introduction

2D human pose estimation is a fundamental computer vision task whose outputs feed action recognition, motion analysis, and real-time fitness-correction systems. For such applications, the gap between benchmark scores and real-world usability matters more than average precision alone: a model that ranks first on a benchmark can be markedly worse in scenes with occluded people, fast motion, or the need for precise joint-angle judgments — and this difference is invisible in the dominant metric.

The de facto standard evaluation protocol is OKS-mAP as defined by the COCO keypoints protocol [1]. It matches and scores predictions by Object Keypoint Similarity (OKS) and aggregates over the whole test set into a single scalar. This design has three blind spots. First, OKS aggregation ignores keypoint visibility: visible, occluded, and missing keypoints contribute equally to the same score, so occlusion-specific accuracy differences are diluted by averaging. Second, the protocol does not measure joint angles: OKS penalizes pixel-level keypoint position error, whereas downstream applications (e.g., exercise correction) care about whether the joint angles derived from those keypoints are accurate — and the two are not linearly related. Third, the protocol is blind to temporal behavior: most systems run frame-by-frame independent inference on video, and the jitter of the output between frames directly determines how the downstream consumer perceives it, yet a static-image protocol says nothing about it. Prior work responds to each problem in isolation — OCHuman organizes evaluation by occlusion level [2]; PoseTrack and SmoothNet extend the static protocol with tracking metrics (MOTA/MOTP) and a temporal-smoothness metric (per-joint acceleration magnitude, Accel), respectively [3,15]; and the biomechanics community has validated joint angles from markerless pose estimation [4–6]. What is missing is a unified, reproducible, cross-dataset supplementary protocol, and any baseline numbers quantifying how large these three blind spots are on real models.

This paper proposes a three-module unified supplementary evaluation protocol (Section 3) and open-sources its execution layer, an eval harness that produces official metrics and protocol metrics from a single inference pass, with three self-check gates. We executed the protocol on four public datasets with two representative models (Sections 4–5) to answer four research questions:

- **RQ1 (occlusion)**: How much does full OKS-mAP mask occlusion-specific accuracy differences? Does model ranking change after visibility stratification?
- **RQ2 (angle)**: Does joint-angle RMSE provide discriminative information beyond OKS-mAP? Is angle error amplified by occlusion?
- **RQ3 (temporal)**: Under tracking-free per-frame inference, how much do models differ in temporal stability? Can smoothing buy stability at an acceptable accuracy cost?
- **RQ4 (protocol economy)**: How do the metrics of the three modules correlate with each other and with full mAP? Is any of them redundant?

Our contributions are threefold: (1) to our knowledge, the first supplementary evaluation scheme that unifies occlusion stratification, angle error, and temporal jitter under one protocol, with explicit statistical conventions (bootstrap confidence intervals) and dataset-consistent stratification rules; (2) an open-source harness and all raw predictions, so that any new model can produce the full set of protocol metrics on the four datasets in one pass; (3) complete baseline numbers for two representative models, quantifying for the first time the magnitude of OKS-mAP's three blind spots (e.g., Δ_occ as large as −0.40 on COCO).

## 2 Related Work

**Pose estimation evaluation protocols.** The COCO keypoints protocol, with OKS at its core, has been the field's de facto standard since 2014 [1]. PoseTrack extends evaluation to multi-person pose tracking in video with a tracking convention centered on MOTA/MOTP [3]; SmoothNet goes further and proposes per-joint acceleration magnitude (Accel, in pixels/frame²) as a temporal-smoothness metric for video pose, and uses that convention in its filtering-vs-baseline comparisons [15]. Our M3 is complementary to Accel: Accel measures the size of the frame-to-frame second difference, whereas our frame-to-frame displacement and median-filter residual measure positional fluctuation and high-frequency noise separately, and we additionally report the accuracy (AP) cost of smoothing. On the data side, CrowdPose and OCHuman provide real visibility/occlusion labels (v=1 in CrowdPose; three occlusion levels in OCHuman) [2,7]. The common boundary of this prior work is that each improves a single dataset in a single dimension; none of it forms a cross-dataset supplementary protocol with unified conventions.

**Occlusion-robustness evaluation.** Prior work has attempted to quantify occlusion's effect on pose estimation: Ghafoor and Mahmood build three experiments with synthetically removed joints to quantify occlusion's impact on 3D pose error [8], and later work continues this line with robustness evaluation under synthetic occlusion [9] and a visibility-aware video occlusion benchmark [10]. The difference from our work: these studies use synthetic occlusions or non-real annotation-based stratification, and each focuses on a single dataset and a single metric. We stratify across datasets using real v labels from three datasets and report Δ_occ against full mAP.

**Joint-angle validation.** Biomechanics and sports science have a line of validity studies for markerless pose estimation: Turner et al. reconstruct joint-angle trajectories from BlazePose keypoints in 89 resistance-training videos and assess movement-technique quality by frame-wise RMSE against a reference movement [4]; Panconi et al. validate DeepLabCut joint angles in gait analysis [5]; Pemasiri et al. build a gait-analysis framework for biomechanical interpretation [6]. These studies answer the question "do estimated angles equal the clinical/marker-based gold standard?" (validity), and they require dedicated marker or expert annotations. We do not repeat validity validation; instead we use ground-truth-derived angles as a protocol metric for model-to-model comparison. The two lines are complementary rather than competing (Section 6 elaborates this boundary). The knee-valgus threshold takes its magnitude reference from the literature: frontal plane projection angle (FPPA) of 16.0–17.4° in a lateral step-down test [11].

**Temporal consistency and smoothing.** Temporal inconsistency of tracking-free per-frame inference has been discussed at a general level by Deep Video Prior and others [12]; in engineering practice, OneEuro causal filtering is widely used to smooth keypoint trajectories [13]. Our work turns "stability of tracking-free raw output" itself into a protocol metric, and systematically compares two filtering domains (normalized-coordinate vs. pixel) in their joint effect on jitter metrics and AP — a comparison that reveals how the choice of filtering domain decides the benefit and cost of smoothing.

## 3 Protocol Design

### 3.0 Overview

The protocol consists of three modules plus one statistical layer, all built on the common foundation of the COCO 17-keypoint definition and the OKS convention. Table 1 gives the overview.

| Module | Metrics | Data | Answers |
|---|---|---|---|
| M1 occlusion stratification | v-stratified mAP50-95(P) + Δ_occ | CrowdPose val / OCHuman test / COCO val | RQ1 |
| M2 angle error | elbow/knee angle RMSE/MAE/P95 + knee-valgus detection consistency (κ) | same as M1 (angles derived from GT, zero extra annotation) | RQ2 |
| M3 temporal jitter | frame-to-frame displacement / jitter / jump rate, before and after OneEuro | PoseTrack18 val (tracking-free per-frame inference) | RQ3 |
| Statistical layer | 10k bootstrap CI + Spearman/Kendall | results of the three modules | RQ4, model comparison |

**Table 1. Protocol overview.**

### 3.1 M1: Occlusion-Stratified mAP

**Stratification convention.** Each GT keypoint is assigned to a layer by its visibility label v: v=2 visible, v=1 occluded, v=0 missing. OKS AP is computed independently within each layer. The v=0 layer is reported but excluded from ranking (keypoints in this layer cannot be estimated and scores are near zero; the layer is kept as evidence of protocol completeness). Each dataset retains its real v labels after conversion to COCO17: CrowdPose's official 14 keypoints map to COCO indices 5–16 (the head_top and neck points are dropped — a mapping-induced truncation, not a corruption of visibility labels; the v=1 labels of the 12 mapped points are fully preserved); OCHuman's official test annotations are already in COCO17 format and pass through unchanged; COCO val itself contains v=1 labels (7.75% of keypoints).

**Computation convention.** We follow the keypoints protocol of pycocotools 2.0.11 (the convention behind official released numbers): mean over 10 OKS thresholds 0.50:0.95; GTs with num_keypoints=0 marked ignore; maxDets=20. Stratification changes only the statistics, not the matching logic: person matching (TP/FP decisions) still uses full-keypoint OKS, and a layer's AP accumulates only the OKS contributions of that layer's GT keypoints. Three gates guarantee implementation correctness (agreement with the official library within 0.0001; synthetic self-checks; the stratification toggle leaves full mAP unchanged).

**Interpretability evidence.** A layer's AP has all GT persons as its recall denominator, so persons without that layer's keypoints can never contribute TPs to it. We therefore report, per layer, both "fraction of GT keypoints in this layer" (gt_frac) and "fraction of GT persons possessing this layer's keypoints" (gt_person_frac). In particular, CrowdPose's v2 layer suffers from the 14→17 mapping truncation (gt_person_frac only 0.591), so its v2-layer AP is systematically underestimated — on CrowdPose, cross-layer comparisons between v2 and v1 should use Δ_occ (which is unaffected by this artifact).

**Primary metric.** Δ_occ = mAP(v=1 layer) − mAP(full); a negative value means the full metric overestimates occluded-scene performance. Per-layer mAP and gt_frac are reported alongside.

### 3.2 M2: Angle Error

**Joint-angle definition.** Four joints are defined by COCO17 indices (Table 2); the inner angle θ = arccos((a·b)/(|a||b|)), where a and b are the vectors from the vertex to the proximal and distal endpoints. Instances with |a| or |b| < 10⁻⁶ are dropped.

| Joint | Vertex | Endpoints (proximal, distal) |
|---|---|---|
| left elbow | 7 | 5, 9 |
| right elbow | 8 | 6, 10 |
| left knee | 13 | 11, 15 |
| right knee | 14 | 12, 16 |

**Table 2. Joint-angle definitions (COCO17 indices, 0-based).**

**Validity mask.** GT angle: all three GT points must have v > 0. Predicted angle: the same three GT points must have v > 0 and the predicted confidence must be > 0 — this keeps GT and predictions comparable on the same domain and avoids biasing toward missing predictions. Per-instance error e = θ_pred − θ_gt; we report RMSE, MAE, and P95, stratified by the vertex's GT visibility into v=2/v=1 layers, in linkage with M1.

**Knee-valgus detection consistency.** The knee-valgus angle φ is defined as the image-plane angle between the hip→knee vector and the hip→ankle vector (how far the knee point deviates from the hip–ankle axis), computed only for instances where all three points have v > 0. φ_pred and φ_GT are binarized at a threshold τ (default τ = 15°, sensitivity analysis τ ∈ {10°, 17°}), and we report detection sensitivity, specificity, and Cohen's κ with GT as the gold standard. Two clarifications: we do not report absolute detection rates (static datasets mix viewpoints, making absolute rates meaningless; consistency is what the protocol can measure); and φ is a 2D image-plane proxy under arbitrary viewpoints, whereas the FPPA literature values come from a standardized frontal camera [11] — our datasets mix viewpoints, so τ takes only a magnitude reference from the literature and carries no clinical-diagnostic meaning.

### 3.3 M3: Temporal Jitter

**Data and matching.** PoseTrack18 val videos are inferred frame by frame independently (no tracking, no cross-frame information); per-frame predictions are matched to GT by Hungarian matching on full-keypoint OKS (same convention as M1). Segments break when the frame gap exceeds 1; only track segments with ≥10 consecutive matched frames are retained. All displacements are normalized by the GT person-box scale s_t = √(w·h) (consistent with the OKS scale convention).

**Three metrics** (computed per keypoint and per track segment, then averaged):

1. frame-to-frame displacement d_t = ‖p_t − p_{t−1}‖₂ / s_t: report mean and P95;
2. jitter j_t = ‖p_t − ŝ_t‖₂ / s_t, where ŝ is the trajectory after a causal 5-frame median filter: targets high-frequency noise (core metric);
3. jump rate: fraction of frames with d_t > τ, default τ = 0.10 (10% of person-box scale), sensitivity τ ∈ {0.05, 0.20}.

**Before/after OneEuro comparison.** A causal OneEuro filter is applied offline to p_t with parameters fixed at Casiez's defaults (min_cutoff = 1.0, beta = 0.007, d_cutoff = 1.0) [13]. The three metrics and the matched-scope AP are recomputed after smoothing, and the accuracy cost is reported. Two filtering domains are used: the normalized-coordinate domain (q domain, the protocol's original design) and the pixel domain. The empirical finding (consistent across both models): filtering in the q domain re-injects the GT person box's per-frame scale noise (|s_t/s_{t−1} − 1|, median 0.97%, P95 8.1%) into the pixel domain during de-normalization, so jitter metrics rise rather than fall and AP drops slightly; direct pixel-domain filtering costs zero AP and lowers all three metrics. **We adopt the pixel domain as the primary convention**, and report the q domain as a sensitivity convention (Table 6, two columns).

### 3.4 Statistical Conventions

- Model comparison: M1/M2 resample at image level and M3 at track-segment level, 10,000 bootstrap replicates (seed 42); we report 95% percentile CIs of the difference Δ and judge significance by whether the CI excludes 0. M1 is a paired design (shared image universe = the union of GT and predicted images of both models); the valid unit sets of M2/M3 differ with matching results, so they use independent resampling, as noted in the protocol.
- The comparison list is fixed before the experiments (2 models × 3 datasets × layers/joints/metrics); every entry is reported with its CI, with no selection of favorable results.
- RQ4: on (dataset × layer) cells we compute the Kendall τ (sign consistency) between protocol-metric differences and full-mAP differences, and the Spearman ρ between metric differences. Cell count n = 6 (3 datasets × 2 layers), extended to n = 24 at joint level; this analysis is exploratory and the sample size is stated in the main text.

### 3.5 Harness and Reproducibility

The execution layer is an open-source eval harness: `model.val(save_json=True)` produces official metrics and prediction JSONs in a single inference pass; the M1/M2/M3 engines are purely offline (NumPy) and do not touch the GPU. Three gates guarantee the conventions are implemented correctly (see 3.1). The reproducibility contract (environment, weights, data, step-by-step commands, convention notes) is in the repository's REPRODUCIBILITY.md. The protocol's JSON contract (schema_version=1) is available for downstream plotting and secondary analysis.

## 4 Experimental Setup

**Datasets.** Four public datasets: COCO val2017 (general-domain control; v distribution 55.4%/7.8%/36.8% for v=2/1/0), CrowdPose val (crowded/occlusion domain; v=1 marks real occlusion; distribution 22.5%/30.8%/remainder), OCHuman test (heavy-occlusion domain; distribution 69.2%/14.3%/remainder), PoseTrack18 val (video temporal; 170 videos / 8,923 annotated frames; GT v ∈ {0,1}, where v=0 is a (0,0) placeholder). All datasets are unified under the COCO17 keypoint format. PoseTrack was obtained under MPII's official academic license and is used for academic evaluation only.

**Models.** Two baseline models: (1) the official YOLO11s-pose pretrained weights [14]; (2) round5-3, obtained by fine-tuning YOLO11s-pose on public datasets including COCO, MPII, CrowdPose, and OCHuman, representing training with occlusion-domain data. The two share the same architecture and inference configuration and differ only in training data, making them suitable for testing whether the protocol is sensitive to training-data differences.

**Inference and conventions.** 640×640 input, frame-by-frame independent inference, predictions saved with save_json. All mAP numbers use the official released-numbers convention of pycocotools 2.0.11. Note that this convention can differ from ultralytics' internal evaluator by more than 20 points (sources: GTs with num_keypoints=0 marked ignore, the OKS doubled-bbox fallback, maxDets=20); every table in the paper states its convention. PoseTrack has no official pycocotools reference, so its official numbers are reported only under the internal-evaluator convention (round5-3 0.1085 / YOLO11s-pose 0.1213 mAP50-95(P)).

## 5 Results

### 5.1 RQ1: Full mAP Masks Occlusion Differences

Table 3 gives the stratified mAP of the two models on three datasets. Three observations.

| Dataset | Model | Full | v=2 layer | v=1 layer | v=0 layer | Δ_occ |
|---|---|---|---|---|---|---|
| COCO val | round5-3 | 0.5712 | 0.5637 | 0.1747 | 0 | **−0.3965** |
| | YOLO11s-pose | **0.5816** | **0.5728** | 0.1719 | 0 | −0.4097 |
| CrowdPose val | round5-3 | **0.7626** | **0.4183** | **0.5183** | 0 | −0.2443 |
| | YOLO11s-pose | 0.6938 | 0.3768 | 0.4573 | 0 | −0.2366 |
| OCHuman test | round5-3 | **0.4775** | **0.4696** | **0.1819** | 0 | −0.2955 |
| | YOLO11s-pose | 0.3379 | 0.3314 | 0.0765 | 0 | −0.2614 |

**Table 3. M1 stratified mAP50-95(P) (pycocotools 2.0.11 convention). Bold = significantly better (bootstrap CI excludes 0).** CrowdPose's v2 layer is affected by the 14→17 mapping truncation (gt_person_frac 0.591); cross-layer comparisons use Δ_occ.

First, **Δ_occ is negative everywhere and large in magnitude**: −0.40 and −0.41 on COCO, −0.24 on CrowdPose, and −0.26 to −0.30 on OCHuman. Full mAP systematically overestimates occluded-scene performance by 24–41 points — a magnitude that can mask substantive differences in any model selection that relies on mAP.

Second, **stratification changes the strength of model-comparison conclusions**. On general-domain COCO, YOLO11s-pose is significantly better in full and v=2-layer mAP (Δ = +0.0104, 95% CI [0.0059, 0.0155]), but the v=1 layer shows no significant difference (Δ = −0.0028, ns) — the significant advantage in the aggregate comes almost entirely from visible keypoints. On the occlusion-domain datasets, round5-3 is significantly better in all three layers (CrowdPose full Δ = −0.0687, OCHuman full Δ = −0.1396, 95% CIs exclude 0), and the gain from occlusion-domain training is equally significant in the v=1 layer (OCHuman v1 Δ = −0.1054).

Third, **the v=0 layer is uniformly 0** (both models, three datasets), confirming the protocol's expectation: missing keypoints cannot be estimated; the layer is kept as completeness evidence, not for ranking.

### 5.2 RQ2: Angle Error and Knee-Valgus Detection Consistency

Table 4 gives elbow/knee angle RMSE with left and right sides pooled (stratified by v=2/v=1). Three observations.

| Dataset | Model | Elbow v=2 | Elbow v=1 | Knee v=2 | Knee v=1 |
|---|---|---|---|---|---|
| COCO val | round5-3 | 26.0 | 35.9 | 20.4 | 29.2 |
| | YOLO11s-pose | 25.4 | 35.6 | 21.0 | 31.2 |
| CrowdPose val | round5-3 | 25.9 | **31.6** | 20.6 | **25.9** |
| | YOLO11s-pose | 26.0 | 35.0 | 21.3 | 30.5 |
| OCHuman test | round5-3 | **35.6** | **45.7** | **24.3** | **32.0** |
| | YOLO11s-pose | 39.6 | 57.7 | 28.3 | 39.7 |

**Table 4. M2 joint-angle RMSE (°, left/right pooled by sample count). Bold = significantly better (bootstrap CI excludes 0).**

First, **angle error is systematically amplified by occlusion**: in every model–dataset combination, v=1-layer RMSE exceeds v=2-layer RMSE (elbow +5.7° to +18.1°, knee +5.3° to +11.4°), with the largest layer gaps in the heavy-occlusion domain (OCHuman). This aligns with M1's Δ_occ observation; the two corroborate each other.

Second, **the angle metrics separate information that full mAP cannot**. On general-domain COCO the two models show no significant angle difference (consistent with the 1-point mAP gap); but on the occlusion domains round5-3 is systematically more accurate in the v=1 layer: CrowdPose elbow 31.6° vs 35.0°, knee 25.9° vs 30.5° (both significant); OCHuman elbow 45.7° vs 57.7°, knee 32.0° vs 39.7° (all significant). Note that on OCHuman the v=2-layer angle difference is also significant, indicating that the benefit of occlusion-domain training is not limited to occluded keypoints themselves.

Third, **knee-valgus detection consistency** (Table 5): round5-3's κ exceeds YOLO11s-pose's on all three datasets (COCO 0.759 vs 0.745; CrowdPose 0.739 vs 0.669; OCHuman 0.602 vs 0.519), and the conclusion is stable under threshold sensitivity analysis (τ ∈ {10°, 17°}). Both models' κ on OCHuman are low (<0.61), suggesting that φ's measurement noise rises under heavy occlusion.

| Dataset | Model | τ=10° κ | τ=15° κ | τ=17° κ | τ=15° sensitivity / specificity |
|---|---|---|---|---|---|
| COCO val | round5-3 | 0.724 | **0.759** | 0.773 | 0.790 / 0.950 |
| | YOLO11s-pose | 0.697 | 0.745 | 0.759 | 0.783 / 0.943 |
| CrowdPose val | round5-3 | 0.711 | **0.739** | 0.731 | 0.785 / 0.937 |
| | YOLO11s-pose | 0.666 | 0.669 | 0.665 | 0.722 / 0.926 |
| OCHuman test | round5-3 | 0.601 | **0.602** | 0.603 | 0.646 / 0.926 |
| | YOLO11s-pose | 0.529 | 0.519 | 0.502 | 0.592 / 0.900 |

**Table 5. Knee-valgus detection consistency (binarized φ vs GT).**

### 5.3 RQ3: Temporal Stability of Tracking-Free Inference and the Cost of Smoothing

On PoseTrack18 val the two models yield 704 / 728 track segments (161 / 160 videos, median segment length 31 frames). Table 6 gives the overall three metrics after person-scale normalization, plus the AP cost of smoothing.

| Model | Convention | disp_mean | jitter_mean | jump_0.05 | jump_0.10 | AP cost |
|---|---|---|---|---|---|---|
| round5-3 | raw | 0.0409 | 0.0521 | 0.2062 | 0.0726 | — |
| | pixel-domain smoothing | 0.0380 | 0.0509 | 0.1870 | 0.0634 | **+0.0001** |
| | q-domain smoothing | 0.0432 | 0.0528 | 0.2200 | 0.0807 | −0.0013 |
| YOLO11s-pose | raw | 0.0430 | 0.0535 | 0.2153 | 0.0777 | — |
| | pixel-domain smoothing | 0.0399 | 0.0523 | 0.1943 | 0.0679 | **+0.0002** |
| | q-domain smoothing | 0.0462 | 0.0546 | 0.2336 | 0.0878 | −0.0029 |

**Table 6. M3 temporal metrics (overall, averaged over 17 keypoints) and smoothing AP cost (matched-scope AP; raw values: round5-3 0.1316 / YOLO11s-pose 0.1408).**

Three observations. First, **the two models show no significant difference in raw temporal stability**: the bootstrap CIs of all seven metrics (disp/jitter means and P95s, three jump-rate thresholds) include 0. Occlusion-domain training data did not change temporal stability — "occlusion training does not affect temporal behavior" is itself an informative result, suggesting that data-side interventions should be considered separately from temporal post-processing.

Second, **pixel-domain OneEuro smoothing buys stability at zero accuracy cost**: the jump rate (τ = 0.05) drops by 9.3% (round5-3) and 9.8% (YOLO11s-pose) relative, frame-to-frame displacement mean drops by about 7%, and AP changes stay within ±0.0002. Third, **q-domain smoothing backfires**: all three metrics rise (jump_0.05 to 0.2200/0.2336) and AP drops slightly — the GT person box's per-frame scale noise is re-injected into the pixel domain during de-normalization (mechanism in 3.3). This comparison shows that when smoothing tracking-free per-frame outputs, the choice of filtering domain decides the benefit; the protocol therefore fixes the pixel domain as the primary convention.

### 5.4 RQ4: Correlations Among Metrics

On the (dataset × layer) cells (n = 6), sign consistency (Kendall τ) between protocol-metric differences (Δ = YOLO11s-pose − round5-3, with RMSE differences sign-flipped so that positive always means YOLO11s-pose is better) and full-mAP differences: stratified mAP τ = 1.0 (12/12 pairs), elbow-angle RMSE τ = 1.0 (12/12), knee-angle RMSE τ = 0.667 (10/12); extended to 24 joint-level cells, τ = 0.719 (165/192). Spearman correlations: full Δ vs stratified Δ ρ = 0.956 (p = 0.003); vs elbow angle ρ = 0.956 (p = 0.003); vs knee angle ρ = 0.717 (p = 0.109); elbow vs knee ρ = 0.771 (p = 0.072).

Two interpretations (exploratory; n = 6 stated as-is). First, **stratified mAP's ranking information is highly consistent with full mAP**; its protocol value lies not in re-ranking but in magnitude diagnosis — the "how much is masked" that Δ_occ provides is something a scalar mAP cannot give. Second, **the angle metrics carry partially independent information**: knee-angle RMSE's correlation with full AP does not reach significance, and on COCO the directions diverge (round5-3 has better knee angles yet lower full mAP; Table 4) — the angle protocol captures a dimension of difference beyond OKS. For protocol economy: stratified mAP and angle error are not redundant; each captures a different blind spot. The temporal metric has cells from only one dataset and awaits testing on more data.

## 6 Discussion

**The magnitude of the three blind spots is quantified for the first time under one protocol.** The consistent observation across four datasets and two models is: full mAP's overestimation of occluded-scene performance (Δ_occ −0.24 to −0.41), occlusion's amplification of angle error (v=1-layer elbow RMSE up to +18.1°), and the high-frequency jitter of tracking-free per-frame outputs (jump rate ≈ 0.21, reducible by about 10% at zero cost with pixel-domain smoothing). Together these numbers show that model selection driven solely by OKS-mAP can misjudge strength precisely in the scenarios where reliability matters most (occlusion, fast motion) relative to what the stratified evidence shows.

**The boundary between the angle protocol and validity studies.** Our angle metrics use ground-truth-derived angles as the reference and answer "which model's joint angles are closer to the annotated truth" — a model-comparison tool. They do not validate "do estimated angles equal a clinical gold standard", which is the task of the markerless validity literature [4–6] and requires markers or expert annotations. The two are complementary: validity studies certify that a specific model is trustworthy in a specific scenario, while our protocol gives any new model comparable angle-dimension scores on standard datasets. Likewise, the knee-valgus detection consistency (κ up to 0.76) asserts only "consistency with GT annotations", not clinical meaning.

**Engineering implications of the temporal metrics.** M3's two findings directly guide system design: first, data-side interventions (occlusion training) do not change temporal stability, so temporal quality must be handled separately; second, the choice of smoothing domain decides the benefit — normalized-domain filtering looks theoretically correct but fails in practice due to the re-injection of GT-box scale noise. The protocol fixes the pixel domain as the primary convention and recommends that any temporal smoothing method report the joint "jitter before/after + AP cost" table.

**Limitations.** (1) The baseline covers only two models of the same architecture; we do not claim coverage of the model space — the protocol's value is its conventions and tooling, and baseline numbers for more models are left to future work. (2) CrowdPose's v2 layer is affected by the 14→17 mapping truncation; on that dataset, cross-layer comparisons use Δ_occ. (3) Angles are 2D image-plane angles affected by viewpoint; they are not 3D joint angles or clinical angles. (4) M3 covers a single dataset; the generality of the temporal conclusions awaits extension. (5) RQ4's correlation analysis has small cell counts (n = 6) and is exploratory. (6) COCO's v=1 layer is only 7.8% of keypoints, so its v=1-layer estimates have larger variance (wider CIs). The convention behind every number (pycocotools 2.0.11) is stated in the tables.

## 7 Conclusion

We proposed and executed a unified supplementary evaluation protocol for 2D human pose estimation, targeting occlusion, joint angle, and temporal stability, together with an open-source harness and a reproducibility contract. Baseline results on two models and four public datasets show that full OKS-mAP systematically masks occlusion-specific accuracy differences (Δ_occ up to −0.41), that the angle-error protocol captures partially independent information beyond OKS, and that pixel-domain smoothing can reduce temporal jitter at zero accuracy cost. The protocol is positioned as a complement to, not a replacement for, OKS-mAP: it does not change the underlying convention of mainstream rankings, but adds three previously invisible diagnostic dimensions for model selection and system design. The code and all raw predictions are open-sourced.

## References

[1] Lin, T.-Y., et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
[2] Zhang, S.-H., et al. Pose2Seg: Detection Free Human Instance Segmentation (OCHuman). arXiv:1803.10683, 2018.
[3] Andriluka, M., et al. PoseTrack: A Benchmark for Human Pose Estimation and Tracking. CVPR 2018; arXiv:1712.09184, 2017.
[4] Turner, J., Clark, J., Keshtmand, N. Markerless Pose Estimation for Resistance Training Technique Assessment. arXiv:2608.24384, 2026.
[5] Panconi, et al. Deep-Learning-Based Markerless Pose Estimation Systems in Gait Analysis. arXiv:2407.10590, 2024.
[6] Pemasiri, et al. Biomechanically Accurate Gait Analysis. arXiv:2603.02499, 2026.
[7] Li, J., et al. CrowdPose: Efficient Crowded Scenes Pose Estimation and a New Benchmark. arXiv:1812.00324, 2018.
[8] Ghafoor, A., Mahmood, A. Quantification of Occlusion Handling Capability of a 3D Human Pose Estimation Framework. IEEE TMM, arXiv:2203.04113, 2022.
[9] Sárándi, I., Linder, T., Arras, K.O., Leibe, B. How Robust is 3D Human Pose Estimation to Occlusion? IROS 2018 Workshop on Robotic Co-workers 4.0. arXiv:1808.09316, 2018.
[10] Garg, Y., Bachu, S., Dutta, A., Lal, R., Bose, S., Ta, C.-K., Asif, M.S., Roy-Chowdhury, A. VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions. arXiv:2508.06757, 2025.
[11] Mendonça, I.L.A., do Nascimento Filho, F.M., Badicu, G., de Vasconcelos, D.P., Lima, T.B., Calmon, V., de Farias Neto, J.P., Jakubiak, G.K., Maciel, L.Y.S., Aidar, F.J., Monteiro da Silva Júnior, W. Analysis of Ankle Mobility and Dynamic Valgus Knee in Runners With and Without Patellofemoral Pain Syndrome. Health Science Reports, 9, e72051, 2026.
[12] Lei, C., et al. Deep Video Prior for Video Consistency and Propagation. IEEE TPAMI, arXiv:2201.11632, 2022.
[13] Casiez, G., Roussel, N., Vogel, D. 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems. CHI 2012.
[14] Jocher, G., et al. Ultralytics YOLO11. https://github.com/ultralytics/ultralytics, 2024.
[15] Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., Xu, Q. SmoothNet: A Plug-and-Play Network for Refining Human Pose Estimation in Videos. ECCV 2022; arXiv:2112.13715, 2022.
