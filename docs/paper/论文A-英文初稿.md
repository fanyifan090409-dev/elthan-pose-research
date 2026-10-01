# Uncovering the Blind Spots of OKS-mAP: A Unified Protocol for Occlusion, Joint-Angle, and Temporal Evaluation of 2D Pose Estimators

> English draft of Paper A (2026-10-01), translated from the Chinese manuscript. Author: Yifan Fan (International Department, Taiyuan Yangfan School).
> All numbers are locked values from the G3 baseline experiments (summary.json, schema_version=1). The Chinese draft remains the source of truth until author sign-off.

## Abstract

The dominant evaluation protocol for 2D human pose estimation — mean average precision based on Object Keypoint Similarity (OKS-mAP) — is insensitive to three properties that matter for downstream applications: it does not distinguish visible from occluded keypoints in the aggregate score; it does not measure joint-angle error; and it is silent on the temporal stability of outputs produced by tracking-free, frame-by-frame inference. We propose a unified supplementary evaluation protocol with three modules: (M1) OKS-mAP stratified by ground-truth visibility labels (v), together with Δ_occ = mAP(v=1) − mAP(full), which quantifies how much the aggregate metric masks occlusion-specific differences; (M2) joint-angle error (RMSE/MAE/P95) against angles derived from ground-truth keypoints, plus knee-valgus detection consistency (Cohen's κ); and (M3) frame-to-frame displacement, median-filter residual jitter, and jump rate under tracking-free per-frame inference, before and after OneEuro smoothing. We executed the protocol on four public datasets (COCO val2017, CrowdPose, OCHuman, PoseTrack18) for two representative models (the official YOLO11s-pose and a model fine-tuned with occlusion-domain data), with all metrics produced by an open-source harness in a single inference pass. Δ_occ is negative for every model–dataset combination (−0.24 to −0.41): the aggregate mAP overestimates occluded-scene performance in every combination tested. Joint-angle error is amplified in occluded layers across all tested combinations, and on the occlusion-domain datasets the angle metrics reveal significant differences between the two models that the aggregate mAP cannot localize. We found no evidence of a difference in raw temporal jitter between the two models, while pixel-domain OneEuro smoothing reduces the jump rate by 9–10% at zero accuracy cost (|ΔAP| ≤ 0.0002). Correlation analysis shows that stratified mAP is highly consistent with full mAP (sign-consistency τ_s = 1.0, 12/12 pairs) — its value lies in magnitude diagnosis rather than re-ranking — while the angle metrics, particularly knee angle, carry partially independent information (ρ = 0.72, p = 0.109, n = 6, exploratory). The protocol and harness are open-sourced and positioned as a complement to, not a replacement for, OKS-mAP.

**Keywords**: human pose estimation; evaluation protocol; occlusion; joint angle; temporal stability; OKS mAP

## 1 Introduction

2D human pose estimation is a fundamental computer vision task whose outputs feed action recognition, motion analysis, and real-time fitness-correction systems. For such applications, the gap between benchmark scores and real-world usability matters more than average precision alone: a model that ranks first on a benchmark can be markedly worse in scenes with occluded people, fast motion, or the need for precise joint-angle judgments — and this difference is invisible in the dominant metric.

The de facto standard evaluation protocol is OKS-mAP as defined by the COCO keypoints protocol [1]. It matches and scores predictions by Object Keypoint Similarity (OKS) and aggregates over the whole test set into a single scalar. This design has three blind spots. First, OKS aggregation ignores keypoint visibility: visible (v=2) and occluded (v=1) keypoints are averaged with equal weight into one score (missing, v=0, keypoints are excluded from OKS averaging), so occlusion-specific accuracy differences are diluted by averaging. Second, the protocol does not measure joint angles: OKS penalizes pixel-level keypoint position error, whereas downstream applications (e.g., exercise correction) care about whether the joint angles derived from those keypoints are accurate — and the two are not linearly related. Third, the protocol is blind to temporal behavior: most systems run frame-by-frame independent inference on video, and the jitter of the output between frames directly determines how the downstream consumer perceives it, yet a static-image protocol says nothing about it. Prior work responds to each problem in isolation — OCHuman organizes evaluation by occlusion level [2]; PoseTrack and SmoothNet extend the static protocol with tracking metrics (MOTA/MOTP) and a temporal-smoothness metric (per-joint acceleration magnitude, Accel), respectively [3,15]; and the biomechanics community has validated joint angles from markerless pose estimation [4–6]. What is missing is a unified, reproducible, cross-dataset supplementary protocol, and any baseline numbers quantifying how large these three blind spots are on real models.

This paper proposes a three-module unified supplementary evaluation protocol (Section 3) and open-sources its execution layer, an eval harness that produces official metrics and protocol metrics from a single inference pass, with three self-check gates. We executed the protocol on four public datasets with two representative models (Sections 4–5) to answer four research questions:

- **RQ1 (occlusion)**: How much does full OKS-mAP mask occlusion-specific accuracy differences? Does model ranking change after visibility stratification?
- **RQ2 (angle)**: Does joint-angle RMSE provide discriminative information beyond OKS-mAP? Is angle error amplified by occlusion?
- **RQ3 (temporal)**: Under tracking-free per-frame inference, how much do models differ in temporal stability? Can smoothing buy stability at an acceptable accuracy cost?
- **RQ4 (protocol economy)**: How do the metrics of the three modules correlate with each other and with full mAP? Is any of them redundant?

Our contributions are threefold: (1) to our knowledge, the first supplementary evaluation scheme that unifies occlusion stratification, angle error, and temporal jitter under one protocol, with explicit statistical conventions (bootstrap confidence intervals) and dataset-consistent stratification rules; (2) an open-source harness and all raw predictions, so that any new model can produce the full set of protocol metrics on the four datasets in one pass; (3) complete baseline numbers for two representative models that quantify, for the first time to our knowledge, the magnitude of OKS-mAP's three blind spots (e.g., Δ_occ as large as −0.40 on COCO).

## 2 Related Work

**Pose estimation evaluation protocols.** The COCO keypoints protocol, with OKS at its core, has been the field's de facto standard since 2014 [1]. PoseTrack extends evaluation to multi-person pose tracking in video with a tracking convention centered on MOTA/MOTP [3]; SmoothNet goes further and proposes per-joint acceleration magnitude (Accel, in pixels/frame²) as a temporal-smoothness metric for video pose, and uses that convention in its filtering-vs-baseline comparisons [15]. Our M3 is complementary to Accel: Accel measures the size of the frame-to-frame second difference, whereas our frame-to-frame displacement and median-filter residual measure positional fluctuation and high-frequency noise separately, and we additionally report the accuracy (AP) cost of smoothing. On the data side, CrowdPose and OCHuman provide real visibility/occlusion labels (v=1 in CrowdPose; three occlusion levels in OCHuman) [2,7]. The common boundary of this prior work is that each improves a single dataset in a single dimension; none of it forms a cross-dataset supplementary protocol with unified conventions.

**Occlusion-robustness evaluation.** Prior work has attempted to quantify occlusion's effect on pose estimation: Ghafoor and Mahmood build three experiments with synthetically removed joints to quantify occlusion's impact on 3D pose error [8], and later work continues this line with robustness evaluation under synthetic occlusion [9] and a visibility-aware video occlusion benchmark [10]. The difference from our work: these studies use synthetic occlusions or non-real annotation-based stratification, and each focuses on a single dataset and a single metric. We stratify across datasets using real v labels from three datasets and report Δ_occ against full mAP.

**Joint-angle validation.** Biomechanics and sports science have a line of validity studies for markerless pose estimation: Turner et al. reconstruct joint-angle trajectories from BlazePose keypoints in 89 resistance-training videos and assess movement-technique quality by frame-wise RMSE against a reference movement [4]; Panconi et al. validate DeepLabCut joint angles in gait analysis [5]; Pemasiri et al. build a gait-analysis framework for biomechanical interpretation [6]. These studies answer the question "do estimated angles equal the clinical/marker-based gold standard?" (validity), and they require dedicated marker or expert annotations. We do not repeat validity validation; instead we use ground-truth-derived angles as a protocol metric for model-to-model comparison. The two lines are complementary rather than competing (Section 6 elaborates this boundary). The knee-valgus threshold takes its magnitude reference from the literature: frontal plane projection angle (FPPA) of 16.0–17.4° in a lateral step-down test [11].

**Temporal consistency and smoothing.** Temporal inconsistency of tracking-free per-frame inference has been discussed at a general level by Deep Video Prior and others [12]; in engineering practice, OneEuro causal filtering is widely used to smooth keypoint trajectories [13]. Our work turns "stability of tracking-free raw output" itself into a protocol metric, and compares two filtering domains (normalized-coordinate vs. pixel) in their joint effect on jitter metrics and AP — a comparison that reveals how the choice of filtering domain decides the benefit and cost of smoothing.

## 3 Protocol Design

### 3.0 Overview

The protocol consists of three modules plus one statistical layer, all built on the common foundation of the COCO 17-keypoint definition and the OKS convention. Table 1 gives the overview.

| Module | Metrics | Data | Answers |
|---|---|---|---|
| M1 occlusion stratification | v-stratified mAP50-95(P) + Δ_occ | CrowdPose val / OCHuman test / COCO val | RQ1 |
| M2 angle error | elbow/knee angle RMSE/MAE/P95 + knee-valgus detection consistency (κ) | same as M1 (angles derived from GT, zero extra annotation) | RQ2 |
| M3 temporal jitter | frame-to-frame displacement / jitter / jump rate, before and after OneEuro | PoseTrack18 val (tracking-free per-frame inference) | RQ3 |
| Statistical layer | 10k bootstrap CI + sign-consistency τ_s + Spearman ρ | results of the three modules | RQ4, model comparison |

**Table 1. Protocol overview.**

### 3.1 M1: Occlusion-Stratified mAP

**Stratification convention.** Each GT keypoint is assigned to a layer by its visibility label v: v=2 visible, v=1 occluded, v=0 missing. OKS AP is computed independently within each layer. The v=0 layer is reported but excluded from ranking (keypoints in this layer cannot be estimated and scores are near zero; the layer is kept as evidence of protocol completeness). Each dataset retains its real v labels after conversion to COCO17: CrowdPose's official 14 keypoints map to COCO indices 5–16 (the head_top and neck points are dropped — a mapping-induced truncation, not a corruption of visibility labels; the v=1 labels of the 12 mapped points are fully preserved); OCHuman's official test annotations are already in COCO17 format and pass through unchanged; COCO val itself contains v=1 labels (7.75% of keypoints).

**Computation convention.** We follow the keypoints protocol of pycocotools 2.0.11 (the convention behind official released numbers): mean over 10 OKS thresholds 0.50:0.95; GTs with num_keypoints=0 marked ignore; maxDets=20. Stratification changes only the statistics, not the matching logic: person matching (TP/FP decisions) still uses full-keypoint OKS, and a layer's AP accumulates only the OKS contributions of that layer's GT keypoints. Three gates guarantee implementation correctness (agreement with the official library within 0.0001; synthetic self-checks; the stratification toggle leaves full mAP unchanged).

**Interpretability evidence.** A layer's AP has all GT persons as its recall denominator, so persons without that layer's keypoints can never contribute TPs to it. We therefore report, per layer, both "fraction of GT keypoints in this layer" (gt_frac) and "fraction of GT persons possessing this layer's keypoints" (gt_person_frac). In particular, CrowdPose's v2 layer suffers from the 14→17 mapping truncation (gt_person_frac only 0.591), so its v2-layer AP is systematically underestimated — on CrowdPose, cross-layer comparisons between v2 and v1 should use Δ_occ (which is unaffected by this artifact).

**Primary metric.** Δ_occ = mAP(v=1 layer) − mAP(full); a negative value means the full metric overestimates occluded-scene performance. Both terms are computed over the same image universe and the same full GT person set (the v=1 layer's recall denominator is the full GT person set, see Interpretability evidence), so Δ_occ isolates the occlusion layer's contribution gap rather than a change in denominator. Per-layer mAP and gt_frac are reported alongside.

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

**Knee-valgus detection consistency.** The knee-valgus angle φ is defined as the image-plane angle between the hip→knee vector and the hip→ankle vector (how far the knee point deviates from the hip–ankle axis), computed only for instances where all three points have v > 0. φ_pred and φ_GT are binarized at a threshold τ_φ (default τ_φ = 15°, sensitivity analysis τ_φ ∈ {10°, 17°}), and we report detection sensitivity, specificity, and Cohen's κ with GT as the gold standard. Two clarifications: we do not report absolute detection rates (static datasets mix viewpoints, making absolute rates meaningless; consistency is what the protocol can measure); and φ is a 2D image-plane proxy under arbitrary viewpoints, whereas the FPPA literature values come from a standardized frontal camera [11] — our datasets mix viewpoints, so τ_φ takes only a magnitude reference from the literature and carries no clinical-diagnostic meaning.

### 3.3 M3: Temporal Jitter

**Data and matching.** PoseTrack18 val videos are inferred frame by frame independently (no tracking, no cross-frame information); per-frame predictions are matched to GT by Hungarian matching on full-keypoint OKS (same convention as M1). Segments break when the frame gap exceeds 1; only track segments with ≥10 consecutive matched frames are retained. All displacements are normalized by the GT person-box scale s_t = √(w·h) (consistent with the OKS scale convention).

**Three metrics** (computed per keypoint and per track segment, then averaged):

1. frame-to-frame displacement d_t = ‖p_t − p_{t−1}‖₂ / s_t: report mean and P95;
2. jitter j_t = ‖p_t − ŝ_t‖₂ / s_t, where ŝ is the trajectory after a causal 5-frame median filter: targets high-frequency noise (core metric);
3. jump rate: fraction of frames with d_t > τ_j, default τ_j = 0.10 (10% of person-box scale), sensitivity τ_j ∈ {0.05, 0.20}.

**Before/after OneEuro comparison.** A causal OneEuro filter is applied offline to p_t with parameters fixed at Casiez's defaults (min_cutoff = 1.0, beta = 0.007, d_cutoff = 1.0) [13]. The three metrics and the matched-scope AP are recomputed after smoothing, and the accuracy cost is reported. Two filtering domains are used: the normalized-coordinate domain (q domain, the protocol's original design) and the pixel domain. The empirical finding (consistent across both models): filtering in the q domain re-injects the GT person box's per-frame scale noise (|s_t/s_{t−1} − 1|, median 0.97%, P95 8.1%) into the pixel domain during de-normalization, so jitter metrics rise rather than fall and AP drops slightly; direct pixel-domain filtering costs zero AP and lowers all three metrics. **We adopt the pixel domain as the primary convention**, and report the q domain as a sensitivity convention (Table 6, two columns).

### 3.4 Statistical Conventions

- Model comparison: M1/M2 resample at image level and M3 at track-segment level, 10,000 bootstrap replicates (seed 42); we report 95% percentile CIs of the difference Δ and judge significance by whether the CI excludes 0. M1 is a paired design (shared image universe = the union of GT and predicted images of both models); the valid unit sets of M2/M3 differ with matching results, so they use independent resampling, as noted in the protocol.
- The comparison list is fixed before the experiments (2 models × 3 datasets × layers/joints/metrics); every entry is reported with its CI, with no selection of favorable results.
- RQ4 (protocol economy): on (dataset × layer) cells we compute the sign-consistency τ_s between protocol-metric differences and full-mAP differences, and the Spearman ρ between the two difference vectors. τ_s counts concordant pairs among all pairwise cell comparisons, excluding pairs whose two cells share an identical full-mAP difference (a tie, for which sign consistency is undefined): at n = 6 (3 datasets × 2 layers), 15 pairwise comparisons minus the 3 same-dataset layer pairs = 12 reported pairs; at the joint level (n = 24: 3 datasets × 4 joints × 2 layers), 276 pairwise comparisons minus 84 same-dataset same-joint layer pairs (3 × 28) = 192 reported pairs. Both analyses are exploratory, and the sample size is stated in the main text.

### 3.5 Harness and Reproducibility

The execution layer is an open-source eval harness: `model.val(save_json=True)` produces official metrics and prediction JSONs in a single inference pass; the M1/M2/M3 engines are purely offline (NumPy) and do not touch the GPU. Three gates guarantee the conventions are implemented correctly (see 3.1). The reproducibility contract (environment, weights, data, step-by-step commands, convention notes) is in the repository's REPRODUCIBILITY.md. The protocol's JSON contract (schema_version=1) is available for downstream plotting and secondary analysis. The harness and this paper's source are open-sourced at https://github.com/fanyifan090409-dev/pose-eval-harness and https://github.com/fanyifan090409-dev/elthan-pose-research; the training fork (ultralytics-based) is at https://github.com/fanyifan090409-dev/elthan-pose.

## 4 Experimental Setup

**Datasets.** Four public datasets: COCO val2017 (general-domain control; v distribution 55.4%/7.8%/36.8% for v=2/1/0), CrowdPose val (crowded/occlusion domain; v=1 marks real occlusion; distribution 22.5%/30.8%/remainder), OCHuman test (heavy-occlusion domain; distribution 69.2%/14.3%/remainder), PoseTrack18 val (video temporal; 170 videos / 8,923 annotated frames; GT v ∈ {0,1}, where v=0 is a (0,0) placeholder). All datasets are unified under the COCO17 keypoint format. PoseTrack was obtained under MPII's official academic license and is used for academic evaluation only.

**Models.** Two baseline models: (1) the official YOLO11s-pose pretrained weights [14]; (2) round5-3, obtained by fine-tuning YOLO11s-pose on public datasets including COCO, MPII, CrowdPose, and OCHuman, representing training with occlusion-domain data. The two share the same architecture and inference configuration and differ only in training data, making them suitable for testing whether the protocol is sensitive to training-data differences. round5-3's training data is restricted to training splits (COCO train2017, MPII train, CrowdPose train, OCHuman train, plus OpenThermalPose train at controlled proportion); none of the four evaluation splits used in this paper (COCO val2017, CrowdPose val, OCHuman test, PoseTrack18 val) appears in its training data, so the protocol numbers are leak-free.

**Inference and conventions.** 640×640 input, frame-by-frame independent inference, predictions saved with save_json. All mAP numbers use the official released-numbers convention of pycocotools 2.0.11, whose doubled-bbox fallback scores a GT keypoint with vg > 0 but zero keypoint area (k1 = 0, the COCO (0,0) placeholder) using a bbox-derived scale rather than dropping it. Note that this convention can differ from ultralytics' internal evaluator by more than 20 points (sources: GTs with num_keypoints=0 marked ignore, the OKS doubled-bbox fallback, maxDets=20); every table in the paper states its convention. PoseTrack has no official pycocotools reference, so its official numbers are reported only under the internal-evaluator convention (round5-3 0.1085 / YOLO11s-pose 0.1213 mAP50-95(P)). Gate 1 (agreement with the official library) measured ≤ 0.0001 on all six model–dataset cells (exactly 0 on five).

## 5 Results

### 5.1 RQ1: Full mAP Masks Occlusion Differences

Table 3 gives the stratified mAP of the two models on three datasets, together with the layer-representativeness evidence gt_frac and gt_person_frac. Three observations.

| Dataset | Layer | round5-3 | YOLO11s-pose | Δ_occ (r5-3) | Δ_occ (YOLO) | gt_frac | gt_person_frac |
|---|---|---|---|---|---|---|---|
| COCO val | full | 0.5712 | **0.5816** | — | — | 1.000 | 1.000 |
| | v=2 | 0.5637 | **0.5728** | — | — | 0.554 | 0.988 |
| | v=1 | 0.1747 | 0.1719 | −0.3965 | −0.4097 | 0.078 | 0.564 |
| | v=0 | 0 | 0 | — | — | 0.368 | 0.947 |
| CrowdPose val | full | **0.7626** | 0.6938 | — | — | 1.000 | 1.000 |
| | v=2 | **0.4183** | 0.3768 | — | — | 0.225 | 0.591 |
| | v=1 | **0.5183** | 0.4573 | −0.2443 | −0.2366 | 0.308 | 0.776 |
| | v=0 | 0 | 0 | — | — | 0.467 | 1.000 |
| OCHuman test | full | **0.4775** | 0.3379 | — | — | 1.000 | 1.000 |
| | v=2 | **0.4696** | 0.3314 | — | — | 0.692 | 1.000 |
| | v=1 | **0.1819** | 0.0765 | −0.2955 | −0.2614 | 0.143 | 0.736 |
| | v=0 | 0 | 0 | — | — | 0.164 | 0.902 |

**Table 3. M1 stratified mAP50-95(P) (pycocotools 2.0.11 convention). Bold = significantly better (bootstrap CI excludes 0). gt_frac = fraction of GT keypoints in the layer; gt_person_frac = fraction of GT persons possessing at least one keypoint of the layer (both are GT properties, identical for the two models). Δ_occ = mAP(v=1 layer) − mAP(full) on the same full GT person set (Section 3.1); its bootstrap CIs are in Appendix A. — = not defined for this layer. CrowdPose's v2 layer is affected by the 14→17 mapping truncation (gt_person_frac 0.591); cross-layer comparisons use Δ_occ.**

First, **Δ_occ is negative everywhere and large in magnitude**: −0.40 and −0.41 on COCO, −0.24 on CrowdPose, and −0.26 to −0.30 on OCHuman. In every combination tested, full mAP overestimates occluded-scene performance by 24–41 points — a magnitude that can mask substantive differences in model selection that relies on mAP alone.

Second, **stratification changes the strength of model-comparison conclusions**. On general-domain COCO, YOLO11s-pose is significantly better in full and v=2-layer mAP (Δ = +0.0104, 95% CI [0.0059, 0.0155]), but we found no evidence of a v=1-layer difference (Δ = −0.0028, ns) — the significant advantage in the aggregate comes almost entirely from visible keypoints. On the occlusion-domain datasets, round5-3 is significantly better in all three layers (CrowdPose full Δ = −0.0687, OCHuman full Δ = −0.1396, 95% CIs exclude 0), and the gain from occlusion-domain training is equally significant in the v=1 layer (OCHuman v1 Δ = −0.1055).

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

Second, **the angle metrics separate information that full mAP cannot**. On general-domain COCO we found no evidence of an angle difference between the two models (consistent with the 1-point mAP gap); but on the occlusion domains round5-3 is more accurate in the v=1 layer in every tested comparison: CrowdPose elbow 31.6° vs 35.0°, knee 25.9° vs 30.5° (both significant); OCHuman elbow 45.7° vs 57.7°, knee 32.0° vs 39.7° (all significant). Note that on OCHuman the v=2-layer angle difference is also significant, indicating that the benefit of occlusion-domain training is not limited to occluded keypoints themselves.

Third, **knee-valgus detection consistency** (Table 5): round5-3's κ exceeds YOLO11s-pose's on all three datasets at τ_φ=15° (COCO 0.759 vs 0.745; CrowdPose 0.739 vs 0.669; OCHuman 0.602 vs 0.519), and the conclusion is stable under threshold sensitivity analysis (τ_φ ∈ {10°, 17°}). Bootstrap CIs (Appendix A, Table A4) establish the CrowdPose and OCHuman gaps as significant (Δκ −0.069 [−0.098, −0.041] and −0.084 [−0.121, −0.046]), whereas the COCO gap is directionally consistent but not significant (Δκ −0.014 [−0.045, 0.016]). Both models' κ on OCHuman are low (<0.61), suggesting that φ's measurement noise rises under heavy occlusion.

| Dataset | Model | τ_φ=10° κ | τ_φ=15° κ | τ_φ=17° κ | τ_φ=15° sensitivity / specificity |
|---|---|---|---|---|---|
| COCO val | round5-3 | 0.724 | **0.759** | 0.773 | 0.790 / 0.950 |
| | YOLO11s-pose | 0.697 | 0.745 | 0.759 | 0.783 / 0.943 |
| CrowdPose val | round5-3 | 0.711 | **0.739** | 0.731 | 0.785 / 0.937 |
| | YOLO11s-pose | 0.666 | 0.669 | 0.665 | 0.722 / 0.926 |
| OCHuman test | round5-3 | 0.601 | **0.602** | 0.603 | 0.646 / 0.926 |
| | YOLO11s-pose | 0.529 | 0.519 | 0.502 | 0.592 / 0.900 |

**Table 5. Knee-valgus detection consistency (binarized φ vs GT). Bold = higher value (round5-3); bootstrap CIs for κ and for the model difference are in Appendix A.**

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

**Table 6. M3 temporal metrics (overall, averaged over 17 keypoints) and smoothing AP cost. Matched-scope AP is the AP recomputed over the same matched track frames used for the jitter metrics (raw values: round5-3 0.1316 / YOLO11s-pose 0.1408); — = raw reference, no smoothing applied. Bold AP cost = within ±0.0002 of zero (zero accuracy cost).**

Three observations. First, **we found no evidence of a difference in raw temporal stability between the two models**: the bootstrap CIs of all seven metrics (disp/jitter means and P95s, three jump-rate thresholds) include 0. We found no evidence that occlusion-domain training data changed temporal stability — "no evidence that occlusion training affects temporal behavior" is itself an informative result, suggesting that data-side interventions should be considered separately from temporal post-processing.

Second, **pixel-domain OneEuro smoothing buys stability at zero accuracy cost**: the jump rate (τ_j = 0.05) drops by 9.3% (round5-3) and 9.7% (YOLO11s-pose) relative, frame-to-frame displacement mean drops by about 7%, and AP changes stay within ±0.0002. Third, **q-domain smoothing backfires**: all three metrics rise (jump_0.05 to 0.2200/0.2336) and AP drops slightly — the GT person box's per-frame scale noise is re-injected into the pixel domain during de-normalization (mechanism in 3.3). This comparison shows that when smoothing tracking-free per-frame outputs, the choice of filtering domain decides the benefit; the protocol therefore fixes the pixel domain as the primary convention.

### 5.4 RQ4: Correlations Among Metrics

On the (dataset × layer) cells (n = 6), we compute the sign-consistency τ_s between protocol-metric differences (Δ = YOLO11s-pose − round5-3, with RMSE differences sign-flipped so that positive always means YOLO11s-pose is better) and full-mAP differences, counting only the 12 pairs with distinct full-mAP differences (Section 3.4): stratified mAP τ_s = 1.0 (12/12 pairs), elbow-angle RMSE τ_s = 1.0 (12/12), knee-angle RMSE τ_s = 0.667 (10/12); extended to 24 joint-level cells, τ_s = 0.719 (165/192). Spearman ρ between the same difference vectors: full Δ vs stratified Δ, ρ = 0.956 (p = 0.003); full Δ vs elbow angle, ρ = 0.956 (p = 0.003); full Δ vs knee angle, ρ = 0.717 (p = 0.109); elbow vs knee, ρ = 0.771 (p = 0.072). The two identical ρ = 0.956 values are not a transcription error: the stratified-mAP and elbow-angle difference vectors rank the six cells identically, and the only rank differences between either of them and the full-mAP differences occur within the two layers of the same dataset, which share an identical full-mAP difference (a tie), leaving Spearman's numerator unchanged in both cases.

Two interpretations (exploratory; n = 6 stated as-is). First, **stratified mAP's ranking information is perfectly sign-consistent with full mAP (τ_s = 1.0)**; its protocol value lies not in re-ranking but in magnitude diagnosis — the "how much is masked" that Δ_occ provides is something a scalar mAP cannot give. Second, **the angle metrics carry partially independent information**: knee-angle RMSE's correlation with full AP does not reach significance, and on COCO the directions diverge (round5-3 has better knee angles yet lower full mAP; Table 4) — the angle protocol captures a dimension of difference beyond OKS. For protocol economy: stratified mAP and angle error are not redundant; each captures a different blind spot. The temporal metric has cells from only one dataset and awaits testing on more data.

## 6 Discussion

**The magnitude of the three blind spots is quantified, to our knowledge for the first time, under one protocol.** The consistent observation across four datasets and two models is: full mAP's overestimation of occluded-scene performance (Δ_occ −0.24 to −0.41), occlusion's amplification of angle error (v=1-layer elbow RMSE up to +18.1°), and the high-frequency jitter of tracking-free per-frame outputs (jump rate ≈ 0.21, reducible by about 10% at zero cost with pixel-domain smoothing). Together these numbers show that model selection driven solely by OKS-mAP can misjudge the relative strength of the two models tested precisely in the scenarios where reliability matters most (occlusion, fast motion) relative to what the stratified evidence shows.

**The boundary between the angle protocol and validity studies.** Our angle metrics use ground-truth-derived angles as the reference and answer "which model's joint angles are closer to the annotated truth" — a model-comparison tool. They do not validate "do estimated angles equal a clinical gold standard", which is the task of the markerless validity literature [4–6] and requires markers or expert annotations. The two are complementary: validity studies certify that a specific model is trustworthy in a specific scenario, while our protocol gives any new model comparable angle-dimension scores on standard datasets. Likewise, the knee-valgus detection consistency (κ up to 0.76) asserts only "consistency with GT annotations", not clinical meaning.

**Engineering implications of the temporal metrics.** M3's two findings directly guide system design: first, data-side interventions (occlusion training) do not change temporal stability, so temporal quality must be handled separately; second, the choice of smoothing domain decides the benefit — normalized-domain filtering looks theoretically correct but fails in practice due to the re-injection of GT-box scale noise. The protocol fixes the pixel domain as the primary convention and recommends that any temporal smoothing method report the joint "jitter before/after + AP cost" table.

**Limitations.** (1) The baseline covers only two models of the same architecture; we do not claim coverage of the model space — the protocol's value is its conventions and tooling, and baseline numbers for more models are left to future work. (2) CrowdPose's v2 layer is affected by the 14→17 mapping truncation; on that dataset, cross-layer comparisons use Δ_occ. (3) Angles are 2D image-plane angles affected by viewpoint; they are not 3D joint angles or clinical angles. (4) M3 covers a single dataset; the generality of the temporal conclusions awaits extension. (5) RQ4's correlation analysis has small cell counts (n = 6) and is exploratory. (6) COCO's v=1 layer is only 7.8% of keypoints, so its v=1-layer estimates have larger variance (wider CIs). Every table states its evaluation convention: M1/M2 numbers follow pycocotools 2.0.11, while PoseTrack's official-metric numbers (Section 4) are reported under the ultralytics internal-evaluator convention because PoseTrack has no official pycocotools reference.

## 7 Conclusion


## Appendix A: Bootstrap Confidence Intervals

All intervals are 95% percentile bootstrap CIs from 10,000 resamples with seed 42. The resampling unit is the image for M1 and M2 (the unit over which each metric is aggregated before the model comparison) and the track segment for M3. Paired comparisons share the same resample index across the two quantities; between-model comparisons resample the two models' units independently. Sign convention: Δ = YOLO11s-pose − round5-3, so Δ<0 favors round5-3 for AP and κ, and favors YOLO11s-pose for RMSE. * marks intervals that exclude 0. Every entry of the fixed comparison list is tabulated; none is selected.

**Table A1. M1 per-layer AP comparison with bootstrap CIs.** Unit: image (both models share the same image universe; paired resampling).

| Dataset | Layer | AP r5-3 | AP YOLO | Δ AP [95% CI] |
|---|---|---|---|---|
| COCO val | full | 0.5712 | 0.5816 | 0.0104 [0.0059, 0.0155]* |
| COCO val | v2 | 0.5637 | 0.5728 | 0.0091 [0.0048, 0.0144]* |
| COCO val | v1 | 0.1747 | 0.1719 | -0.0028 [-0.0070, 0.0022] |
| COCO val | v0 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| CrowdPose val | full | 0.7626 | 0.6938 | -0.0687 [-0.0754, -0.0616]* |
| CrowdPose val | v2 | 0.4183 | 0.3768 | -0.0415 [-0.0465, -0.0362]* |
| CrowdPose val | v1 | 0.5183 | 0.4573 | -0.0610 [-0.0677, -0.0540]* |
| CrowdPose val | v0 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| OCHuman test | full | 0.4775 | 0.3379 | -0.1396 [-0.1508, -0.1281]* |
| OCHuman test | v2 | 0.4696 | 0.3314 | -0.1383 [-0.1488, -0.1261]* |
| OCHuman test | v1 | 0.1819 | 0.0765 | -0.1055 [-0.1153, -0.0973]* |
| OCHuman test | v0 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |

**Table A2. M1 Δ_occ (mAP(v=1) − mAP(full)) per model, and the model difference, with bootstrap CIs.** Unit: image (paired); ΔΔ_occ is computed element-wise on the paired resample.

| Dataset | Δ_occ r5-3 [95% CI] | Δ_occ YOLO [95% CI] | ΔΔ_occ [95% CI] |
|---|---|---|---|
| COCO val | -0.3965 [-0.4075, -0.3858]* | -0.4097 [-0.4205, -0.3989]* | -0.0132 [-0.0177, -0.0086]* |
| CrowdPose val | -0.2443 [-0.2565, -0.2322]* | -0.2366 [-0.2488, -0.2246]* | 0.0077 [0.0041, 0.0110]* |
| OCHuman test | -0.2955 [-0.3080, -0.2842]* | -0.2614 [-0.2729, -0.2527]* | 0.0341 [0.0237, 0.0429]* |

**Table A3. M2 joint-angle RMSE (°, sides pooled) with bootstrap CIs.** Unit: image, resampled independently per model; last column n_img: round5-3 / YOLO11s-pose. Δ = YOLO11s-pose − round5-3 (negative = YOLO lower error).

| Dataset | Joint | Layer | RMSE r5-3 | RMSE YOLO | Δ RMSE [95% CI] | n_img |
|---|---|---|---|---|---|---|
| COCO val | Left elbow | v=2 | 27.0 | 26.3 | -0.7 [-3.1, 1.7] | 1720 / 1727 |
| COCO val | Left elbow | v=1 | 33.5 | 36.6 | 3.1 [-4.8, 10.9] | 248 / 247 |
| COCO val | Right elbow | v=2 | 24.9 | 24.5 | -0.5 [-2.8, 1.9] | 1732 / 1730 |
| COCO val | Right elbow | v=1 | 37.7 | 34.8 | -2.9 [-9.1, 3.5] | 290 / 292 |
| COCO val | Left knee | v=2 | 20.3 | 20.9 | 0.6 [-2.0, 3.1] | 1320 / 1325 |
| COCO val | Left knee | v=1 | 30.8 | 32.2 | 1.4 [-8.9, 11.3] | 180 / 182 |
| COCO val | Right knee | v=2 | 20.5 | 21.2 | 0.7 [-1.6, 3.2] | 1302 / 1306 |
| COCO val | Right knee | v=1 | 27.7 | 30.3 | 2.6 [-6.9, 12.3] | 202 / 205 |
| CrowdPose val | Left elbow | v=2 | 27.5 | 27.3 | -0.2 [-2.9, 2.6] | 790 / 793 |
| CrowdPose val | Left elbow | v=1 | 31.3 | 35.0 | 3.7 [1.4, 6.1]* | 1319 / 1321 |
| CrowdPose val | Right elbow | v=2 | 24.1 | 24.6 | 0.6 [-2.0, 3.2] | 810 / 811 |
| CrowdPose val | Right elbow | v=1 | 31.9 | 35.0 | 3.1 [0.7, 5.5]* | 1343 / 1338 |
| CrowdPose val | Left knee | v=2 | 21.1 | 21.3 | 0.2 [-3.3, 3.7] | 575 / 575 |
| CrowdPose val | Left knee | v=1 | 25.7 | 30.5 | 4.9 [2.1, 7.7]* | 1085 / 1081 |
| CrowdPose val | Right knee | v=2 | 20.0 | 21.2 | 1.2 [-2.0, 4.4] | 570 / 571 |
| CrowdPose val | Right knee | v=1 | 26.2 | 30.5 | 4.4 [1.4, 7.2]* | 1104 / 1096 |
| OCHuman test | Left elbow | v=2 | 35.6 | 39.7 | 4.2 [1.7, 6.6]* | 1931 / 1951 |
| OCHuman test | Left elbow | v=1 | 45.4 | 56.2 | 10.8 [7.0, 14.7]* | 860 / 871 |
| OCHuman test | Right elbow | v=2 | 35.6 | 39.5 | 3.9 [1.4, 6.4]* | 1935 / 1947 |
| OCHuman test | Right elbow | v=1 | 46.0 | 59.2 | 13.2 [9.2, 17.3]* | 822 / 831 |
| OCHuman test | Left knee | v=2 | 24.4 | 28.0 | 3.6 [1.1, 6.2]* | 1507 / 1514 |
| OCHuman test | Left knee | v=1 | 30.7 | 39.5 | 8.8 [3.7, 14.0]* | 498 / 499 |
| OCHuman test | Right knee | v=2 | 24.2 | 28.5 | 4.3 [1.8, 6.9]* | 1499 / 1503 |
| OCHuman test | Right knee | v=1 | 33.3 | 39.9 | 6.6 [0.9, 12.4]* | 525 / 534 |

**Table A4. Knee-valgus detection consistency (κ, sensitivity, specificity) with bootstrap CIs.** Unit: image (binarized φ instances pooled per image), resampled independently per model. Δ = YOLO11s-pose − round5-3.

| Dataset | τ_φ | κ r5-3 [95% CI] | κ YOLO [95% CI] | Δκ [95% CI] | Δ sens. [95% CI] | Δ spec. [95% CI] |
|---|---|---|---|---|---|---|
| COCO val | 10° | 0.724 [0.702, 0.745] | 0.697 [0.673, 0.719] | -0.027 [-0.058, 0.003] | -0.015 [-0.043, 0.014] | -0.012 [-0.029, 0.005] |
| COCO val | 15° | 0.759 [0.738, 0.780] | 0.745 [0.723, 0.767] | -0.014 [-0.045, 0.016] | -0.006 [-0.039, 0.026] | -0.006 [-0.019, 0.006] |
| COCO val | 17° | 0.773 [0.751, 0.794] | 0.759 [0.737, 0.780] | -0.014 [-0.045, 0.017] | -0.006 [-0.039, 0.027] | -0.006 [-0.017, 0.006] |
| CrowdPose val | 10° | 0.711 [0.692, 0.729] | 0.666 [0.646, 0.685] | -0.045 [-0.073, -0.018]* | -0.029 [-0.052, -0.004]* | -0.017 [-0.032, -0.001]* |
| CrowdPose val | 15° | 0.739 [0.719, 0.757] | 0.669 [0.648, 0.690] | -0.069 [-0.098, -0.041]* | -0.063 [-0.091, -0.034]* | -0.011 [-0.024, 0.002] |
| CrowdPose val | 17° | 0.731 [0.711, 0.750] | 0.665 [0.643, 0.687] | -0.067 [-0.096, -0.037]* | -0.067 [-0.097, -0.037]* | -0.007 [-0.019, 0.005] |
| OCHuman test | 10° | 0.601 [0.577, 0.625] | 0.528 [0.502, 0.554] | -0.073 [-0.108, -0.037]* | -0.041 [-0.072, -0.010]* | -0.030 [-0.051, -0.010]* |
| OCHuman test | 15° | 0.602 [0.576, 0.627] | 0.519 [0.490, 0.546] | -0.084 [-0.121, -0.046]* | -0.054 [-0.090, -0.017]* | -0.026 [-0.041, -0.011]* |
| OCHuman test | 17° | 0.603 [0.576, 0.628] | 0.501 [0.473, 0.530] | -0.101 [-0.140, -0.062]* | -0.071 [-0.110, -0.032]* | -0.026 [-0.039, -0.011]* |

**Table A5. M3 temporal metrics: per-domain point values, within-model domain differences (px−raw, q−raw), and between-model difference on the raw domain, all with bootstrap CIs.** Unit: track segment; domain differences are paired (same resample index), between-model differences resample independently. Δ (model) = YOLO11s-pose − round5-3.

| Metric | raw r5-3 | raw YOLO | Δ raw [95% CI] | px−raw r5-3 [95% CI] | px−raw YOLO [95% CI] | q−raw r5-3 [95% CI] | q−raw YOLO [95% CI] |
|---|---|---|---|---|---|---|---|
| disp_mean | 0.0409 | 0.0430 | 0.0022 [-0.0019, 0.0062] | -0.0029 [-0.0031, -0.0027]* | -0.0031 [-0.0033, -0.0030]* | 0.0023 [0.0015, 0.0031]* | 0.0031 [0.0022, 0.0041]* |
| jitter_mean | 0.0521 | 0.0535 | 0.0014 [-0.0047, 0.0075] | -0.0012 [-0.0013, -0.0011]* | -0.0013 [-0.0013, -0.0012]* | 0.0007 [0.0003, 0.0011]* | 0.0010 [0.0006, 0.0015]* |
| disp_p95 | 0.1194 | 0.1246 | 0.0051 [-0.0081, 0.0183] | -0.0086 [-0.0093, -0.0076]* | -0.0088 [-0.0096, -0.0080]* | 0.0079 [0.0050, 0.0111]* | 0.0103 [0.0070, 0.0140]* |
| jitter_p95 | 0.1669 | 0.1719 | 0.0050 [-0.0177, 0.0275] | -0.0038 [-0.0046, -0.0032]* | -0.0044 [-0.0049, -0.0035]* | 0.0031 [0.0008, 0.0056]* | 0.0043 [0.0021, 0.0073]* |
| jump_0.05 | 0.2062 | 0.2153 | 0.0091 [-0.0145, 0.0326] | -0.0191 [-0.0206, -0.0177]* | -0.0210 [-0.0224, -0.0195]* | 0.0139 [0.0091, 0.0188]* | 0.0183 [0.0132, 0.0235]* |
| jump_0.1 | 0.0726 | 0.0777 | 0.0051 [-0.0086, 0.0190] | -0.0092 [-0.0102, -0.0082]* | -0.0098 [-0.0108, -0.0090]* | 0.0081 [0.0052, 0.0113]* | 0.0100 [0.0067, 0.0135]* |
| jump_0.2 | 0.0152 | 0.0184 | 0.0032 [-0.0015, 0.0079] | -0.0025 [-0.0030, -0.0021]* | -0.0030 [-0.0035, -0.0026]* | 0.0024 [0.0011, 0.0038]* | 0.0030 [0.0014, 0.0047]* |

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
