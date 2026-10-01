# 论文 A G5：文献检索记录与 novelty 边界

> 2026-09-30 第一轮检索（Giiisp searchArxiv 多字段接口，免鉴权，绕过系统代理）。
> 状态口径：元数据层面「已核验」=题名/作者/年份与已知事实一致；「待核验」=需读 PDF 确认细节。
> 检索词与命中仅记录有效轮次，无效轮次（OneEuro 词命中 ADC 噪声等无关文献）不列。

## 一、候选论文清单

| # | 论文 | 作者 | 年份 | arXiv | 与协议的关系 | 状态 |
|---|---|---|---|---|---|---|
| 1 | PoseTrack: A Benchmark for Human Pose Estimation and Tracking | Andriluka et al. | 2017 | 1710.10000 | **M3 时序：jitter 指标原始出处**（追踪评估），需读原文确认公式 | 已核验 |
| 2 | Pose2Seg: Detection Free Human Instance Segmentation（OCHuman） | Zhang et al. | 2018 | 1803.10683 | **M1 遮挡分层：按遮挡级别分层评测先例**，需读协议细节 | 已核验 |
| 3 | CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark | Li et al. | 2018 | 1812.00324 | M1 数据源（v=1 遮挡标记）；其评估也报告遮挡关节子集 | 已核验 |
| 4 | Quantification of Occlusion Handling Capability of a 3D Human Pose Estimation Framework | Ghafoor, Mahmood | 2022 | 2203.04113 | **M1 遮挡量化先例**：合成缺失关节三协议 + MPJPE，3D 域、非真实标注分层。IEEE TMM | 待核验 |
| 5 | Markerless Pose Estimation for Resistance Training Technique Assessment | Turner et al. | 2026 | 2608.24384 | **M2 角度：健身域 markerless 技术评估先例**（与产品域最近） | 待核验 |
| 6 | Deep-Learning-Based Markerless Pose Estimation Systems in Gait Analysis: DeepLabCut Custom Training and the Refinement Function | Panconi et al. | 2024 | 2407.10590 | M2 角度：markerless 步态分析关节角验证先例 | 待核验 |
| 7 | Biomechanically Accurate Gait Analysis（3D 重建框架） | Pemasiri et al. | 2026 | 2603.02499 | M2 邻域：生物力学解释性步态框架 | 待核验 |
| 8 | Deep Video Prior for Video Consistency and Propagation | Lei et al. | 2022 | 2201.11632 | M3 邻域：逐帧独立图像算法的时序不一致性框架（TPAMI） | 待核验 |
| 9 | Towards Accurate Human Pose Estimation in Videos of Crowded Scenes | Yuan et al. | 2020 | 2010.10008 | M1/M3 邻域：拥挤视频（ACM MM Grand Challenge） | 待核验 |
| 10 | Hand Visibility Detector: Per-Keypoint Visibility Estimation for Hands | Hara et al. | 2026 | 2608.11574 | 邻域：逐关键点可见性估计（可见性作可靠性指标，非评估协议） | 待核验 |

另有两处不依赖本接口的引用锚点（训练知识，写作时补 PDF）：
- **1€ Filter（OneEuro）**：Casiez, Roussel, Vogel, CHI 2012——M3 平滑滤波引用；
- **COCO keypoints 评估协议**：Lin et al., ECCV 2014——OKS mAP 出处，M1 对照口径。

## 二、novelty 边界逐模块评估

| 模块 | 最接近先例 | 先例做什么 | 我们的差异 | 声明口径 |
|---|---|---|---|---|
| M1 遮挡分层 | OCHuman 按遮挡级别评测（#2）；Ghafoor 合成缺失关节量化（#4） | 单数据集分层 / 合成遮挡 3D 误差 | 2D OKS-mAP 按 **GT 真实 v 标注**跨三数据集统一分层 + 与全量 mAP 的 Δ_occ 对照 | **不声称首创分层**，声称「统一口径 + 跨数据集 + 开源 harness」 |
| M2 角度误差 | 生物力学 markerless 验证线（#5/#6/#7） | 估测角 vs 标记点/专家金标准（效度研究） | 模型间比较协议：GT 标注推导角度作金标准、按顶点可见性分层、代偿检出一致性（κ） | 明确「GT 推导 ≠ 临床金标准」的局限；新颖点=角度进入模型评估协议并与遮挡联动 |
| M3 时序抖动 | PoseTrack jitter（#1）；Deep Video Prior（#8） | 追踪质量评估 / 通用时序一致性 | **无追踪逐帧独立推理**的抖动（测模型原始输出稳定性）+ OneEuro 前后与 mAP 联动 | 不声称首创抖动概念，声称「平滑前原始稳定性 + 精度-稳定性权衡」量化 |
| 统一协议 | **未检索到「遮挡分层+角度+时序」三合一先例** | — | 三模块统一 + 开源 harness 一次跑全 | 贡献 1 维持「统一补充协议」口径（如 G5 深读有新发现则再收窄） |

## 三、对论文 A 的影响

1. 贡献声明维持三件套但措辞收紧：M1 措辞从「分层评测」重心移到「跨数据集统一口径 + Δ_occ 量化盲区」；Related Work 必须各引 #1/#2/#4 与生物力学验证线。
2. M2 的分界写法：与 markerless 效度研究的区别 = 我们不验证「估测角度是否等于临床真值」，而是「角度作为模型区分度的评估指标」——GT 推导角度是模型比较的工具而非金标准，讨论章节明确写。
3. M3 需深读 PoseTrack jitter 原文公式（#1），决定「沿用其定义 vs 自定义中值滤波抖动」并在文中说明理由；倾向自定义（我们测的是无追踪原始输出，PoseTrack 测的是追踪轨迹平滑偏差）。
4. G1 协议文档的开放验证项 G5 一条关闭；「Mendonça FPPA 原文复核」与「OCHuman v 三级映射核验」仍开（G4 数据盘点时做）。

## 四、下一步（G5 第二轮，写作前）

- 读 #1 PoseTrack jitter 公式、#2 OCHuman 分层评测细节、#4 Ghafoor 三协议、#5 Turner 技术评估指标（PDF 均可用 giiisp filestore 或 arXiv 镜像取）；
- 补充检索一次「unified pose evaluation protocol」变体词确认无统一协议先例；
- 写作时补 COCO/OneEuro/Janisch（OpenPose CCC=0.95，问卷线已有）引用的题录。

---

# G5 第二轮（2026-10-01）

> G3 基准实验同日收官，G4 PoseTrack 官方表单用户已提交 → G5 第二轮启动。
> 方法：Giiisp searchArxiv 多字段接口（免鉴权）九组检索词，pageSize=10，全部 code 200；
> 命中 JSON 存档 Mac `~/isef-data/g5-search/`。状态口径同第一轮。

## 一、九组检索词与结论摘要

| 检索词（简） | 结论 |
|---|---|
| occlusion stratified keypoint evaluation visibility mAP | 无 v 分层评估协议先例；命中均为遮挡处理方法/可见性预测，非评估协议 |
| OKS keypoint similarity limitations evaluation | **无评估侧 OKS 批判论文**；YOLO-Pose 2204.06806 / ER-Pose 2603.08681 是训练侧 OKS loss（动机段落可用） |
| joint angle error pose estimation validation markerless | 命中健身域近邻 #5（2608.24384，已第一轮收录）+ AthletePose3D、DeepLabCut 步态线 |
| temporal jitter keypoint tracking stability | 无抖动评估协议先例；命中全是追踪方法（Detect-and-Track、APT 系） |
| unified pose evaluation protocol occlusion angle temporal | **无三合一统一协议先例**（唯一「UniEval」是 LLM 多模态评估，无关）→ novelty 边界维持 |
| human pose estimation evaluation protocol benchmark survey | 四篇 survey 均无分层+角度+时序组合协议章节 |
| OCHuman occlusion benchmark evaluation | 复核 OCHuman 1803.10683；新见 **1808.09316 How Robust is 3D HPE to Occlusion?**（遮挡鲁棒性评测，3D） |
| CrowdPose crowded benchmark | 复核 CrowdPose 1812.00324；新见 JRDB-Pose 2210.11940（拥挤多模态基准） |
| keypoint jitter smoothing OneEuro evaluation | 无平滑-抖动评估先例；新见 **2608.03516 Detecting Pose Estimation Failures via Keypoint Self-Consistency**（失败检测邻域） |

## 二、新增候选论文

| # | 论文 | 年份 | arXiv | 与协议的关系 | 状态 |
|---|---|---|---|---|---|
| 11 | How Robust is 3D Human Pose Estimation to Occlusion? | 2018 | 1808.09316 | M1 邻域：遮挡鲁棒性评测（3D、合成遮挡） | 待核验 |
| 12 | Benchmarking Fish Dataset and Evaluation Metric in Keypoint Detection (FishPhenoKey, PMP 指标) | 2024 | 2405.12476 | OKS 盲区叙事佐证：域专用关键点指标先例（鱼形态） | 待核验 |
| 13 | APTv2: Benchmarking Animal Pose Estimation and Tracking | 2023 | 2312.15612 | M3 邻域：视频追踪基准（追踪指标系，无抖动指标） | 待核验 |
| 14 | Detecting Pose Estimation Failures via Keypoint Self-Consistency | 2026 | 2608.03516 | M3 邻域：关键点自一致性失败检测（单帧内，非帧间） | 待核验 |
| 15 | AthletePose3D: A Benchmark Dataset for 3D HPE and Kinematic Validation in Athletic Movements | 2025 | 2503.07499 | M2 邻域：运动员运动学验证（3D） | 待核验 |
| 16 | VOccl3D: A Video Benchmark for 3D HPE under real Occlusions | 2025 | 2508.06757 | M1 邻域：真实遮挡视频基准（3D） | 待核验 |

## 三、第二轮裁决

1. **统一协议 novelty 边界维持成立**：九组检索无一命中「遮挡分层 + 角度误差 + 时序抖动」三合一评估协议。贡献 1 口径不变（统一跨数据集补充协议 + 开源 harness）。
2. **M1 声明不变**：分层单点先例仍为 OCHuman 协议（1803.10683）；新增 3D 遮挡鲁棒性评测线（#11/#16）进 Related Work，与 2D 真实 v 标注分层划界。
3. **M2 声明不变**：#5（2608.24384 阻力训练 BlazePose 角度 RMSE）确认为最近先例——它验证「框架能否恢复动作模式」且单视频、无分层、无代偿检出；我们做「角度作为模型间区分度指标」的跨数据集协议。Related Work 必引。
4. **M3 声明不变**：抖动稳定性评估无先例；PoseTrack 原文（#1）指标是追踪质量（MOTA 系），与「无追踪逐帧推理的原始输出稳定性」划界成立。#13/#14 列为邻域。
5. **G5 收敛**：novelty 边界核验关闭。剩余写作前深读清单：#1 jitter 公式、#2 OCHuman 分层细节、#5 评估指标细节（三项 PDF 深读留给 G6 写作期做，不影响 10/11 开工）。
