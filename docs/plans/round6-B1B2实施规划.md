# round6 B1+B2' 实施规划（2026-09-29，提前启动）

> 决策记录（用户 2026-09-29 拍板）：以苹果为主线、安卓真机基准暂缓（iPhone 数据已有：CPU4 89ms / CPU8 90ms / GPU 17.7ms）；剩余数据集任务**保留不删**（#106 OpenThermalPose/Yoga、竖屏素材 #20/#98，用户后续拍摄）；round6 从 11 月提前到现在：先学源码 → 规划 → 直接开改。
> 依据：`docs/plans/yolo源码优化方案.md`（B1-B4 调研成文）。

## 1. 源码现状事实（2026-09-29 侦察结论，训练机）

- py310 环境用 **pip 版 ultralytics 8.4.128**（site-packages），改造仓库实际为 `/home/ubuntu/isef/ultralytics-dev`（round6 专用 clone，8.4.128 逐字节 pristine 起步，`pip install -e` 装进 py310）。
- 关键类：`v8PoseLoss`（utils/loss.py:651）、`KeypointLoss`（:318，OKS 式距离）、`calculate_keypoints_loss`（:776）、head 侧 `Pose.kpts_decode`（head.py:660）。
- 8.4.128 已内置**逐关键点 sigma**（`getattr(model, "kpt_oks_sigmas")`，上游 PR #25656 已合）——B1.1「OKS-sigma 入 loss」从改造降级为「验证 + 自定义 sigma yaml 入口」。
- v 通道现状：`kpt_mask = gt_kpt[...,2] != 0`（二值），v=1/2 等价（官方 #3409 盖章）；kobj 用 BCE 监督 v 通道。
- 权重：box=7.5 / cls=0.5 / dfl=1.5 / **pose=12.0** / kobj=1.0（default.yaml）；训练走 Python API 脚本（scripts/train_pose_*.py）+ tmux。
- 上游空白确认：官方主线至今**无可见性软加权**——B2' 是真实空缺。
- **开关传递机制（2026-09-29 实锤）**：hyp 键必须进 `cfg/default.yaml`（`get_cfg` 会静默丢弃 unknown overrides）；`detect/train.py:147` 训练前 `model.args = self.args`，criterion 首次 forward 懒构建（tasks.py:409）→ hyp 必为含开关的最终 args。

## 2. 改造接口契约

### B2' 可见性加权（第一个改，~5 行）
- 落点：`calculate_keypoints_loss`，把二值 `kpt_mask` 换成软权重 `w_v = {v==2: 1.0, v==1: 0.5, v==0: 0}`（遮挡点半权重不弃用）；kobj BCE 不动。
- 回退：git tag `round6-base` 一键切回。

### B1.1 OKS-sigma（零改动或一行）
- 验证 8.4.128 已按 OKS_SIGMA 逐点计算；消融组用 data.yaml 加 `kpt_oks_sigmas` 自定义值做对照。

### B1.2 骨长+方向一致性（主自研项）
- 新增独立模块 `utils/bone_loss.py`：`BoneConsistencyLoss(nn.Module)`。
- 骨段集 E：COCO17 连接（左右肩-肘、肘-腕、髋-膝、膝-踝、肩-髋、双肩连线、双髋连线），躯干长归一化去尺度歧义。
- 公式（arXiv 2212.02163）：`L_bone = Σ_{e∈E} λ_e (‖ê−e‖² + (1−cos θ_e))`，ê=预测骨向量、e=GT 骨向量；两端点任一 v==0 则该骨段乘 0。
- 接口：`forward(pred_kpts[bs,17,3], gt_kpts[bs,17,3]) -> scalar`；调用点在 v8PoseLoss.loss() 末尾并入（乘不确定性权重）。

### B1.3 对称性（小项）
- 左右肩/髋/膝/踝 y 坐标高度差软惩罚（L2，小权重）。

### B1.4 角度合理性（保守弱权重）
- 膝/肘三点弯曲角超出 [0°,180°) 惩罚；深蹲膝角可选高斯先验（康复综述警示：markerless 角度未达临床精度，权重必须保守）。

### B1.5 不确定性加权（arXiv 2408.07985，自写 ~15 行）
- 解析解：L2 型项 σ*=√L；温度 softmax 归一 `ω_j = exp((1/sg[L_j])/T) / Σ exp(...)`，stopgradient。骨长/对称/角度三项自动配权，免手工网格。

### B1.6 「手工 vs 可学习」对照（防 SEAL-pose 反方）
- 手工版 = 批内统计先验；可学习版 = 每骨段可训练 (μ,σ) 高斯先验负对数似然。round6 消融必做此对照。

## 3. 消融矩阵（round6）

基线（原版 8.4.128，round5-2 init）→ +B2' → +B1.2 → +B1.3 → +B1.4 → +B1.5 → 手工 vs 可学习。
同数据同配方（30 epoch、有效 lr 0.005、seed 42、batch 128/卡）；主指标 CrowdPose val / OCHuman test，COCO val 不许掉 >0.3。

## 4. 环境与回滚

- 改造仓库：`/home/ubuntu/isef/ultralytics-dev`（clone 自 8.4.128），`pip install -e` 装进 py310。
- 回滚点：每步一个 git commit + 功能开关（新增项默认关，靠 default.yaml 的 hyp 键打开，保证任意一步可独立回退）；旧 wheel 有 `ultralytics-8.4.128.tar.gz` 兜底。
- 门禁：每步后跑单测 + 冒烟脚本（真实训练路径合成 batch 前向/反向），不过不提交。
- 已打 tag：round6-base（ec36986）→ b2-vis-weight（29437d6）→ b1.2-bone（9678bf5）→ b1.3-symmetry（394c66c）→ b1.4-angle（0232c4c）→ b1.5-uncertainty（935cf8c）→ b1.6-prior-contrast。

## 5. 实施顺序与进度（2026-09-29）

0 env 准备（editable + tag）→ 1 B2' → 2 B1.1 验证 → 3 B1.2 骨长 → 4 B1.3 对称 → 5 B1.4 角度 → 6 B1.5 不确定性 → 7 B1.6 对照 → 8 消融训练矩阵。

- [x] 0 env：ultralytics-dev + editable + round6-base tag
- [x] 1 B2'：calculate_keypoints_loss 软权重（v==2→1.0 / v==1→0.5 / v==0→0），kobj BCE 不动；单测 test_b2_vis_weight.py
- [x] 2 B1.1 验证：8.4.128 已逐点 OKS sigma；自定义入口 = data.yaml `kpt_oks_sigmas`（pose/train.py:91 已验证）
- [x] 3 B1.2 骨长：bone_loss.py `BoneConsistencyLoss`（长度+方向、躯干归一化、v 乘积权重、退化 GT 跳过）
- [x] 4 B1.3 对称：`SymmetryLoss` + 解剖先验统一框架（ModuleDict + 单次对齐 + loss_items 键整轮稳定）
- [x] 5 B1.4 角度：`AnglePlausibilityLoss`（肘 10°/膝 20° 铰链）；**键名 angle_prior**——default.yaml:112 `angle: 1.0` 是 OBB 定向角增益，撞键会让基线静默开启本项
- [x] 6 B1.5 不确定性：`uncertainty_weights`（ω=softmax((1/L)/T)，stopgradient）；手工权重退化为开/关门；**T 是量级参数**（冒烟中 1/L 跨 4 个数量级，T=1 饱和，消融时按首 epoch loss 量级定）
- [x] 7 B1.6 对照：手工版 `BoneStatsLoss`（批内统计 μ_j）+ 可学习版 `LearnableBonePrior`（每骨段可训练 μ/σ 高斯 NLL）；消融二选一开启
- [ ] 8 消融训练矩阵（见 §3；每步单测+冒烟全绿后已逐 commit 提交，代码侧就绪）

## 6. 保留事项（不删，用户明确要求）

- #106 OpenThermalPose v1+v2 下载归档、Yoga 评估。
- #20/#98 竖屏素材采集（用户后续拍摄；用途：竖屏验证集、W4 内测错误帧回流、动作集演示视频）。
