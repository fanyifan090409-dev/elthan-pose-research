# 论文 A G2：eval harness 开发规划

> 2026-09-30 成文。上游契约：docs/plans/论文A-G1评估协议.md（同日用户确认定稿）。
> 实施窗口 10/4–10/6，本规划提前成文；开工前如 G1 契约有改动，先改本文再动码。

## 0. 架构决策（一次说清）

**单次推断双算架构**：每个 (模型, 数据集) 只跑一次 `model.val(save_json=True)` → 同时产出 ①官方全量指标 ②预测 JSON。M1/M2 指标引擎纯离线读「预测 JSON + GT 标注」计算，不再碰 GPU。

理由：
- 门禁 1（复现官方 mAP ≤0.5 点）拿同一次运行的两个数直接对拍，零歧义；
- 门禁 3（分层不改变匹配逻辑）用「同一份预测 JSON：官方全量 vs 自算全量」对拍；
- 指标引擎是纯 NumPy 代码 → 可单测、可合成桩自检（门禁 2）。

**不做的事**：不改 ultralytics-dev 源码（训练 repo 零污染）；不新增依赖——numpy 2.2.6 / scipy 1.15.3（Hungarian）/ pandas / matplotlib / pycocotools / ultralytics 8.4.128 已全在 py310（2026-09-30 已核验）。

## 1. 目录结构（训练机 `~/isef/eval-harness/`，独立 git repo）

```
eval-harness/
├── README.md            # 用法：一条命令跑全指标
├── AGENTS.md            # 本目录约定 + 与 G1 契约的对应关系
├── configs/
│   ├── models.yaml      # round5-3 best.pt / yolo11s-pose.pt 路径（T1 核对存在）
│   └── datasets.yaml    # coco-val / crowdpose-val / ochuman-test / posetrack-val data.yaml 路径
├── harness/
│   ├── run_infer.py     # T2 推断层：val(save_json=True) → official.json + preds.json
│   ├── stratified_map.py# T3 M1：OKS 匹配复刻 + v 分层 AP 累积
│   ├── angles.py        # T4 M2：关节角 / 有效性掩码 / 代偿检出一致性
│   ├── temporal.py      # T5 M3：轨迹提取 / d_t / j_t / 跳变率
│   ├── oneeuro.py       # T5 因果 OneEuro（Casiez 默认参数）
│   ├── statistics.py    # T6 bootstrap CI / Spearman / Kendall
│   └── report.py        # T7 JSON 落盘 + md 表格 + 三图
├── tests/
│   ├── test_gates.py    # T8 门禁 2 合成自检 + 门禁 1/3 对拍入口
│   └── fixtures/        # 小型合成 GT / preds JSON
└── runs/                # 每次运行输出（gitignore）
```

## 2. 数据模型（输出 JSON schema = G6 作图契约）

```json
{
  "model": "round5-3", "dataset": "crowdpose-val",
  "official": {"map50_95": 0.0, "map50": 0.0},
  "m1": {
    "full": 0.0,
    "layers": {"v2": {"ap": 0.0, "gt_frac": 0.0}, "v1": {"ap": 0.0, "gt_frac": 0.0}, "v0": {"ap": 0.0, "gt_frac": 0.0}},
    "delta_occ": 0.0
  },
  "m2": {
    "joints": {"13": {"rmse": 0.0, "mae": 0.0, "p95": 0.0, "n": 0, "by_v": {"v2": {"rmse": 0.0}, "v1": {"rmse": 0.0}}}},
    "compensation": {"tau15": {"sens": 0.0, "spec": 0.0, "kappa": 0.0}, "tau10": {}, "tau17": {}}
  },
  "m3": {"per_kpt": {"raw": {"d_mean": 0.0, "j_mean": 0.0}, "smoothed": {}}, "jump_rate": {"tau0.10": 0.0}, "map_smoothed": 0.0},
  "stats": {"bootstrap_ci": {}, "corr": {}}
}
```

关节键用 COCO17 索引 7/8/13/14；样本量 n 与层占比 gt_frac 必须随每个数一起报（协议要求）。

## 3. 任务清单

| # | 任务 | 内容 | 依赖 | 预估 |
|---|---|---|---|---|
| T1 | 脚手架 | 目录 + git init + AGENTS.md + configs（核对两权重路径、三个 data.yaml 存在）+ 空模块骨架 + 首次提交 | — | 0.5d |
| T2 | 推断层 | run_infer.py：val(save_json=True, imgsz=640) → official.json + preds.json；修 CrowdPose data.yaml 的 names 历史遗留（7 类 → person，不影响数值但影响报告可读性）；COCO val 冒烟跑通；双卡参数留给 M3 | T1 | 0.5d |
| T3 | M1 引擎 | 先精读 ultralytics metrics.py（8.4.128）确认 OKS sigmas / v=0 处理 / 匹配顺序 → 逐行对照复刻 → v 分层累积（匹配逻辑不动，只分层统计）→ Δ_occ | T2 | 1d |
| T4 | M2 引擎 | 四关节内角 + 有效性掩码（GT v>0 且 pred conf>0）+ RMSE/MAE/P95 + 顶点 v 分层 + 误差直方图数据 + φ 二值化一致性（τ∈{10,15,17}，灵敏度/特异度/κ 自实现） | T2 | 0.5-1d |
| T5 | M3 引擎 | PoseTrack GT 转 COCO17（15→17 共有关节映射，M3 只报可映射关节）+ 逐帧匈牙利匹配轨迹（≥10 帧段）+ d_t/j_t/跳变率（person-scale 归一化）+ 因果 OneEuro 前后重算 + 平滑后 mAP 变化 | T2 + G4 数据落地 | 1-1.5d |
| T6 | 统计层 | bootstrap 10k 百分位 CI（图像级/轨迹级）、Spearman、Kendall、预注册比较清单固定输出 | T3, T4（M3 就绪后并入） | 0.5d |
| T7 | 报告层 | JSON 落盘（schema 见 §2）+ md 表格 + 三图（分层 mAP 柱状 / 角度误差分布 / 抖动时序曲线示例） | T6 | 0.5d |
| T8 | 门禁 + 单测 | 门禁 2 合成自检（GT 角度代预测→RMSE=0；常数轨迹→抖动=0、跳变率=0）+ 门禁 1/3 对拍（COCO val 双算）→ 全绿宣布 G2 完成 | T3-T7 | 0.5-1d |

衔接（属 G3，不占 G2 工期）：2 模型 × 3 数据集 M1/M2 全量跑 + PoseTrack M3 全量跑，10/7 起。

依赖链：T1 → T2 → {T3 ∥ T4} → T6 → T7 → T8；T5 与 T3/T4 并行，阻塞点只在 PoseTrack 数据（G4：rsync → 拼包解压 → GT 转换）。

## 4. 门禁（G1 §6，T8 执行）

1. 两模型全量 mAP50-95(P) 官方 vs harness ≤0.5 点；
2. 自检：GT 角度代 GT 预测 → RMSE=0；常数轨迹 → 抖动=0、跳变率=0；
3. 分层实现不改变匹配逻辑（同模型全量 mAP 不因开启分层而变）。

## 5. 风险与回滚

- **OKS 复刻偏差**（最大风险）：T3 先读源码再写码，逐行对照；门禁 3 兜底。若偏差 >0.5 点，回 T3 不进入 T6。
- **PoseTrack 15→17 映射**：v0.45 为 15 关节，M3 只报告共有关节并在论文声明；官方标注格式若有变（如新版 17 点），以落地数据实际为准改映射。
- **M3 推理量**：PoseTrack val 逐帧推理量未实测，可能需双卡分片（用户已同意双卡）；run_infer 预留 device 参数。
- **M2 代偿指标样本量**：φ 要求髋/膝/踝三点全可见，静态数据集样本可能少——n 过小（<100）时报告标注「样本不足，仅参考」（同 M1 占比 <1% 口径）。
- 回滚点：T1–T8 每任务一个 conventional commit；repo 独立于 ultralytics-dev 与 scripts/ 旧脚本，无相互污染。

## 6. 时间窗口

10/4 T1-T2 → 10/5 T3-T5（T5 若 PoseTrack 未就绪则顺延到数据落地当天）→ 10/6 T6-T8 → 10/7 起 G3 基准实验。
