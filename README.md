# elthan-pose-research

AI 健身姿态矫正研究实验的全流程记录：数据收集 → 格式转换 → 训练轮次 → 消融 → 端侧部署 → 评估协议。
配套代码仓库：[pose-eval-harness](https://github.com/fanyifan090409-dev/pose-eval-harness)（论文 A 评测 harness）、
[elthan-pose](https://github.com/fanyifan090409-dev/elthan-pose)（论文 B：解剖先验 loss 源码级改动）。

## 目录结构

| 目录 | 内容 |
|---|---|
| `lab-notebook/` | 逐日实验日志（学习笔记 + 实验记录），文件名为 `YYYY-MM-DD-` 前缀 + 描述 |
| `scripts/` | 全部实验脚本：数据集下载/筛选/转换、训练、评测、NCNN 导出、可视化 |
| `datasets/*/data.yaml` | 每轮训练/评测的数据集配置（train/val 路径与类别定义） |
| `skills/ml-training-pitfalls/` | 训练全流程避坑知识库（5 阶段 54 坑 + 2 脚本），本实验实战提炼 |
| `docs/plans/` | 研究方案文档：论文 A（评估协议）/ 论文 B（B1-B4 解剖先验）/ G1-G5 各阶段 |
| `docs/` | 文献数据汇总、已知问题与风险清单、开发者工具踩坑记录 |

## 训练轮次谱系（内部 val 口径，Pose mAP50-95）

| 轮次 | 模型 | 数据 | 结果 |
|---|---|---|---|
| round1-3 | YOLO11n-pose | COCO 15k 采样 | 0.394 |
| round2-2 | YOLO11n-pose | COCO 50k，双卡 DDP | 0.429 |
| round3 | YOLO11s-pose | 全量 COCO + MPII | 0.502 |
| round4 | YOLO11s-pose | + CrowdPose（遮挡域） | 0.510 |
| round5（弃用） | YOLO11s-pose | lr 翻倍实验 | 0.472（崩盘） |
| round5-2 | YOLO11s-pose | + OCHuman + OTP | 0.507 |
| **round5-3（最终）** | YOLO11s-pose | 同上，续训 30 epoch | **0.50532** |

LOO 消融（round5 数据）：去 CrowdPose 0.50546 / 去 OCHuman 0.50234 / 去 OTP 0.50533。
过拟合自检：100 图子集 → 0.919（验证管线可收敛）。

round6（B1 解剖先验 loss + B2' 可见性加权）：内部 val Box mAP50-95 全部落在 0.609–0.613 噪声级；
外部评测 CrowdPose / OCHuman 无一变体超过 baseline（0.53601 / 0.39911）——作为负面结果写入论文 B。
详细见 `docs/plans/round6-B1B2实施规划.md` 与 `lab-notebook/2026-09-30-round6矩阵收官.md`。

## 数据集清单

| 数据集 | 来源 | 用途 |
|---|---|---|
| COCO2017 (person keypoints) | cocodataset.org | 主体训练/验证 |
| MPII Human Pose | mpi-inf.mpg.de | 训练 |
| CrowdPose | crowdpose benchmark | 遮挡域训练 + 外部评测 |
| OCHuman | 遮挡人体基准 | 遮挡训练 + 外部评测 |
| OpenThermalPose (OTP) | 热成像姿态 | 域泛化训练 |
| PoseTrack18 | 时序姿态 | 论文 A 时序抖动指标（M3） |

转换与筛选规则见 `scripts/`（fetch_web_data/、screen_ochuman.py、screen_otp.py 等），评测 GT 转换见 eval-harness 仓库 `harness/build_gt17.py`。

## 关键脚本索引

- 训练：`scripts/train_pose_round5.py`、`train_pose_ablation*.py`、`round6_run.sh`、`round6_ablation_row.py`
- 评测：`scripts/bench_pose_*.py`（各轮外部评测）、`eval_ncnn_ap*.py`（端侧 AP 对拍）
- 导出：`scripts/export_pose_ncnn*.py`、`cut_pose_onnx*.py`
- 数据：`scripts/dl-datasets.sh`、`fetch_web_data/*`、`make_overfit_subset.py`

## 可复现性

- 评估协议全链路复现契约：见 [pose-eval-harness/REPRODUCIBILITY.md](https://github.com/fanyifan090409-dev/pose-eval-harness)（环境/权重/数据/步骤/口径/归档）。
- 训练权重：pose-eval-harness 的 GitHub Releases 提供 round5-3 best.pt；官方 yolo11s-pose 见 ultralytics assets release。
- 原始数据与全部产物双份归档于外接硬盘（本仓库不含原始数据与 runs 产物）。
- 论文状态：论文 A（评估协议）2026-10 月 arXiv；论文 B（解剖先验）2026-12 月。

## 许可

实验脚本与文档：MIT。数据集版权归各原始发布方，请遵循各自许可。
