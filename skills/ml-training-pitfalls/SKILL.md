---
name: ml-training-pitfalls
description: 训练模型全流程避坑知识库（数据→训练→评估→部署→归档）。当用户要微调/训练 YOLO 或视觉/姿态模型、规划训练实验、排查训练崩溃/指标异常/评估口径/导出部署问题、或需要可复现性清单与归档校验时使用。含 50+ 个实战坑（症状→根因→修法→出处），源自 YOLO pose 训练全流程实战。
when_to_use: 用户提到「训练模型」「微调」「训一轮」「消融」「评估指标对不上」「导出部署」「量化」「可复现」「归档」等场景时
---

# ML 训练全流程避坑

不是教程，是坑位地图。每个坑按 **症状 → 根因 → 修法 → 出处** 四段记录，全部来自 YOLO pose 训练实战（数据轮 1-5 + 消融 + round6 源码改动 + 端侧部署 + 评估协议），并按 GitHub 公开资料融合补充。

## 使用方式

1. 开新训练任务前：过一遍下方对应阶段的「硬门禁」。
2. 出问题排查时：按阶段读 `references/` 对应文件，先对症状再查根因。
3. 收尾归档时：执行 `scripts/verify_rsync_archive.sh` 零增量校验。

## 硬门禁（每阶段 4 条，违规必返工）

### 数据
- 下载前先小样验证（链接可用性/文件格式/标注结构），再全量下载。
- 格式转换后做分布审计：各层样本数、空壳样本（v 全 0 / 无 bbox）、映射裁剪丢点统计——转换裁剪会系统性改变分布，报告必须注明。
- 文件名/ID 全局唯一性检查：跨视频拼数据集时逐视频 stem 会撞车，需要数值化重映射。
- 筛选口径脚本化：同类数据集（多来源）必须用同一筛选脚本，禁止手工筛选。

### 训练
- 先跑过拟合自检：100 图小样本可收敛（指标接近满分）才允许全量开训。
- 自定义 loss 逐项打印数值与梯度，loss 恒 0 = 没接进计算图；AMP fp16 下大数值计算（坐标平方类）必须转 fp32。
- 解析 results.csv 必须按表头动态取列，禁止固定列号（新 loss 列会挤偏移）。
- 每轮训练 MLflow + git 双记录；单轮只改一个变量。

### 评估
- 口径钉死：官方发布口径（如 pycocotools 协议）与框架内部指标分开存档，论文表格注明口径（两者可差 20+ 点）。
- 合成自检先于真实数据：GT 代预测 → RMSE=0；常数轨迹 → 抖动=0；分层不改匹配。
- 配对比较先对齐 universe：两模型预测图集/单元集不同时不能直接配对统计。
- 内部 val 差异要会读噪声级（±0.005 内基本无意义），判决必须上外部评测集。

### 部署
- FP16 先做：通常与 PT 逐位一致，INT8 必须实测掉点后再决策（可能掉 4+ 点）。
- 对拍工具输出逐位对比：导出链任何一环引入的差异都要能定位。
- 真机基准三坑：线程/推理标志只在 load 时生效、输入名要按转换后图、锁屏会拒启动。
- 量化/剪枝/换骨干任何一步都要在最终部署画布（如 640×640 非 rect）上复测。

### 归档与可复现
- 备份后必须零增量校验：rsync 脚本禁手写 `-e`（双 `-e` 静默失败）；macOS openrsync 的 stats 字段是 `Number of files transferred`（无 regular），sed 按此解析。
- git 提交/推送前全仓 token 扫描（token/password/api_key 模式 + 排除 tokenizer 类噪音词）。
- 复现清单（REPRODUCIBILITY.md）里的每个路径都必须实际验证存在，笔误会毁掉整个可复现声明。
- 推送 GitHub 前先测连通性：github.com:443 直连不通时走 ssh.github.com:443 + repo 级 deploy key。

## 元教训（跨阶段）

1. **口径不钉死，数字全是假**——同一模型同一数据集，官方口径与框架内部口径差 20+ 点；任何数字先问口径。
2. **合成桩先于真实数据**——自检门禁能拦住 90% 的实现错误，成本远低于跑完真实数据才发现。
3. **负面结果如实写**——6 组改进全负也是结果，比选择性报告更有价值；外部评测是唯一判决。
4. **每步产物落盘可对拍**——official.json 双段存档、csv 留档、单元 npz 落盘，事后任何数字都能追溯重算。

## 资源

- `references/01-data.md`：数据下载/格式转换/筛选配比坑（12 坑）
- `references/02-training.md`：训练配置/AMP/DDP/崩溃排查坑（12 坑）
- `references/03-evaluation.md`：评估口径/门禁/统计坑（12 坑）
- `references/04-deployment.md`：导出/NCNN/INT8/真机基准坑（8 坑）
- `references/05-reproducibility.md`：git/rsync/归档/GitHub 推送坑（10 坑）
- `scripts/verify_rsync_archive.sh`：零增量校验（openrsync 兼容）
- `scripts/plot_results_csv.py`：results.csv 画训练曲线（按表头取列）

## 出处与融合

实战出处：elthan-pose-research / pose-eval-harness / elthan-pose 三仓库（论文 A/B 配套）。
融合参考：Conchylicultor/Deep-Learning-Tricks（数值稳定、BN 技巧）、computer-vision-with-marco/yolo-training-template（数据清洗、增强同步标签）、Lance0218/Pytorch-DistributedDataParallel-Training-Tricks（seed/warmup）。
