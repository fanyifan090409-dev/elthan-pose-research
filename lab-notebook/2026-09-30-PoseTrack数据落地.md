# PoseTrack18 数据落地（2026-09-30）

## 结论

PoseTrack18 图片+标注已落地训练机并验证可用：val 170 视频 / 8,923 有标帧 / 逐帧存在性检查 0 缺失。供论文 A G2 T5（M3 时序抖动模块）使用。

## 链路

- 百度网盘 18 分卷（.aa–.ar，82GB，普通 tar 字节切分）→ Mac ~/Desktop/study/
- Mac 侧 tar -tf 校验 163,412 条 entries、结构 images/{train,val}/<video>/<frame>.jpg
- rsync Tailscale 直连 LAN → 训练机 /home/ubuntu/isef/datasets/posetrack18_parts/（约 5MB/s）
- cat 拼包 → /home/ubuntu/isef/datasets/posetrack18_images.tar（82G）→ tar -tf 再校验 163,412 条一致
- 只解压 val 子集 → /home/ubuntu/isef/datasets/posetrack18/images/val/（20,272 jpg，与 tar 清单 diff 零差异）
- labels v0.45 解压 → posetrack18/posetrack_data/（annotations/{train,val,test} 每视频独立 COCO JSON）

## 关键数字

- val 标注 JSON 170 个（另有 3 个 .al 列表文件，非视频）
- 有标帧 8,923；逐帧存在性检查 0 缺失
- 磁盘：tar 82G 保留作归档（train 未解压，需要时再解）；val 图片已就位；剩余空间充足

## M3 三个数据事实（已记入 G2 T5 任务描述）

1. val 只标 30 连续帧 + 每 4 帧抽标（100 帧视频约 48 帧有标）→ 轨迹段必须按 frame_id 间隔 >1 断段，d_t/j_t 只算间隔=1 的连续段
2. GT 的 v 第三位是 COCO 格式占位符（全 1，非遮挡标注）→ M3 匹配不做 v 过滤
3. keypoints 长度 51 = 17×3，具体关节映射以各 JSON 的 categories 定义为准（README 只保证 15 点名超集）

## 坑

- 分卷是普通 tar（非 gzip）：第一次 tar -tzf 报 "not in gzip format"，改 tar -tf 通过
- 网盘截图只见 6.5GB 系当时只显示已下完的卷，实际 18 卷 82GB
