# 模型引擎选型：SGLang 对比 vLLM（Qwen3-VL-8B，BF16 高强度）

日期：2026-09-23
结论：**维持 vLLM 为生产引擎**。SGLang 留档备选，等 sm_120a 工具链成熟后复测。

## 动机

产品路线里推理 API 要拆分为独立服务，引擎选 vLLM 还是 SGLang 需要一个数据依据，而不是跟风。
按预注册判定标准跑对照实验：多轮 TTFT 低 ≥20% 或聚合吞吐 ≥1.3× 才切 SGLang，且无新增故障、单卡显存 ≤32G。

## 方法

- 同一权重：`/home/ubuntu/isef/models/qwen3-vl-8b-instruct`，两边都是 BF16 高强度（生产同配置），不做量化弱化。
- vLLM：生产实例本身，GPU 1 :8000，`--gpu-memory-utilization 0.85 --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes`，全程零停机。
- SGLang 0.5.20：GPU 0 :8001，`--mem-fraction-static 0.80 --context-length 16384 --attention-backend triton --sampling-backend pytorch --cuda-graph-max-bs-decode 16`，`SGLANG_ENABLE_JIT_DEEPGEMM=0`。
- 基准脚本：`scripts/bench_speed.py` + `scripts/bench_quality.py`（OpenAI 兼容接口，流式 + include_usage，P50/P95）。
- 四组速度 + 一组质量：
  - A 单流解码（1002 token 提示，5 轮，TTFT + tok/s）
  - B 并发聚合吞吐（并发 1/2/4/8，3 轮）
  - C TTFT 三档输入（约 2K/8K/16K token，5 样本）
  - D 多轮共享前缀（6000 字符前缀 × 5 轮对话，2 大轮；RadixAttention vs vLLM prefix cache）
  - 质量组：24 条固定提示 temp=0（教练话术 8 + 结构化 JSON 8 + Python 代码 8），校验长度/JSON 合法性/代码语法，跨引擎比对

## 原始数据

### A 单流解码（5 轮）

| 指标 | SGLang | vLLM |
|---|---|---|
| TTFT P50 | 27 ms | 24 ms |
| 解码速度 | 89.8–90.1 tok/s | 97.5–97.8 tok/s |
| 总时长 | 0.74–0.88 s | 0.68–0.80 s |

### B 并发聚合吞吐（tok/s，3 轮中位数）

| 并发 | SGLang | vLLM | vLLM/SGLang |
|---|---|---|---|
| 1 | 87.2 | 94.7 | 1.09× |
| 2 | 158.4 | 175.8 | 1.11× |
| 4 | 304.4 | 310.0 | 1.02× |
| 8 | 557.7 | 575.3 | 1.03× |

（vLLM 并发 8 首轮 4.7s/128 tok/s 为一次性冷启动异常，后两轮正常，已剔除。）

### C TTFT 三档输入（P50，前缀缓存命中）

| 输入 | SGLang | vLLM |
|---|---|---|
| ~2K token | 28 ms | 19 ms |
| ~8K token | 43 ms | 33 ms |
| ~16K token | 64 ms | 52 ms |

冷首包（同提示首发的完整单 token 请求时长，近似冷 prefill）：

| 输入 | SGLang | vLLM |
|---|---|---|
| ~2K token | 96 ms | 85 ms |
| ~8K token | 621 ms | 509 ms |
| ~16K token | 1181 ms | 874 ms |

SGLang 冷 prefill 在 16K 档慢约 35%（triton attention 后端 vs vLLM 在 5090 上的优化注意力）。

### D 多轮共享前缀（3992 token 前缀 + 5 轮）

| 指标 | SGLang | vLLM |
|---|---|---|
| 各轮 TTFT | 28–36 ms | 27–30 ms |
| 解码速度 | 82.9–85.9 tok/s | 94.6–96.9 tok/s |

两边的共享前缀缓存都生效（第 5 轮 TTFT 与第 1 轮持平，说明前缀只算一次）。

### 质量组（24 条，temp=0）

| 组 | SGLang 校验 | vLLM 校验 | 跨引擎逐字一致 |
|---|---|---|---|
| 教练话术 | 8/8 | 8/8 | 5/8 |
| 结构化 JSON | 8/8 | 8/8 | 6/8 |
| Python 代码 | 8/8 | 8/8 | 8/8 |

两边全部通过校验。代码组 8/8 逐字一致；话术/JSON 有 5 条文案不同但都合法（temp=0 下 kernel 级不确定性的正常表现）。

### 显存

- SGLang：27.5 GB（mem-fraction 0.80，GPU 0），单卡 ≤32G ✓
- vLLM：26.4 GB（0.85，GPU 1）✓

## 判定（对照预注册标准）

| 标准 | 结果 |
|---|---|
| 多轮 TTFT 低 ≥20% | ✗ 不成立：vLLM 反而低约 13%（30ms vs 34ms） |
| 聚合吞吐 ≥1.3× | ✗ 不成立：vLLM 在全部并发档高 2–11% |
| 无新增故障 | 跑测零失败，但见下方部署成本记录 |
| 单卡显存 ≤32G | 两边都过 |

**结论：维持 vLLM。**

## 选型依据（比数字更重的部分）

性能上两引擎互有短长但差距全部 <13%，真正拉开差距的是部署成本：

1. **vLLM 在这台机器上 pip 即装即跑**，生产实例已连续稳定运行。
2. **SGLang 0.5.20 在 RTX 5090（sm_120a）上需要 8 轮工具链手术才能启动**：
   - pip 包没有 nvcc，系统 CUDA 12.8 报 "NVCC version must be at least 12.9"；
   - `nvidia-cuda-nvcc` 13.0.88：前端输出 PTX 9.4，自带 ptxas 只认 9.0，报 "Unsupported .version 9.4"；
   - 升到 13.4.92 又和 cu13 头文件版本不匹配（flashinfer JIT 报头文件不兼容）；
   - 最终方案：conda `cuda-nvcc` 13.3（nvcc/cicc/ptxas 全家一致）+ `lib64/libcudart.so` 软链（conda 包把库放 `lib/`，JIT 却在 `lib64` 找）；
   - flashinfer JIT 仍缺 `curand.h`，注意力换 triton 后端、采样换 pytorch 后端绕开；
   - CUDA graph 捕获失败，压到 `--cuda-graph-max-bs-decode 16` + `--mem-fraction-static 0.80`。
3. 结论：在 sm_120a 支持成熟（flashinfer 修复）之前，SGLang 在这台 5090 上的维护成本不值得 ~10% 以内的性能差异。备选结论成立，复测触发条件写进待办。

## 测量方法学记录（给后来的自己）

1. **requests.post 不带 `stream=True` 会把 SSE 整包缓冲**，首版脚本所有 TTFT 都等于总时长（假象），排查了客户端/服务端/代理后才定位到这一个参数。修复后数据才可信。
2. Qwen3-VL 在 temp 0.7 下偶发思考模式（空 content 前缀约 0.8s），"首字可见时间"不稳定，因此 TTFT 主指标定为**首个 SSE 数据包时间**，content 首字时间作参考。
3. C 组同提示重复发送会命中 radix/prefix 缓存，warm TTFT 与冷 prefill 分开记录，不可混用。
