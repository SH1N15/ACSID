# ACSID — AMD 192GB 环境方案

> **文档版本**：v1-amd
> **修订日期**：2026-08-19
> **硬件**：1x AMD Instinct MI300X 192GB HBM3
> **创新点**：与 A10 方案完全一致（ACSID），仅训练配置不同

## 1. 与 A10 方案的差异

| 维度 | A10 方案 (v4) | AMD 方案 (本文件) |
|------|-------------|----------------|
| GPU | A10 24GB 单卡 | 1x MI300X 192GB |
| 微调方式 | QLoRA (4-bit + LoRA r=8) | **全参数微调** |
| 有效 batch | 16 (micro=1 x accum=16) | **1024** (micro=16 x accum=64) |
| GRPO group_size | 2-4 | **16** (论文原配置，192GB 可承载) |
| RL 数据量 | 10%-30% 子集 | 全量 |
| optimizer | paged_adamw_32bit (bitsandbytes) | adamw_torch |
| 分布式 | 单卡 | 单卡 (192GB 显存足够，不需要 ZeRO) |
| 训练质量 | QLoRA 近似 | 全参数，接近论文原配置 |

**核心优势**：AMD 192GB 可以跑接近论文原配置的全参数微调，不需要 QLoRA 近似。GRPO group_size=16 不减配，RL 用全量数据。这直接解决了 A10 方案最大的两个风险：QLoRA 效果打折和 GRPO group size 太小。

**核心风险**：ROCm 生态兼容性。bitsandbytes 不支持 AMD，需要替换。DeepSpeed ZeRO-2 官方支持 ROCm 但需要验证。trl 的 GRPO trainer 在 ROCm 上未验证。

## 2. 环境

### 2.1 ROCm 兼容性清单

| 组件 | A10 (CUDA) | AMD (ROCm) | 处理方式 |
|------|-----------|-----------|---------|
| PyTorch | torch+cu118 | torch+rocm6.x | pip install torch --index-url https://download.pytorch.org/whl/rocm6.x |
| bitsandbytes | 0.48.1 | 不支持 | **移除**，全参数微调不需要 |
| DeepSpeed | 0.18.0 | 官方支持 ROCm | 保留，Phase 0 验证 |
| trl | 0.24.0 | 未验证 | 保留，Phase 0 验证 |
| flash attention | flash_sdp | ROCm 有自己的 CK | 保持 `enable_flash_sdp(False)`，用 math backend |
| torchrec | +cu118 | 不支持 | 移除 |
| fbgemm_gpu | +cu118 | 不支持 | 移除 |
| nvidia-* 包 | 需要 | 不需要 | 移除 |
| gensim | CPU | CPU | 保留 |

### 2.2 分阶段运行流程

```text
AMD MI300X 192GB 单卡全流程

阶段 A：Qwen3-Embedding-4B (单卡，离线)
        -> 离线生成 2560d text embedding
        -> 保存到磁盘，释放显存

阶段 B：Item2Vec (CPU) + RQ-VAE (单卡)
        -> Item2Vec 用 CPU 训练
        -> RQ-VAE + P 在单卡上训练
        -> 生成 3 套 SID

阶段 C：SFT (单卡)
        -> Qwen2.5-3B-Base 全参数微调
        -> micro_batch=16, accum=64, 有效 batch=1024
        -> 192GB 显存足够，不需要 ZeRO 分布式

阶段 D：GRPO (单卡)
        -> 基于 SFT checkpoint
        -> group_size=16 (论文原配置，192GB 可承载模型+ref+16 generations)
        -> 全量 RL 数据
```

## 3. 代码修改清单

本文件夹内的文件是 MiniOneRec 原始代码的 AMD 适配版本，不修改原文件。

### 3.1 sft.py

| 修改项 | 原始 | AMD 版本 |
|--------|------|---------|
| bitsandbytes 导入 | `import bitsandbytes as bnb` | 删除（未使用，且 AMD 不支持） |
| flash_sdp | 无 | 新增 `torch.backends.cuda.enable_flash_sdp(False)` (RL 和 SDP 兼容性) |
| 其他 | 不变 | 不变 |

sft.py 原始代码已经是全参数微调（bf16 加载，无量化），optim 已是 adamw_torch，不需要额外修改。

### 3.2 rl.py

| 修改项 | 原始 | AMD 版本 |
|--------|------|---------|
| flash_sdp | `enable_flash_sdp(False)` | 保留 |
| mem_efficient_sdp | `enable_mem_efficient_sdp(False)` | 保留 |
| optimizer | `paged_adamw_32bit` | **改为 `adamw_torch`** |

### 3.3 sft.sh

| 修改项 | 原始 | AMD 版本 |
|--------|------|---------|
| nproc | 8 (torchrun) | **不使用 torchrun** (单卡) |
| batch_size | 1024 | 1024 (不变) |
| micro_batch_size | 16 | 16 (不变) |
| accumulation | 1024/(16*8)=8 | 1024/(16*1卡)=**64** |
| base_model | your_model_path | Qwen2.5-3B-Base 路径 |

### 3.4 rl.sh

| 修改项 | 原始 | AMD 版本 |
|--------|------|---------|
| num_processes | 8 (accelerate) | **不使用 accelerate** (单卡) |
| train_batch_size | 64 | 64 (不变) |
| num_generations | 16 | **16 (不减配!)** |
| gradient_accumulation | 2 | **2** (不变) |
| RL data | 全量 | 全量 (不减配) |

### 3.5 requirements.txt

移除所有 CUDA-only 包（torchrec+cu118, fbgemm_gpu+cu118, bitsandbytes, nvidia-*）。
保留 DeepSpeed (ROCm 支持)。
torch 改为 ROCm 版本安装。

### 3.6 zero2_opt.yaml

单卡不需要 ZeRO。sft.py/rl.py 原代码已经支持单卡模式（ddp=False 分支）。如果显存不够可启用ZeRO-2，但 192GB 预计足够。

## 4. 实验设计

与 A10 方案一致：

**SFT**（3 组 x 2 seed = 6 次）：
- MiniOneRec (Text SID) x seeds {42, 123}
- Fixed-CF (alpha=0.3) x seeds {42, 123}
- ACSID (adaptive alpha_i) x seeds {42, 123}

**GRPO**（2 组 x 2 seed = 4 次）：
- MiniOneRec (Text SID) x seeds {42, 123}
- ACSID (adaptive alpha_i) x seeds {42, 123}

**消融**（SFT 阶段, seed=42）：
- alpha=0 vs alpha=0.3 vs adaptive
- SID Collision: 3 套 SID 的碰撞率 + Unique SID Ratio

**总计 10 次训练**。

### 4.1 训练时间估算

| 阶段 | V100 32GB x4 参考 | MI300X 192GB 估算 | 说明 |
|------|-------------------|---------------------|------|
| SFT | ~2h10min (6.5 epochs) | 2-4h | 全参数，batch=1024，单卡 |
| GRPO | ~12h (减配: num_gen=8) | 12-18h (全配: num_gen=16) | 192GB 可承载 group=16 |
| 单组 SFT+GRPO | ~14h | 14-22h | |
| 6组SFT + 4组GRPO | - | ~56-100h | 全部实验 |

## 5. Phase 0 验证清单

在正式训练前必须完成：

- [ ] 安装 ROCm 版 PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`
- [ ] 验证 4 卡可见：`python -c "import torch; print(torch.cuda.device_count())"`
- [ ] 验证 DeepSpeed ZeRO-2：加载小模型 forward+backward，确认梯度同步正常
- [ ] 加载 Qwen2.5-3B-Base bf16，前向推理正常
- [ ] 验证 trl 可导入（`from trl import GRPOConfig`）
- [ ] 验证 minionerec_trainer.py 可导入（`from minionerec_trainer import ReReTrainer`）
- [ ] 确认 bitsandbytes 已从环境中移除或不可导入（避免冲突）

## 6. 文件结构

```
acsid_amd/                    # 本文件夹
├── PLAN_AMD.md              # 本文档
├── sft.py                   # MiniOneRec/sft.py 的 AMD 适配版 (去 bitsandbytes)
├── rl.py                    # MiniOneRec/rl.py 的 AMD 适配版 (optim=adamw_torch)
├── sft.sh                   # 单卡 SFT 启动脚本
├── rl.sh                    # 单卡 GRPO 启动脚本
├── run_experiments.sh        # 全量实验矩阵 (6 SFT + 4 GRPO = 10 次)
├── requirements.txt         # ROCm 兼容依赖
└── config/
    └── zero2_opt.yaml        # DeepSpeed ZeRO-2 (备用，单卡预计不需要)
```

使用时将本文件夹的文件复制到 MiniOneRec/ 目录覆盖原文件（建议先备份原文件）。
ACSID 创新模块（../acsid/）与 SID 构造代码不变，三套 SID 由 A10 或 AMD 生成后直接使用。