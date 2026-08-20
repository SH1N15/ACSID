# ACSID：自适应协同信号增强的生成式推荐

> **文档版本**：v4 — 修正 P 训练方式、alpha clamp、实验缩减、RL trainer 适配
> **修订日期**：2026-08-19
> **项目名称**：ACSID (Adaptive Collaborative Semantic ID)

## 1. 研究问题

**协同信号应该在 SID 构造阶段注入，还是等到 RL 阶段再作为 reward 注入？**

核心假设：

> 纯文本 SID 只利用语义信息，无法充分表达用户行为中的协同关系。如果在 SID 构造阶段提前融合协同信号，可以得到更适合生成式推荐和后续 RL 优化的 item representation。

MiniOneRec 的 reward ablation 中，使用 SASRec logits 作为 collaborative reward 的变体出现了性能下降；作者将其归因于可能的 reward hacking / reward misalignment。这是原论文的实验观察和作者解释，不是本文的发现。基于此，本文研究是否应该把 CF 信息前移到 SID representation，而非在 RL 阶段作为 reward 注入。

---

## 2. 方法

### 2.1 Baseline：官方 MiniOneRec

```text
title + description -> Qwen3-Embedding-4B -> 2560d text embedding
                                              -> RQ-VAE -> 3-level SID
                                              -> Qwen2.5-3B-Base
                                              -> QLoRA SFT -> GRPO -> Constrained Decoding
```

### 2.2 Ours：ACSID

只改 SID 构造，下游全部复用官方：

```text
Qwen3-Embedding-4B -> text embedding -------+
                                            +-> Adaptive Fusion -> RQ-VAE -> 3-level SID
Train interaction -> Item2Vec --------------+
                                            -> Qwen2.5-3B-Base
                                            -> QLoRA SFT -> GRPO -> Constrained Decoding
```

### 2.3 核心创新：自适应协同残差注入

用 item 在训练集中的交互次数驱动自适应权重（v2，2026-08-20 由球面加权重构为残差注入）：

```
alpha_i = alpha_max * min(1, log(1 + n_i) / log(1 + n_ref))

z_hat_c = Normalize(P(z_cf))

z_i = z_text + alpha_i * ||z_text|| * z_hat_c
```

- n_i: item i 在 train 中的交互次数
- n_ref: 参考频次（train item frequency 中位数）
- alpha_max: 上限，默认 0.3; 严格满足 0 <= alpha_i <= alpha_max
- z_text: Qwen3-Embedding-4B 的 2560d embedding (frozen, 离线生成，不归一化)
- z_cf: Item2Vec 的 256d embedding (train-only)
- P: Linear(256, 2560)，与 RQ-VAE 联合训练 (见 2.4)；只学习方向，输出尺度被 Normalize 丢弃
- ||z_text||: 逐 item 的 text embedding 范数，在 FusionModule.forward 内现算（训练/建索引自动一致）

冷启动 item (n_i -> 0) -> alpha_i = 0 -> z_i 逐位等于 z_text (纯文本)
热门 item (n_i >= n_ref) -> alpha_i = alpha_max -> 残差幅度达到自身范数的 30%

alpha_i 的语义：协同残差的相对位移幅度（最多把 item 表示掰弯自身范数的 alpha_max 倍）。方向由 P 学习，幅度由 alpha_i 调度。

**为什么是残差而不是球面加权**（v1 教训）：v1 公式 Normalize[(1-a)·Normalize(z_text)+a·Normalize(P(z_cf))] 为使 alpha 独立于两侧 embedding 尺度而对 z_text 做 L2 归一化，把全部 item 压到单位球面。Qwen embedding 本身各向异性，归一化后有效方差极小，RQ-VAE 第 0 层 codebook 的 argmin 分配塌缩到 1-3/256 个 token（实测 text 碰撞率 0.65，fixed/adaptive 0.85，upstream 0.004）。残差式让 z_text 的原始分布完全不被动，alpha 语义照样有保证（尺度被 Normalize+||z_text|| 调度锁死）——两难解除。text 基线因此逐字节等价于官方 upstream 路径（原始 embedding 直入 RQ-VAE）。

### 2.4 P 的训练方式

P 是 Linear(256, 2560)，约 65.5 万参数，与 RQ-VAE 联合训练。P 的输出只取方向（L2 normalize 后），尺度被丢弃；残差幅度完全由 alpha_i * ||z_text|| 调度，因此 P 的输出尺度无法泄入 alpha 语义。RQ reconstruction/quantization loss 同时更新 P 和 RQ-VAE，梯度裁剪同时覆盖两者。

```text
Item2Vec 256d
      |
      v
Linear P (256 -> 2560)        <-- 可训练，与 RQ-VAE 同时优化
      |
      v
L2 Normalize(P(z_cf))          <-- 只取方向
      |
      v
z_i = z_text + alpha_i * ||z_text|| * Normalize(P(z_cf))
      |                            <-- z_text 原样保留，残差有界
      v
RQ-VAE (codebook 优化 + P 优化 联合训练)
```

RQ-VAE 结构不变，仅在其输入侧叠加一个有界的协同残差。Baseline、Fixed-CF、ACSID 三组的 RQ-VAE 训练协议保持一致（相同 codebook size、level 数、optimizer、epochs、seed），仅输入表示不同。

三套 RQ-VAE checkpoint 分别训练（RQ_text / RQ_fixed / RQ_adaptive），因为三个 embedding 空间不同，不能共享同一个 RQ-VAE。

注意：P 使得经验信号可被 RQ-VAE 重构，但不保证 P(z_cf) 严格落在 text embedding 的语义空间内。如需严格对齐，需要增加显式 alignment loss（如 MSE 对齐到 text embedding），但那会引入额外训练目标和超参。本项目不做显式对齐；alpha 的可解释性由"方向学习 + 幅度调度"的分工保证，而非通过空间对齐。

### 2.5 为什么不改下游

| 组件 | 是否改动 | 理由 |
|------|---------|------|
| RQ-VAE | 输入侧增加 P | 结构不变，训练协议不变 |
| SFT | 不改 | 换 SID 映射表即可，训练脚本零改动 |
| GRPO | 不改 | 直接用官方 reward / advantage / KL / sampling |
| Constrained Decoding | 不改 | SID 格式不变 |

创新是 **representation-level CF injection**，不是 RL 算法创新。如果同时改 reward、advantage、KL、sampling strategy，无法判断性能提升来自哪里。

### 2.6 RL Trainer 说明

官方 MiniOneRec 有自定义的 `minionerec_trainer.py`（ReReTrainer），而非标准 TRL GRPOTrainer。它集成了 constrained beam search、group reward 和推荐专用逻辑。本项目以官方 `rl.py + minionerec_trainer.py + LogitProcessor.py` 为基础做 QLoRA 适配，不替换成标准 HuggingFace GRPOTrainer，否则会破坏 MiniOneRec 的 constrained beam search、group reward 和推荐专用逻辑。

---

## 3. 实验设计

### 3.1 主实验

**SFT 阶段**（3 组 x 2 seed = 6 次训练）：

| 方法 | SID 构造 | Seeds |
|------|---------|-------|
| MiniOneRec (Baseline) | 纯 Text | 42, 123 |
| Fixed-CF | Text + CF, alpha=0.3 固定 | 42, 123 |
| **ACSID** | **Text + CF, alpha_i 自适应** | **42, 123** |

**GRPO 阶段**（2 组 x 2 seed = 4 次训练）：

| 方法 | SID 构造 | Seeds |
|------|---------|-------|
| MiniOneRec (Baseline) | 纯 Text | 42, 123 |
| **ACSID** | **Text + CF, alpha_i 自适应** | **42, 123** |

Fixed-CF 只在 SFT 阶段出现，用于验证"固定权重 vs adaptive"。不烧额外 RL。

**总计 10 次训练**，符合 A10 24GB 的算力条件。

### 3.2 消融实验

alpha 消融（在 SFT 阶段，seed=42）：

| alpha 设置 | 说明 |
|------------|------|
| alpha=0 | 等价于纯文本 SID（Baseline） |
| alpha=0.3 固定 | Fixed-CF |
| alpha_i 自适应 | ACSID |

这就是完整的 SID construction ablation，证明自适应融合不是简单加 CF。

### 3.3 SID Collision 对比

| 方法 | 报告指标 |
|------|---------|
| Text SID | Collision Rate, Unique SID Ratio |
| Fixed-CF SID | Collision Rate, Unique SID Ratio |
| ACSID | Collision Rate, Unique SID Ratio |

这是最直接的机制证据，不需要额外训练，只需分析三套 SID 映射表。

### 3.4 评测指标

精简为：

**推荐性能**：HR@5, HR@10, NDCG@5, NDCG@10

**SID 结构**：Collision Rate, Unique SID Ratio

**RL 稳定性**：Invalid SID Rate

多 seed 平均值报告，不加 bootstrap CI。

### 3.5 最终需要证明的三件事

1. ACSID 降低 SID 碰撞
2. ACSID 提高 SFT 推荐性能
3. ACSID + GRPO 优于 Text SID + GRPO

---

## 4. 技术栈

### 4.1 模型

| 组件 | 选择 | 说明 |
|------|------|------|
| Backbone LLM | Qwen2.5-3B-Base | 官方曾报告 Instruct 模型复现时可能出现 CC (invalid-item) 非零，建议尝试 base model；本项目采用 Base 以降低 constrained decoding 兼容性风险。这不是理论上 Base 必然不会生成非法 token，而是工程上降低风险 |
| Item Text Encoder | Qwen3-Embedding-4B | 论文原配置，冻结使用，离线生成 2560d embedding |
| 协同 embedding | Item2Vec (SGNS) | gensim 实现，只用 train 交互训练，CPU 运行 |
| 微调方式 | QLoRA | 4-bit 量化 + LoRA r=8 |

### 4.2 训练框架

| 组件 | 选择 | 说明 |
|------|------|------|
| 量化 | 4-bit (bitsandbytes) | A10 24GB 必须 QLoRA |
| LoRA | r=8, alpha=16 | 轻量适配 |
| RL trainer | 官方 minionerec_trainer.py (ReReTrainer) | 不替换为标准 TRL GRPOTrainer，保留 constrained beam search 和推荐专用逻辑 |
| RL optimizer | paged_adamw_32bit | A10 是 NVIDIA，bitsandbytes 可用 |
| 混合精度 | bf16 | A10 支持 bf16 |

### 4.3 数据

| 数据集 | 说明 |
|--------|------|
| Amazon Review - Industrial_and_Scientific | 仓库自带，3686 items, 36259 train / 4532 valid / 4533 test |

SFT 实际样本量（3 个数据集拼接）：
- SidSFTDataset: ~36,259 条（推荐任务）
- SidItemFeatDataset: ~3,686 条（SID-Title 对齐）
- FusionSeqRecDataset: ~36,259 条（序列对齐）
- 总计约 76,000 条

RL 只使用训练数据的一小部分（10%-30%），先从数万量级样本开始。官方明确提出为控制 RL 成本可以只用数万量级样本。

### 4.4 硬件运行流程

全部工程在 A10 24GB 上完成，但分阶段执行，不同阶段不同时驻留 GPU：

```text
A10 24GB

阶段 A：Qwen3-Embedding-4B
        -> 离线生成 2560d item embeddings
        -> 保存到磁盘，释放模型显存

阶段 B：Item2Vec (CPU) + RQ-VAE
        -> Item2Vec 用 CPU 训练（规模小，不需要 GPU）
        -> RQ-VAE 在 A10 上训练（含 P 投影）
        -> 生成 3 套 SID，保存到磁盘

阶段 C：Qwen2.5-3B-Base
        -> QLoRA SFT
        -> GRPO（基于官方 minionerec_trainer）
```

4B embedding 模型和 3B LLM 不同时驻留 GPU。

---

## 5. 硬件配置

### 5.1 算力

| 环境 | 硬件 | 用途 |
|------|------|------|
| 云端 NVIDIA | A10 24GB | 全部训练 + 预处理（embedding 离线生成、RQ-VAE、SFT、GRPO） |
| 本地 | - | 代码开发、文档编写 |

### 5.2 显存策略

A10 24GB 跑 Qwen2.5-3B-Base + QLoRA + GRPO：

```text
4-bit 量化
LoRA r=8
micro batch=1
gradient accumulation=8~16
gradient checkpointing=on
max length=512
GRPO group size=2~4
```

### 5.3 复现说明

本项目属于计算受限的轻量复现（parameter-efficient / resource-constrained reproduction），不是官方原配置（4-8 张 A100/H100 80GB）的严格复现。官方 README 明确 RL 可以只使用数万条量级的子集，与本项目将 RL 数据缩小到 10%-30% 的思路吻合。

---

## 6. 研究流程

严格按以下顺序执行：

```text
1. 跑通官方 MiniOneRec (Baseline SFT + GRPO)
        |
2. 统计原始 SID collision
        |
3. 只用 train 训练 Item2Vec (CPU)
        |
4. 构造 Fixed-CF SID (alpha=0.3, P 与 RQ-VAE 联合训练)
        |
5. 构造 ACSID (alpha_i 自适应, P 与 RQ-VAE 联合训练)
        |
6. SFT：Baseline vs Fixed-CF vs ACSID (各 2 seeds)
        |
7. GRPO：Baseline vs ACSID (各 2 seeds)
        |
8. Collision + alpha 消融
        |
9. 最终分析
```

---

## 7. 实现路线图

### Phase 0：环境准备（1-2 天）

- [ ] A10 安装 PyTorch + transformers + trl + bitsandbytes
- [ ] 验证：加载 Qwen2.5-3B-Base 4-bit，做一次前向推理
- [ ] 验证：QLoRA 单步 forward + backward
- [ ] 验证：官方 minionerec_trainer.py 在当前 trl/transformers 版本下可加载
- [ ] 数据检查：确认仓库自带 SID 文件、train/valid/test CSV 完整
- [ ] 用仓库自带 SID 跑一次 SFT baseline（复现官方，不改动 SID 构造）

### Phase 1：Baseline SFT + GRPO 复现（2-3 天）

- [ ] 修改 sft.py/sh：Qwen2.5-3B-Base + 4-bit + LoRA r=8
- [ ] micro batch=1, gradient accumulation=16, gradient checkpointing=on
- [ ] SFT 训练，early stopping (patience=1)
- [ ] 评测 HR@5/10, NDCG@5/10
- [ ] 修改 rl.py/sh：基于官方 minionerec_trainer.py，GRPO group size=2~4，RL 只用 10%-30% 训练数据
- [ ] GRPO 训练 2 epochs
- [ ] 评测对比 SFT-only vs SFT+GRPO

### Phase 2：SID 构造创新（2-3 天）

- [ ] 从训练集 CSV 提取用户-物品交互序列
- [ ] 训练 Item2Vec / SGNS（gensim, CPU），得到 item 协同 embedding（dim=256）
- [ ] 实现 P：Linear(256, 2560)，与 RQ-VAE 联合训练
- [ ] 计算 alpha_i = alpha_max * min(1, log(1+n_i) / log(1+n_ref))（n_ref=训练集中位数）
- [ ] 构造 Fixed-CF embedding（alpha=0.3 固定）
- [ ] 构造 ACSID embedding（alpha_i 自适应）
- [ ] 对未出现在 train 交互中的 item 使用 text-only fallback（alpha_i=0）
- [ ] 分别用 3 种 embedding 输入 RQ-VAE（含 P），生成 3 套 SID
- [ ] RQ-VAE 训练协议一致：相同 codebook size、level 数、optimizer、epochs、seed
- [ ] 重新生成全部下游文件：RQ-VAE checkpoint（3 套）、index.json（3 套）、item.json / info/*.txt（3 套）、train/valid/test CSV 中 SID 字段（3 套）
- [ ] 统计碰撞率、Unique SID ratio，对比 3 套 SID

**关键注意**：convert_dataset.py 读 .inter 文件，但训练代码直接读 CSV 中 SID 字段。需要适配转换脚本，确保 SID 字段正确写入 CSV。

### Phase 3：SFT 对比实验（2-3 天）

- [ ] Baseline (Text SID) SFT：2 个种子（42, 123）
- [ ] Fixed-CF SFT：2 个种子
- [ ] ACSID SFT：2 个种子
- [ ] 每组评测 HR@5/10、NDCG@5/10
- [ ] 报告非法 SID 生成率

### Phase 4：GRPO 对比实验（2-3 天）

- [ ] Baseline GRPO：2 个种子（42, 123）
- [ ] ACSID GRPO：2 个种子
- [ ] 评测对比 SFT-only vs SFT+GRPO

### Phase 5：消融与最终分析（1-2 天）

- [ ] alpha 消融：alpha=0 vs alpha=0.3 vs adaptive（SFT 阶段, seed=42）
- [ ] SID Collision 对比：3 套 SID 的碰撞率 + Unique SID ratio
- [ ] 分层分析：按 item 流行度分组（冷启动 vs 热门），看 ACSID 在哪类 item 上获益最大
- [ ] Case study：展示 ACSID 如何区分 Text SID 中碰撞的 item 对
- [ ] 最终可视化：碰撞率图、alpha 分布图、性能对比表

### Phase 6：工程化与文档（1 天）

- [ ] 整理代码，模块化
- [ ] 写 README：环境配置、复现步骤、实验结果
- [ ] 准备面试展示材料

---

## 8. 时间线

| 阶段 | 时间 | 累计 |
|------|------|------|
| Phase 0 | 1-2 天 | 1-2 天 |
| Phase 1 | 2-3 天 | 3-5 天 |
| Phase 2 | 2-3 天 | 5-8 天 |
| Phase 3 | 2-3 天 | 7-11 天 |
| Phase 4 | 2-3 天 | 9-14 天 |
| Phase 5 | 1-2 天 | 10-16 天 |
| Phase 6 | 1 天 | 11-17 天 |
| **总计** | **约 2-2.5 周** | |

---

## 9. 面试故事线

### 9.1 推广搜算法

- 生成式推荐 paradigm：SID 构造 -> 自回归生成
- 协同信号注入时机：SID 构造阶段 vs RL reward 阶段（本文的对比就是回答这个问题）
- Item2Vec 协同 embedding：只用 train 交互，避免数据泄露
- 自适应权重设计：冷启动 vs 热门 item 的差异化处理

### 9.2 LLM 算法工程师

- QLoRA SFT：4-bit + LoRA 在 24GB 显存上跑 3B 模型
- GRPO：基于官方自定义 trainer 的 constrained beam search + group reward
- Base vs Instruct 模型选择的工程判断
- 对 LLM 在推荐场景的理解：scaling law、世界知识注入

### 9.3 大模型工程师

- 单卡显存工程：A10 24GB 上跑 3B 模型的全流程优化
- QLoRA + GRPO 的工程实现
- 分阶段 GPU 调度：embedding 离线生成 -> RQ-VAE -> SFT -> GRPO
- 实验设计：多 seed + 消融实验 + collision 分析

---

## 10. 代码结构规划

```
MiniOneRec/                    # 原仓库代码（复现用）
├── rq/
│   ├── rqvae.py               # RQ-VAE 训练（改：输入侧增加 P 投影）
│   └── text2emb/              # 文本 embedding 生成（原代码）
├── sft.py                     # SFT 训练（改适配 QLoRA + 3B）
├── rl.py                      # RL 训练（改适配 QLoRA + 小 batch GRPO）
├── minionerec_trainer.py      # GRPO trainer（保留，做 QLoRA 适配）
├── LogitProcessor.py         # 约束解码（原代码，不改）
├── evaluate.py                # 评测（原代码）
├── data.py                    # 数据管线（原代码）
├── sft.sh                     # 【改】QLoRA 配置
├── rl.sh                      # 【改】小 batch GRPO 配置
└── requirements.txt           # 【改】保持 NVIDIA CUDA 包

acsid/                         # 【新增】ACSID 核心模块
├── item2vec.py                # Item2Vec / SGNS 训练（gensim, CPU）
├── adaptive_fusion.py         # alpha_i 计算 + P 投影 + 融合
├── generate_sid.py            # 用 fused embedding 跑 RQ-VAE 生成 SID
└── analyze_collision.py       # 碰撞率、Unique SID ratio 分析

experiments/                   # 【新增】实验管理
├── run_sft.sh                 # 通用 SFT（传入 SID 类型 + seed）
├── run_grpo.sh                # 通用 GRPO（传入 SID 类型 + seed）
└── results/                   # 实验结果（JSON 格式）
```

---

## 11. 关键参数配置

### 11.1 SFT 参数（QLoRA）

| 参数 | 值 | 说明 |
|------|-----|------|
| base_model | Qwen2.5-3B-Base | 非 Instruct |
| quantization | 4-bit | bitsandbytes |
| LoRA r | 8 | |
| LoRA alpha | 16 | |
| micro_batch_size | 1 | A10 显存受限 |
| gradient_accumulation_steps | 16 | 1 x 16 = 16 有效 batch |
| learning_rate | 3e-4 | 论文配置，可能需调小 |
| num_epochs | 10 | early stopping, patience=1 |
| gradient_checkpointing | on | 必须开 |
| max_length | 512 | |
| precision | bf16 | |

### 11.2 GRPO 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| model_path | SFT checkpoint | |
| micro_batch_size | 1 | |
| gradient_accumulation_steps | 8-16 | |
| GRPO group size | 4-8 | A10 显存受限，远小于官方 16 |
| beam_search | True | |
| beam_width | 4-16 | 视显存 |
| reward_type | ranking | rule + rank |
| num_train_epochs | 2 | |
| learning_rate | 1e-5 | |
| RL data ratio | 10%-30% | 只用训练数据子集 |
| optimizer | paged_adamw_32bit | A10 是 NVIDIA，bitsandbytes 可用 |
| trainer | 官方 minionerec_trainer.py | 不替换为标准 TRL GRPOTrainer |

### 11.3 RQ-VAE 参数（3 套一致）

| 参数 | 值 |
|------|-----|
| codebook_levels | 3 |
| codebook_size | 256 |
| learning_rate | 1e-3 |
| epochs | 10000 |
| batch_size | 20480 |
| optimizer | 相同 |
| seed | 相同 |

### 11.4 Item2Vec 参数

| 参数 | 值 |
|------|-----|
| embedding_dim | 256 |
| window_size | 5 |
| min_count | 1 |
| epochs | 20 |
| 运行设备 | CPU |

### 11.5 ACSID 自适应融合参数

| 参数 | 值 | 说明 |
|------|-----|------|
| alpha_max | 0.3 | 协同残差相对位移上限（自身范数的 30%） |
| n_ref | 训练集中位数 | 参考频次 |
| P | Linear(256, 2560) | 与 RQ-VAE 联合训练，只学方向 |
| 未覆盖 item | alpha_i=0 | 逐位等于 z_text |

---

## 12. 审计危险点与应对

### 12.1 数据泄露

Item2Vec 只用 train 交互训练。valid/test 交互不参与，否则将未来行为泄漏进 SID。

### 12.2 换 SID 是全链路

必须同步重新生成：RQ-VAE checkpoint、index.json、item.json / info/*.txt、train/valid/test CSV 中 SID 字段。convert_dataset.py 读 .inter 文件，但训练代码直接读 CSV 中 SID 字段，需要适配。

### 12.3 QLoRA 与全参数微调的差异

QLoRA 是 A10 24GB 的硬约束。LoRA r=8 对生成式 SID 任务可能足够（SID 生成是结构化预测，不需要大范围权重改动），但效果可能弱于全参数微调。通过 Baseline 复现（同样用 QLoRA）控制变量：三组实验 QLoRA 配置相同，差异只来自 SID 构造方式，不影响对比公平性。

### 12.4 GRPO group size 减小的风险

官方用 num_generations=16，A10 上只能 4-8。V100 复现报告减半到 8 后效果明显下降。应对：先看 SFT 对比是否已有 ACSID 优势，GRPO 作为补充。GRPO 用子集 + 更多 epoch 尝试弥补。

### 12.5 P 的训练稳定性

P 与 RQ-VAE 联合训练，P 的参数量小（65.5 万），RQ-VAE 的 reconstruction loss 驱动 P 优化。P 的输出经 L2 normalize 只保留方向，尺度无法泄入 alpha；残差幅度由 alpha_i·||z_text|| 有界调度，梯度裁剪同时覆盖 RQ-VAE 与 P 的参数。P 只需要学习将 CF embedding 投影到对 RQ-VAE 重构有用的方向，不需要严格对齐到 text embedding 空间。可能的失败模式是 P 学不到有用方向（adaptive 退化为 text——安全失败，不劣于基线）；如出现，考虑分两步：先冻结 RQ-VAE 预训练 P（用 text embedding 做 reconstruction 监督），再联合微调。

---

## 13. 论文故事

> MiniOneRec 的 SID 主要由商品文本语义构造，本文提出 ACSID，在不改变后续 LLM、SFT 和 GRPO 的情况下，将仅由训练集交互学习的协同表示自适应地融合到 SID 构造阶段，从而获得同时具有语义和协同行为信息的商品离散表示。

最终实验证明三件事：

1. ACSID 降低 SID 碰撞
2. ACSID 提高 SFT 推荐性能
3. ACSID + GRPO 优于 Text SID + GRPO

---

## 14. 参考资源

- 论文：arXiv:2510.24431 (MiniOneRec)
- 代码：https://github.com/AkaliKong/MiniOneRec
- 模型权重：https://huggingface.co/kkknight/MiniOneRec
- 类似方向项目https://github.com/zhangengyu/Minionerec_1
- V100 复现报告：知乎文章（4xV100 32GB，SFT 2h10min，RL 12h）
- 相关论文（创新性验证，确认无 RQ-VAE 前自适应融合先例）：
  - TIGER (Rajput et al., 2023) - RQ-VAE 用于推荐
  - LC-Rec (Zheng et al., 2024) - LLM 与 SID 对齐
  - RecZero (Kong et al., 2025) - 推理增强推荐
  - FlexCode / Semantics Meet Signals (2025-11) - 量化后双码本
  - FACE (2025-10) - CF embedding 量化自编码器映射
  - eLLa-Rec (2025-04) - 协同知识投影到 LLM 语义空间
  - TCA4Rec (2026-01) - Token 级协同对齐
  - Restoring Collaborative Signals (2026-07) - 生成阶段补充协同
  - FedCGR (2026-08) - 联邦场景协同注入