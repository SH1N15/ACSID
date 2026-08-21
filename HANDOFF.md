# ACSID — acsid-amd 分支交接文档 v2

> 交接时间：2026-08-21（Phase 3 SFT 评测跑完，三模式 HR/NDCG/CC 已出；下一步 Phase 4 GRPO）
> 分支：`acsid-amd`，远程：`https://github.com/SH1N15/ACSID.git`（私有，`gh` 登录账号 `SH1N15`）
> 分支状态：所有 commit 已 push 到 `origin/acsid-amd`，工作树干净
> 云端环境：1x AMD MI300X 192GB / ROCm 7.2.3 / torch 2.11.0 / Python 3.12

---

## 1. 项目本质

ACSID：把协同过滤信号（Item2Vec）前移到 RQ-VAE 输入阶段，与文本 embedding 融合生成 SID。下游 SFT/GRPO/Constrained Decoding 完全复用官方 MiniOneRec，只换 SID 映射表。

**方法公式（v2 残差注入，2026-08-20 重构后已验证）**：

```
alpha_i = alpha_max * min(1, log(1+n_i) / log(1+n_ref))
z_i = z_text + alpha_i * ||z_text|| * Normalize(P(z_cf))
```

text 模式（无 CF）：`z_i = z_text`（纯原始文本，等价 upstream MiniOneRec）
fixed 模式：`alpha_i = alpha_max = 0.3`（所有 item 常量）
adaptive 模式：`alpha_i` 按上式，冷启动 `alpha_i=0` 退化为纯文本
P 只学方向，幅度由 alpha 调度。z_text 永不归一化。

---

## 2. 已完成

### Phase 2：SID 构造 ✅（三套 SID 均可用，已入库）

残差注入范式验证通过，碰撞率全面收敛：

| mode | collision_rate | unique | level-0 distinct |
|---|---|---|---|
| upstream | 0.0043 | 3670/3686 | 48 |
| text | 0.0030 | 3675/3686 | 86 |
| **fixed** | **0.0** | **3686/3686** | 79 |
| **adaptive** | **0.0011** | **3682/3686** | 56 |

文件：`experiments/results/collision.json`（已提交入库）

关键工程修复：
- `trainer.py` fast path：`len(dataset) <= batch_size` 时全量常驻 GPU，10000 epochs 从 ~2h 降至 ~20 分钟
- grad clipping 覆盖 RQ-VAE + Fusion 两层
- `generate_sid.py`：旧 ckpt 目录清理（防 mtime 误选）+ `--force_regen`

### Phase 3：SFT 训练 ✅（3 组 × seed=42）

三模式 SFT 全部完成。text/fixed/adaptive 三组均触发 early stopping（patience=3，eval_loss 收敛后停，具体 epoch 以各自 `trainer_state.json` 为准）。

**运行产物位置**（`/mnt/workspace/ACSID/MiniOneRec/`）：
```
output_dir/sft_text_seed42/final_checkpoint/    # 只有 safetensors + tokenizer（可加载）
output_dir/sft_fixed_seed42/final_checkpoint/
output_dir/sft_adaptive_seed42/final_checkpoint/

# 其余 checkpoint-XXX/ 是中间产物，可删（释放 ~100GB）
```

**SFT 加速配置（最终稳定版）**：
| 项 | 值 |
|---|---|
| micro_batch_size | 64 |
| gradient_accumulation | 1024/64=16（脚本自动算） |
| precision | bf16 |
| attention | `attn_implementation="sdpa"`（SDPA kernel）|
| gradient_checkpointing | False |
| dataloader | 默认（数据充足，不瓶颈） |
| torch.compile | False（ROCm 不稳定） |

### Phase 3 评测 ✅（2026-08-21 跑完，三模式 evaluate.py + calc.py）

> 命令：`SKIP_MODES="" PHASES="eval" SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh`（log 在 `MiniOneRec/logs/eval_sft_phase3_*.log`，每 ~32 分钟 beam search）

结果（test 4533 条，beam=50，CC=非法 SID 计数）：

| metric | text 基线 | fixed α=0.3 | adaptive 我们 | adaptive vs text | fixed vs text |
|---|---|---|---|---|---|
| NDCG@1  | 0.0547 | 0.0688 | 0.0613 | +12% | +26% |
| NDCG@5  | 0.0667 | 0.0877 | 0.0778 | +17% | +31% |
| NDCG@10 | 0.0734 | 0.0952 | 0.0850 | +16% | +30% |
| NDCG@50 | 0.0911 | 0.1167 | 0.1031 | +13% | +28% |
| HR@5    | 0.0774 | 0.1059 | 0.0935 | +21% | +37% |
| HR@10   | 0.0984 | 0.1295 | 0.1160 | +18% | +32% |
| HR@50   | 0.1802 | 0.2296 | 0.1999 | +11% | +27% |
| CC      | 0      | 0      | 0      | —    | —    |

完整 NDCG/HR@{1,3,5,10,20,50} 已在归档 log。CC 全 0，constrained decoding 干净。

**结论**：
1. **证明目标 #2（ACSID 提升 SFT 推荐性能）成立** —— adaptive 在 NDCG/HR 所有 K 上全面胜过 text 基线（+11%~+21%）。
2. **意外发现**：fixed > adaptive > text。fixed（α=0.3 常量）反超 adaptive ~10%。**机制假设**：adaptive 对冷启 item 令 α=0 退出 CF（安全但放弃信号），fixed 对所有 item 一律 α=0.3（含冷启），齐齐残差位移使 collision 也最低（0.0 vs 0.0011）。alpha=0.3 恰可能比自适应更稳定的位移幅度。**待 Phase 5 分层验证**（按 item 流行度分组对比 HR/NDCG，证实是否冷启 item 的损失拖低了 adaptive）。
3. fixed>adaptive 不否定 adaptive 的冷启安全性价值，但叙事点要诚实——这是消融结果的一部分，Phase 5 正式分析。

产物：`MiniOneRec/results/eval_sft_{text,fixed,adaptive}_seed42.json`，`MiniOneRec/logs/eval_sft_phase3_*.log`。

或用 run_experiments.sh 统一跑：
```bash
SKIP_MODES="" PHASES="eval" SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh
```

---

## 3. 待完成（下一位 AI）

> 状态：GRPO 段 `sft_ckpt` 路径阻塞已清除（commit `63b850c` 已修），下一位可直接跑 eval → GRPO，无需改任何脚本。

### Phase 4：GRPO（2 组 × seed=42）

- Text baseline GRPO + Adaptive GRPO
- ✅ `sft_ckpt` 路径已修正：`acsid_amd/run_experiments.sh:138` 已指向 `output_dir/sft_${mode}_seed${seed}/final_checkpoint`（commit `63b850c` 修复，eval 段 `:105` 同样已正确）。**无需再改，直接跑：**
  ```bash
  cd /mnt/workspace/ACSID/MiniOneRec
  source ../.venv-amd/bin/activate
  SKIP_MODES="" PHASES="grpo" SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh
  ```
- 注意：GRPO 段不读 `SKIP_MODES`，固定只跑 `text adaptive` 两模式（fixed 不进 GRPO，符合计划——fixed 只在 SFT 阶段验证"固定权重 vs adaptive"）。
- **epoch 2→1（偏离 PLAN_AMD §11.2 的记录）**：实测全量配置 13194 步 × ~7.5s/步 ≈ 25.7h，远超云实例单会话 8h 上限。2026-08-21 改为 1 epoch（6597 步 ≈ 13.7h），数据保持全量。两 mode 设置完全一致，对比公平性不受影响。
- **跨会话续跑机制（已实现）**：`rl.py` 训练前用 `get_last_checkpoint` 探测 `output_dir/grpo_*/checkpoint-*`，有则显式 `trainer.train(resume_from_checkpoint=...)` 续跑（裸 train() 默认从头训！）；`run_experiments.sh` GRPO 循环对已有 `final_checkpoint/` 的 mode 直接跳过。**下个会话只需 git pull 后原样重跑同一条命令**。save_steps=0.25 → 每 ~1650 步一个恢复点（含 optimizer states）。瓶颈在约束 beam search 的 Python 掩码循环（GPU 等喂），micro_batch 已翻倍至 128 无效，group=16 不减，接受跨会话。
- 前置：先确认 `output_dir/sft_{text,adaptive}_seed42/final_checkpoint/` 还在——§7 的清理只删 `checkpoint-*` 中间产物，保留 `final_checkpoint/`，不要误删。

### Phase 5：消融与最终分析

- alpha 消融已随主实验完成（alpha=0→text，0.3 固定→fixed，自适应→adaptive）
- 分层分析：冷启 item vs 热门 item 分别看 HR/NDCG
- Case study：text 中碰撞 item 对在 adaptive 中是否分开

### Phase 6：工程化文档

README 环境配置、复现步骤、实验结果。面试故事材料。

---

## 4. 重要环境坑（云端踩过的）

1. **存储上限 100GB**：单个 SFT checkpoint ~18GB（optimizer.pt 11.5GB + 模型 5.7GB），三个满了就满了。seed=123 已取消（不过加 seed 很容易，改过 epochs=5 就够）。

2. **bitsandbytes 不可用**：ROCm 没有 bnb build，evaluate.py 已删死 import。

3. **transformers 4.57 `resize_token_embeddings`**：`mean_resizing=True` 会调用 `torch.linalg.cholesky_ex`（无 LAPACK）→ `sft.py` 已用 `mean_resizing=False` + try/except 降级。

4. **wandb 非交互模式**：必须设 `WANDB_MODE=disabled`，不然 nohup 下 prompt 卡死。已在 `sft.py` 里内置。

5. **显存问题**：micro_batch=64 + bf16 能跑（97% VRAM 但不 OOM），128 会 OOM。

6. **ROCm 上 python 进程遗留**：上次 text 训练结束后 GPU 显存没释放，不影响（下一个进程正常重新占用）。

7. **GitHub 偶发网络 reset**：push 失败重试。

8. **qwen3-embedding 4B**：如果不是 ModelScope 下载，地址在 https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B。

---

## 5. 文件地图

```
acsid/                          # SID 构造核心（两分支共享）
  adaptive_fusion.py             # 残差注入公式（v2）
  generate_sid.py                # 编排器（Item2Vec→alpha→3×RQ-VAE→index→CSV）
  regenerate_csv_sid.py          # SID 列替换
  analyze_collision.py           # 碰撞率统计

acsid_amd/                      # AMD 分支独有
  sft.py                         # SFT 入口（bf16全参数，SDPA，无LoRA/bnb）
  rl.py                          # GRPO 入口（adamw_torch）
  sft.sh / rl.sh                # 单次启动模板
  run_experiments.sh             # 总控（PHASES/SEEDS_STR/SKIP_MODES 可调）
  setup_env.sh                   # venv 构建（幂等）

MiniOneRec/                     # 原仓库
  rq/trainer.py                  # RQ-VAE trainer（fast path + grad clip 完整）
  rq/generate_indices.py         # build_fused_matrix（残差路径与 trainer 一致）
  evaluate.py                    # 评测生成（受约束 beam search）
  calc.py                        # HR/NDCG 计算
  data.py                        # 数据集类（CSV/JSON 两套）

experiments/
  results/collision.json          # Phase 2 碰撞率结果
  run_phase2_sid. Experiment entry point.

HANDOFF.md                      # 本文档（交接用）
PROJECT_PLAN.md                 # A10 版方案文档（QLoRA 分支 main）
PLAN_AMD.md                     # AMD 版方案文档（本文档）
```

---

## 6. 关键 commit 链（最近 10 个）

```
2c971b2 perf: drop useless dataloader/HIP optimizations (GPU already compute-bound)
760558d results: final Phase 2 collision rates (residual injection)
63b850c feat: SKIP_MODES env var; point eval/GRPO at final_checkpoint
b10c114 fix: set WANDB_MODE=disabled in sft.py directly
e8ff831 → 98eae8a: 几轮调优和回退（TF32/cudnn/group_by_length 试了发现没用）
4f63378 perf: SDPA attention + micro_batch=64 (128 OOM'd)
a3a6306 docs: reduce experiment to single seed (42) — 100GB storage constraint
0b8ae63 feat: rework fusion to residual injection; text path equals upstream
```

## 7. 下一位 AI 接手第一步

```bash
# 1. 拉代码
cd /mnt/workspace/ACSID && git pull origin acsid-amd

# 2. 确认三个 SFT checkpoint 都在
ls -la output_dir/sft_text_seed42/final_checkpoint/
ls -la output_dir/sft_fixed_seed42/final_checkpoint/
ls -la output_dir/sft_adaptive_seed42/final_checkpoint/

# 3. 跑评测（最优先）
cd MiniOneRec && source ../.venv-amd/bin/activate
SKIP_MODES="" PHASES="eval" SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh

# 4. 把三组 calc.py 输出发给 AI 分析

# 5. 清理中间 checkpoint（释放 ~100GB）
rm -rf output_dir/sft_*/checkpoint-*
```

---
*文档写入：2026-08-20，acsid-amd 分支，HEAD 2c971b2*
