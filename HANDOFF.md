# ACSID — `acsid-amd` 分支交接文档

> 交接日期：2026-08-20
> 交接时分支状态：`acsid-amd` @ `9ccf0a6`（已与 `origin/acsid-amd` 同步，工作区内容干净）

## 1. 仓库 / 分支

- **remote**: `https://github.com/SH1N15/ACSID.git`（私有，`gh` 登录账号 `SH1N15`）
- **当前分支**: `acsid-amd` — AMD MI300X 全参数方案
- **对比分支**: `main` — A10 24GB QLoRA 方案（Phase 2 代码完成，未跑训练）

## 2. 项目本质

ACSID（Adaptive Collaborative Semantic ID）：
把协同过滤信号（Item2Vec）**前移到 RQ-VAE 输入阶段**，通过学习投影 `P` + 逐 item 自适应权重 `alpha_i` 融合进 SID 构造；下游 SFT / GRPO / 约束解码不改。
产出三套 SID（`text` / `fixed` / `adaptive`），目标是证明 **ACSID 降低碰撞率** 且 **优于 text-only**。
完整方法见 `PROJECT_PLAN.md`（A10 版）与 `acsid_amd/PLAN_AMD.md`（AMD 版）。

核心公式（v2 残差注入，2026-08-20 重构，见 §5 决策记录）：
```
alpha_i = alpha_max * min(1, log(1+n_i) / log(1+n_ref))   # adaptive
z_i = z_text + alpha_i * ||z_text|| * Normalize(P(z_cf))  # text 模式: z_i = z_text（原始，等价 upstream）
```

## 3. 硬件 / 环境

- **1x AMD Instinct MI300X 192GB HBM3**
- 云端镜像：`ubuntu22.04-rocm7.2.3-py312-torch2.11.0-1.39.0`
- 云端工作目录：`/mnt/workspace/ACSID`（Linux, root）
- 隔离 venv：`.venv-amd`（`--system-site-packages` 继承系统 torch 2.11+ROCm 7.2.3）
  - 需手动激活：`source .venv-amd/bin/activate`
  - 环境脚本：`acsid_amd/setup_env.sh`（幂等，可重建）

## 4. 目录结构

```
acsid/                      # 与 main 共享的核心创新代码
  item2vec.py               # Item2Vec/SGNS，仅 train，CPU
  adaptive_fusion.py        # compute_alpha + FusionModule(P + 残差注入)
  generate_sid.py           # 端到端编排（RQVAE+P -> 3 套 index.json -> 重写 CSV，check=True fail-fast）
  regenerate_csv_sid.py     # 绕开 .inter，只重写 CSV SID 列 + info/*.txt
  analyze_collision.py      # 碰撞率 / Unique SID Ratio 对比
acsid_amd/                  # AMD 独有
  PLAN_AMD.md               # AMD 方案文档
  sft.py  rl.py             # 全参数适配（删 bnb，optim=adamw_torch，sys.path 注入等）
  sft.sh  rl.sh  run_experiments.sh   # 自定位单卡启动
  setup_env.sh              # 隔离 venv 构建
  requirements.txt          # ROCm 兼容依赖
  config/zero2_opt.yaml     # 单卡备份配置（未被脚本引用）
  README.md                 # AMD 运行指南
MiniOneRec/rq/              # RQ-VAE 改动（datasets/rqvae/trainer/generate_indices）
experiments/                # run_phase2_sid.sh + results/collision.json
```

## 5. 待决问题 → 已决策（2026-08-20）

### 碰撞率异常高 —— 已修复：融合范式重构为残差注入
- **现象**：text 碰撞率 0.65 / fixed 0.85 / adaptive 0.85（upstream 0.004）；level-0 codebook 仅用 1-3/256 token；recon loss 0.0000。
- **根因确认**：所有模式对 `z_text` 做 L2-normalize（`trainer._prepare_input`），压塌到单位球面 + Qwen embedding 各向异性 → codebook argmin 塌缩。且 fixed/adaptive 的 CF 信息实际被球面加权稀释，根本没有注入（这解释了为何比 text 更糟）。
- **为什么没走原候选方案 A/B/C**：方案 A（text 侧不 normalize、其余保留）有内在矛盾——z_text 保持大尺度而 `Normalize(P(z_cf))` 在单位球面，α=0.3 时 CF 相对贡献仅约 2%，α 语义失效；且融合后再 Normalize 仍会塌回球面（≈当前已塌缩的 text 路径 + 噪声）。
- **已实施（残差注入）**：`z_i = z_text + α_i·‖z_text‖·Normalize(P(z_cf))`；方向由 P 学习、幅度由 α 调度，z_text 永不归一化，text 基线逐字节等价 upstream。改动：
  1. `acsid/adaptive_fusion.py::FusionModule.forward` — 残差式重写；删除死代码 `fuse_full`
  2. `MiniOneRec/rq/trainer.py::_prepare_input` — text 分支返回原始 z_text
  3. `MiniOneRec/rq/generate_indices.py::build_fused_matrix` — text 分支同步
  4. `trainer.py` — 梯度裁剪纳入 fusion/P 参数（原 bug：只 clip RQ-VAE）
  5. `acsid/generate_sid.py` — 训练前清理该 mode 旧 ckpt 目录（防 mtime 误选中止残留）；`--force_regen` 强制重生成 cf/alpha
  6. 文档同步：PLAN_AMD / PROJECT_PLAN §2.3-2.4、§11.5、§12.5，acsid/README
- **待办**：云端重跑三模式 RQ-VAE + analyze_collision 验证。验收：text ≤ 0.01（upstream 量级）；方向预期 adaptive ≤ fixed ≤ text；若 fixed/adaptive ≈ text 则为 P 未学到方向的安全失败，调 α_max 或上 P 两步预热，不推翻范式。

## 6. 云端上次运行状态

- 三模式 RQ-VAE 全量训练被**手动中止**（碰撞率异常，需先决定 §5 再重跑）。
- 依赖/环境已确认可用：torch 2.11 ROCm 可见；缺 scikit-learn 已补；RQ-VAE 能起训练。
- 正确跑法是 `bash acsid_amd/setup_env.sh` → `source .venv-amd/bin/activate` → 从 `MiniOneRec/rq/` 跑 `generate_sid.py`。

## 7. 建议接手第一步

1. `cd /mnt/workspace/ACSID && git pull origin acsid-amd`（残差注入改动已推送）
2. 冒烟验证（text 模式，短 epochs，确认 level-0 distinct 恢复）：
   ```bash
   cd MiniOneRec/rq && source ../../.venv-amd/bin/activate
   python ../../acsid/generate_sid.py --dataset Industrial_and_Scientific \
     --epochs 200 --batch_size 2048 --eval_step 50 --device cuda:0 --modes text
   ```
   通过标准：level-0 distinct token 明显大于 3（塌缩时只有 3）。
3. 全量三模式：
   ```bash
   python ../../acsid/generate_sid.py --dataset Industrial_and_Scientific \
     --epochs 10000 --batch_size 20480 --eval_step 50 --device cuda:0 \
     --modes text fixed adaptive
   ```
4. 碰撞对比：
   ```bash
   python ../../acsid/analyze_collision.py --base ../data/Amazon \
     --dataset Industrial_and_Scientific --include_upstream \
     --out_json ../../experiments/results/collision.json
   ```
   核对 upstream vs text/fixed/adaptive（验收见 §5 待办）。

## 8. 已修复 Bug 清单（供排查参考，勿重复定位）

1. `acsid_amd/rl.py` 语法错误（`optim="..."` 逗号落进注释）→ py_compile 抓到
2. `acsid_amd/sft.py`/`rl.py` 缺 `sys.path` 注入（导入 MiniOneRec 包失败）
3. `acsid_amd/sft.py` `freeze_LLM` NameError（`original_vocab_size` 未定义）
4. `acsid_amd/sft.py` TokenExtender 忽视多模式 `--sid_index_path`
5. `acsid_amd/sft.sh`/`rl.sh`/`run_experiments.sh` cwd 错位 → 自定位
6. `acsid_amd/setup_env.sh` heredoc 语法错误；半残 venv 检测；不再 `--upgrade setuptools`
7. `acsid_amd/requirements.txt` 8 个坏 pin（`==` 指向不存在的 PyPI 版本）+ 缺 `scikit-learn`
8. `acsid/regenerate_csv_sid.py` glob 太宽（匹配到 Office_Products）+ `item.json` 路径（`index/` 子目录）
9. `MiniOneRec/rq/trainer.py` `delete_file(old_save)` 缺 `[1]`；`utils.py::delete_file` tuple 防御
10. `acsid/generate_sid.py` 显式 `--item_json` + `subprocess.run(check=True)` fail-fast

## 9. 注意事项 / 已知环境陷阱

- 云端 GitHub 访问偶尔不稳（`github.com:443` 偶发 reset，`api.github.com` 正常）→ push 失败重试即可，非仓库问题。
- `requirements.txt` 不 pin torch（避免拉 CUDA/CPU wheel）；继承镜像预装 torch。
- 不要在云端 `pip install --upgrade setuptools`（会撞系统 torch<82 / vllm<80 约束）。
- SID 构造的 RQ-VAE 部署在 `MiniOneRec/rq/`，运行 cwd 必须是 `MiniOneRec/rq/`。
