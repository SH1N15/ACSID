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

核心公式：
```
alpha_i = alpha_max * min(1, log(1+n_i) / log(1+n_ref))   # adaptive
z_i = Normalize[(1-alpha_i) * Normalize(z_text) + alpha_i * Normalize(P(z_cf))]
```
（`Normalize(z_text)` 这一处当前在 `trainer._prepare_input` 中实现 —— **见 §5 待决问题**）

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
  adaptive_fusion.py        # compute_alpha + FusionModule(P + L2norm 加权)
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

## 5. 待决问题（接手核心任务）

### 碰撞率异常高
- **现象**：text 模式 5000 epochs 碰撞率停在 ~0.79（官方 upstream 是 0.004）；200-epoch 版本 level-0 仅用 3/256 个 token；recon loss 已到 0.0000（过拟合）→ 判断为**分布退化**而非欠训。
- **疑似根因**：为三组公平性，当前代码对所有模式（含 text）在 `trainer._prepare_input` 里对 `z_text` 做 `L2-normalize`，把 latent 分布压塌到单位球面小区域。官方 MiniOneRec 直接喂**原始未归一化**文本 embedding。
- **候选方案 A（推荐）**：text 模式保留原始 `z_text`（不 L2norm）；fixed/adaptive 的 text 侧同样保持原始尺度，仅对 `P(z_cf)` 侧与融合后做 L2norm（保留 alpha 语义）。三组仍需一致。需同步改 3 处：
  1. `MiniOneRec/rq/trainer.py::_prepare_input`（text 分支不再 normalize）
  2. `acsid/adaptive_fusion.py::FusionModule.forward`（text 侧去掉头一个 normalize）
  3. `MiniOneRec/rq/generate_indices.py::build_fused_matrix`（text 分支一致）
  并更新 `acsid/README.md` 注明"text 基线是否归一化"。
- **候选方案 B**：text 完全走官方原始路径（连归一化都不加），归一化仅作为 adaptive 特有 —— 公平性变差。
- **候选方案 C**：维持现状（归一化）——碰撞率 0.79 不可接受。

## 6. 云端上次运行状态

- 三模式 RQ-VAE 全量训练被**手动中止**（碰撞率异常，需先决定 §5 再重跑）。
- 依赖/环境已确认可用：torch 2.11 ROCm 可见；缺 scikit-learn 已补；RQ-VAE 能起训练。
- 正确跑法是 `bash acsid_amd/setup_env.sh` → `source .venv-amd/bin/activate` → 从 `MiniOneRec/rq/` 跑 `generate_sid.py`。

## 7. 建议接手第一步

1. `cd /mnt/workspace/ACSID && git pull origin acsid-amd`
2. 读 `acsid_amd/PLAN_AMD.md` §12 与本文 §5，选定 text-normalize 处理方案
3. 按方案改 §5 列出的 3 处文件（+ 文档）
4. 重跑：
   ```bash
   cd MiniOneRec/rq && source ../../.venv-amd/bin/activate
   python ../../acsid/generate_sid.py --dataset Industrial_and_Scientific \
     --epochs 10000 --batch_size 20480 --eval_step 50 --device cuda:0 \
     --modes text fixed adaptive
   ```
5. 碰撞对比：
   ```bash
   python ../../acsid/analyze_collision.py --base ../data/Amazon \
     --dataset Industrial_and_Scientific --include_upstream \
     --out_json ../../experiments/results/collision.json
   ```
   核对 upstream vs text/fixed/adaptive（预期方向 adaptive ≤ fixed ≤ text）。

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
