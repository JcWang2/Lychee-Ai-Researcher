# V2.3.9 — HERA 上限能力 + PACT 执行能力（全量 MLE-Bench 治本泛化）

> 本版本**不入 git 仓库**：只构建交付包 + 部署脚本，用于三个图像任务重测。

## 一、为什么做这版

三个图像任务（aerial / aptos / dog-breed）官方复评显示内部 best 与官方分有差距，
根因不是 timeout 不够，而是执行范式上限低：

1. **预训练权重根本没有进容器**：executor 只挂载 `torch_cache → /root/.cache/torch`，
   而 timm 的 `pretrained=True` 默认从 **HuggingFace hub 缓存**（`~/.cache/huggingface/hub`）
   加载 → 容器里永远是 from-scratch 训练 → 图像分上不去；
2. **模板能力薄**：旧 finetune 模板无 LR schedule / 数据增强（flip 是空操作）/ AMP /
   TTA / label smoothing / 集成，epochs 上限 10 但预算常被压到 probe 级；
3. **提交格式毒**：概率行和偏差 >1e-6 会被官方 atol 检查拒绝（dog-breed 实测
   `row sums differ by 1.2e-5`），旧模板只写 `%.6f`、不做行归一化。

全部按**数据形态/指标族驱动**修复，**零竞赛名硬编码**——三个图像任务只是验收样本，
改动对全量 MLE-Bench 的 image/text/tabular/mask/detection/audio/pixel 都通用。

## 二、HERA 上限能力

### 新能力（capability_registry + program_compiler）

| method_id | renderer | 能力 |
|---|---|---|
| `image.finetune.timm.v2` | `image_finetune_timm_v2` | image_size 64..384、epochs 2..12、`lr_schedule=cosine/step`、`augment=flip/rcrop/strong/none`（flip 真正生效 + strong=裁剪/翻转/色彩抖动）、AMP/autocast、AdamW、CosineAnnealingLR、H-flip TTA（推理 logits 平均）、label_smoothing 0..0.3 |
| `image.finetune.ensemble.v1` | `image_finetune_ensemble` | `model_names`(≤3) × `seeds`(≤2)，最多 4 个成员，logits 平均后 softmax；同样带 TTA / AMP / 精度修复 |

**旧模板统一修复**（`image.embedding.timm.v1` / `image.finetune.timm.v1`）：
概率输出 `%.6f → %.9f`，多分类概率行统一 `clip[0,1] + renormalize`（`_norm_proba_row`），
与 v2 模板一致，官方 `atol=1e-6` 行和检查必过。

### 预训练权重真正可用（PACT executor）

- `V2_HF_CACHE`（默认 `<torch_cache 同级>/v2_hf_cache`）挂载为容器
  `~/.cache/huggingface`；`torch_cache` 继续挂 `~/.cache/torch`；
- preflight 同时 glob `~/.cache/torch/hub/checkpoints/*` **和**
  `~/.cache/huggingface/hub/models--*`，合并进 `pretrained_available` 白名单 →
  `_cached_weights_seen` → ResourceProfiler 看到真实权重数；
- 容器统一 `--shm-size=1g`（DataLoader 多 worker 场景防共享内存爆）。

### 资源预算上提（hera/portfolio.py，仅模态驱动）

- `cached_weights > 0 且 GPU ≥ 24GB` 的 **image** 任务：`max_budget × 1.35`（封顶 7200s）、
  `epochs_max` 提到 12、`cached_factor` 上限 1.45；
- tabular/text 的 epochs 与预算规则**不动**（已用测试锁住，杜绝误伤其他模态）。

## 三、PACT 执行能力

### 提交自检（pact/publisher.py）

`publish_certified` 拷贝后做**形状驱动的概率行自检**（不读竞赛名）：
- 检测：多数值列 + 采样行和≈1（≥60% 行在 5% 容差内）；
- 修复：NaN→0、负值→0、>1→1、`|行和-1|>1e-8` 则重新归一化，`%.9f` 写回；
- 回归/分类输出原样交付；无 certified 记录仍 no-op。

### 官方复评收尾（v2_closed_loop.py `_finalize`）

- 有 submission 时尽力调用本地 `mlebench.grade.grade_csv`（控制环境有 mlebench +
  prepared 数据时），把 `score / 阈值 / 奖牌` 写进 `run_report.json` 与 `RESULT_SUMMARY.official_grade`；
- **fail-open**：环境没有 mlebench / 数据未 prepare → `status=skipped`，绝不阻塞 run。

## 四、验证

`python test_v2_239.py`（已接入 `install_v2_execution_layer.sh` 离线测试列表）：

1. registry：v2/ensemble 能力存在且 schema 含新参数（image_size≤384、epochs≤12、list 上限）；
2. compiler：两个新模板 + 旧 embed 模板确定性渲染（`%.9f` / `_norm_proba_row` /
   `import torch.nn as nn` / 无残留 token），tabular 拒绝 image 方法；
3. resource：cached weights + 大 GPU 才提升 image epochs/预算；tabular 不受影响；
4. executor：`--shm-size=1g` + HF/torch 双挂载 + preflight HF glob 与合并解析；
5. publisher：毒提交（负值/NaN/行和 1.5）自动修复为行和 1.0、9 位小数；回归不动；
6. finalize：无 submission / mlebench 缺失 → `skipped`；成功 → `status=ok` 带分数；
7. harness 端到端：合成小图跑 finetune.v2 与 ensemble（本地无 timm/torchvision 时 SKIP，
   服务器执行环境必跑）。

## 五、服务器验收（三个图像任务重测）

```bash
cd /mnt/data/stage42_delivery/incoming
tar -xzf ai_scientist_execution_layer_v2_20260807_v239.tar.gz && sha256sum -c ai_scientist_execution_layer_v2_20260807_v239.tar.gz.sha256
cd ai_scientist_execution_layer_v2_20260807_v239
export DEPLOY_ROOT=/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2
bash install_v2_execution_layer.sh --target $DEPLOY_ROOT/MLE-bench/agents/aisci --run-tests
# 期望 V2_OFFLINE_TESTS=PASS（含 test_v2_239） / V2_INSTALL_VERIFY=PASS

# 新 trio = 三个图像任务（run_v2_a100_lite_v239.sh 默认）
export LLM_MODEL=qwen3.8-max STATE_ROOT=/mnt/data/v2_state_lite \
       V2_EXEC_IMAGE=pact-stage42-p8:20260727T112909Z_legacy_l1 \
       V2_EXEC_PYTHON=/opt/conda/envs/agent/bin/python3 \
       V2_TORCH_CACHE=/mnt/data/v2_torch_cache V2_HF_CACHE=/mnt/data/v2_hf_cache
nohup bash run_v2_a100_lite_v239.sh > run_v2_lite_v239_outer.log 2>&1 &
bash monitor_v2_v239_live.sh
```

验收点：
- `preflight: ... PREFLIGHT_HF_PRETRAINED=...` / `pretrained cache: N weights`（N 含 HF 权重）；
- `receipt: verdict=success`、`NEW BEST`、RESULT_SUMMARY 含 `official_grade.status=ok`（或 skipped + 原因）；
- 官方复评分数较 v2.3.8 提升（aptos 目标 bronze+、dog-breed 目标 logloss 显著下降）。
