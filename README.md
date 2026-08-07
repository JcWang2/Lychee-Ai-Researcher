# v2.3.3 — 模板编译执行层（Template-Compiled Execution Layer）

交付包 `ai_scientist_execution_layer_v2_20260806_v233`（incremental 包，部署在既有 v2 基座之上）。

## 是什么

v2.3.x 把「每个 child 让 LLM 现场写训练脚本」的执行范式，换成 **HERA 选方法 + 结构化 MethodInvocationV1 + ProgramCompiler 编译出确定性脚本**：

- 训练阶段 **0 次 LLM codegen**：模板编译出的脚本直接在 exec 镜像里跑（tabular/image/text/timeseries 全覆盖）；
- 15k-prompt 的 legacy implementer 仅保留为 **兜底 fallback**（模板修复失败时才用，主路径不再触发）；
- 方法调用可 **bit-for-bit 重放**：seal 时钉住 `code_hash + invocation_hash + template_hash`，实验可审计。

## 数据形态驱动（治本泛化，无竞赛名硬编码）

modality / task_type 全部由**数据内容 + prompt 证据**决定：

| 形态 | 判定依据 | 能力 |
|---|---|---|
| image | 布局解析出图像目录（train/train_images/… 泛化命名）+ magic 校验（JPEG SOF/PNG IHDR 头解析，非只看扩展名）+ 维度探测（≥50 个图像文件或可解析尺寸） | 预缓存 64/128/192/256 四档 + timm embed/finetune |
| text | 内容采样：非数值比例 ≥0.6 +（平均词数≥3 / 空格密集≥50% / 长串≥40%），id/编码列守卫；**v2.3.3：需强散文证据（平均词数≥4 或长串≥40%）或文本列占优** | TF-IDF + LogReg/Ridge/MLP |
| timeseries | 强时序 prompt 或（forecast 等中提示词 + 日期列可被 `datetime.strptime` 解析，纯数字 year/month 不算） | lag/rolling + HistGB + 严格 time_holdout |
| tabular | 无图像/文本证据；**含杂散散文列（如 passenger Name）时按占比/散文强度判定，不误判为 text** | logistic/histgb/mlp/ensemble |

### v2.3.3 关键修复：mixed 数据不再被杂散文本列劫持

- 实测发现：spaceship-titanic 同款表（12 个数值/类别特征 + 1 个 `Name` 列）会被旧规则判成 text，导致 HERA 丢掉 tabular 方法空间；
- 新规则：文本列必须**强散文证据**（平均词数≥4 或 ≥40% 长串）**或占可用特征多数**才算 text；`Name` 这类"空格密集但词少"的弱文本列 → modality 保持 tabular（仍记录在 `text_columns` 供审计）；
- 编译 harness：tabular 模式自动丢弃杂散文本列（避免 sparse 输入喂给 HistGB/MLP 崩溃），timeseries lag 特征循环排除文本列（否则全 NaN lag 会清空所有行）；
- 旧 manifest（v231 恢复场景）无 modality 字段时，烘焙了 TEXT_COLUMNS 即按 text 分支兼容。

### registry 实例隔离（v2.3.2）

- `CapabilityRegistry` 每个实例持有自己的 spec 副本：built-in 的 `broken` 标记只对本实例（daemon 进程）生效，不跨实例污染；
- ephemeral 能力的 `broken` 仍持久化到 `<state_dir>/capabilities/`，重启后依然排除。

## 内置模板

```
tabular.linear.logistic.v1   tabular.gbdt.histgb.v1
tabular.neural.mlp.v1        image.embedding.timm.v1
image.finetune.timm.v1       ensemble.sklearn_soft_vote.v1
text.embedding.tfidf.v1      text.neural.mlp.v1
timeseries.lag_histgb.v1
```

+ `ephemeral.*`（Phase C 合成，run-local，位于 `<state_dir>/capabilities/`）。

## 安装与验证

```bash
tar -xzf ai_scientist_execution_layer_v2_20260806_v233.tar.gz
cd ai_scientist_execution_layer_v2_20260806_v233
sha256sum -c MANIFEST.sha256
export DEPLOY_ROOT=/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2
bash install_v2_execution_layer.sh --target $DEPLOY_ROOT/MLE-bench/agents/aisci --run-tests
# 期望 V2_PACKAGE_MANIFEST=PASS / V2_PYCOMPILE=PASS / V2_OFFLINE_TESTS=PASS / V2_INSTALL_VERIFY=PASS
```

## 离线测试：9 套全 PASS

```
test_v2_metrics.py  test_v2_contracts.py  test_v2_pact.py
test_v2_hera.py     test_v2_stage_controller.py  test_v2_resource_profiler.py
test_v2_l1_transactional.py  test_v2_closed_loop.py  test_v2_23.py
```

- `test_v2_hera.py`：内容级 text 判定、id 编码列保持 tabular、timeseries 日期证据、**mixed 数据（Name 列）保持 tabular**；
- `test_v2_23.py`：proposer 四路径、合成预算、0-LLM materialize、seal hash、bit-for-bit 重放、broken 排除、无竞赛名硬编码、text/timeseries 能力空间与编译端到端、**mixed tabular 渲染 tabular 模式**、**timeseries + 杂散文本列不崩**、builtin broken 不跨实例 / ephemeral broken 跨重启存活。

## 服务器运行

```bash
cd /mnt/data/stage42_delivery/incoming
export LLM_MODEL=qwen3.8-max
export STATE_ROOT=/mnt/data/v2_state
export V2_EXEC_IMAGE=pact-stage42-p8:20260727T112909Z_legacy_l1
export V2_EXEC_PYTHON=/opt/conda/envs/agent/bin/python3
export V2_TORCH_CACHE=/mnt/data/v2_torch_cache
V2_PKG_DIR=ai_scientist_execution_layer_v2_20260806_v233 \
V2_PKG_TAR=ai_scientist_execution_layer_v2_20260806_v233.tar.gz \
V2_LLM_ENV=/mnt/data/stage42_delivery/latest_ai_scientist_v6.env \
  nohup bash run_v2_a100_3tasks_v23.sh smoke > run_v2_smoke_v233_outer.log 2>&1 &
bash monitor_v2_v233_live.sh
# smoke 通过后：
V2_PKG_DIR=ai_scientist_execution_layer_v2_20260806_v233 \
V2_PKG_TAR=ai_scientist_execution_layer_v2_20260806_v233.tar.gz \
V2_LLM_ENV=/mnt/data/stage42_delivery/latest_ai_scientist_v6.env \
  nohup bash run_v2_a100_3tasks_v23.sh 24h-mle > run_v2_24h_mle_v233_outer.log 2>&1 &
```

验收点：日志出现 `COMPILED proposal=... method=... template_hash=...`；`[llm] FAIL role=codegen` 计数为 0；HERA 非空 branch/axis/intent；出现真实 `NEW BEST` 与 submission。详见 `A100_DEPLOYMENT.md` 与 `docs/V2_ARCHITECTURE_V233.md`。
---

## v2.3.8 — 通用结构识别 + RLE/bbox/音频确定性基线（2026-08-07）

交付包 `ai_scientist_execution_layer_v2_20260807_v238`（增量包，v2.3.7 / v2.4.0-M1 之上）。

### 治本目标：不再按竞赛名特判，任何新 case 靠结构信号识别

| 信号（纯内容/形态，hera/analyzer.py） | 判定 | 输出 |
|---|---|---|
| RLE 掩码目标列：目标列抽样 200 个非空值，>=60% 匹配 `\d+( \d+)+`（空/`-` 为合法"无掩码"） | `mask_target` | `task_type=segmentation` → `modality=image_mask` |
| bbox 坐标列集（6 组列名之一完整出现）或 JSON box 列（任意列 >=50% 含 bbox/x/y/w/h 键） | `bbox_columns` | `task_type=detection` → `modality=image_detection` |
| 音频文件树（递归扫描 wav/mp3/flac/ogg/m4a/aac/opus/wma >= 50 个） | `audio_file_count` | `modality=audio` |

覆盖优先级：`image_pixel > image_mask > image_detection > audio`；文本/表格信号不能劫持以上形态。

### 新能力（capability_registry + program_compiler，全部 gpu=False 确定性基线）

- `image.mask.rle.baseline.v1`：空/全图掩码 + RLE 提交（hubmap / tgs / contrails / uw-madison / vesuvius）；
- `image.detection.bbox.baseline.v1`：占位值复用（vinbigdata）/ 多数类+单位框（siim）/ 空串（kuzushiji）；
- `audio.tabular.baseline.v1`：多数类或每类频率（freesound / tensorflow-speech / mlsp）。

### 布局合成（data_layout.py）

无 train/test 表的布局（vesuvius / contrails / tensorflow-speech / mlsp 形状）：目录标签表合成、sample 拷贝表、id-only test.csv；flat 候选守卫防劫持 `prepared/public`；`train_image_level.csv / train_study_level.csv` 与 `train_images.zip / test_images.zip` 支持。

### 指标修复（metrics_registry.py）

`segmentation → dice`、`detection → map_at_k`；未知竞赛按实测 task_type 重推断指标（不再永远停留 accuracy）。

### 覆盖清单（结构识别 + 基线可提交）

hubmap-kidney-segmentation、tgs-salt-identification-challenge、google-research-identify-contrails-reduce-global-warming、vesuvius-challenge-ink-detection、uw-madison-gi-tract-image-segmentation、vinbigdata-chest-xray-abnormalities-detection、kuzushiji-recognition、siim-covid19-detection、freesound-audio-tagging-2019、tensorflow-speech-recognition-challenge、mlsp-2013-birds。

### 诚实边界

3D-object（JSON answers、实际 tabular）、NFL 视频 tracking、序列类任务不在本版承诺内；它们不会被误判为 image_mask/image_detection/audio，但可能仍落入 tabular 兜底。v2.3.8 同时携带 v2.4.0-M1（HERA deep diagnostics + difficulty-ladder prompt wiring）。

### 服务器运行（先装到部署树，再起新 case）

```bash
cd /mnt/data/stage42_delivery/incoming
tar -xzf ai_scientist_execution_layer_v2_20260807_v238.tar.gz
cd ai_scientist_execution_layer_v2_20260807_v238 && sha256sum -c MANIFEST.sha256
export DEPLOY_ROOT=/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2
bash install_v2_execution_layer.sh --target $DEPLOY_ROOT/MLE-bench/agents/aisci --run-tests
# 期望 V2_OFFLINE_TESTS=PASS（含 test_v2_238 / test_v2_240） / V2_INSTALL_VERIFY=PASS

# 新 trio（RLE 掩码 / bbox 检测 / 音频各一个）：
export LLM_MODEL=qwen3.8-max STATE_ROOT=/mnt/data/v2_state_lite \
       V2_EXEC_IMAGE=pact-stage42-p8:20260727T112909Z_legacy_l1 \
       V2_EXEC_PYTHON=/opt/conda/envs/agent/bin/python3 V2_TORCH_CACHE=/mnt/data/v2_torch_cache
nohup bash run_v2_a100_lite_v238.sh > run_v2_lite_v238_outer.log 2>&1 &
bash monitor_v2_v238_live.sh
```

验收点：日志出现 `profile: task_type=segmentation modality=image_mask`、`task_type=detection modality=image_detection`、`modality=audio`；`receipt: verdict=success` 与 `NEW BEST`（确定性基线即可产生合法提交）。

---

## v2.3.9 — HERA 上限能力 + PACT 执行能力（2026-08-07，**不入 git 仓库**）

交付包 `ai_scientist_execution_layer_v2_20260807_v239`（增量包，v2.3.8 之上）。

### 治本目标：图像类分数上限 + 全量泛化，零竞赛名硬编码

| 改动 | 文件 | 说明 |
|---|---|---|
| 预训练权重进容器 | pact/executor.py | 新增 `V2_HF_CACHE` 挂载 `~/.cache/huggingface`（timm pretrained 默认走 HF hub）；preflight 同时探测 torch hub + HF hub 权重，合并进 `pretrained_available` |
| 模板能力上提 | capability_registry.py / program_compiler.py | 新 `image.finetune.timm.v2`（cosine/step LR、flip/rcrop/strong 增强、AMP、H-flip TTA、label smoothing、12 epochs、image_size≤384）与 `image.finetune.ensemble.v1`（≤3 模型 × ≤2 seeds = 4 成员 logits 平均） |
| 提交精度修复 | program_compiler.py（全部图像模板） | 概率输出 `%.9f` + 通用行归一化（clip+renorm），官方 `atol=1e-6` 行和检查必过 |
| 预算上提 | hera/portfolio.py | cached weights + GPU≥24GB 的 image 任务：预算 ×1.35、epochs_max→12；tabular/text 规则不变（测试锁定） |
| 提交自检 | pact/publisher.py | 概率类提交 NaN/负值/行和偏差自动修复写回，回归原样 |
| 官方复评 | v2_closed_loop.py | `_finalize` 尽力用本地 mlebench 复评，`RESULT_SUMMARY.official_grade` 记录分数/奖牌；fail-open 不阻塞 |
| 容器健壮性 | pact/executor.py | `--shm-size=1g`（DataLoader 多 worker） |

详见 `docs/V2_ARCHITECTURE_V239.md`。测试：`python test_v2_239.py`（8 组离线测试，
harness 端到端在服务器执行环境跑）。安装/运行命令见上节 v2.3.8 模板 + 新增 `V2_HF_CACHE`。


---

## v2.5.0 — 声明式方法架构（2026-08-07，**不入 git 仓库**）

交付包 `ai_scientist_execution_layer_v2_20260807_v250`（大版本，v2.3.9 之上）。

### 治本目标：方法/资源/指标规则全部声明化，决策权归还 Analyzer

三层原则：
1. **注册表 = 先验知识库**：capability_registry + `_TEMPLATE_REGISTRY`（15 renderer）+ 成本表 +
   随机基线表 + 指标分发表 + 资源规则表 + 跨任务经验表（`V2_EXPERIENCE_JSON`）；
2. **MethodSelector = 检索器**：按 DatasetContract 元数据过滤 + 成本/经验打分，只返回候选，不决策；
3. **Analyzer/Planner = 最终决策者**：prompt 注入 PRIOR KNOWLEDGE 并明示
   "final research decision is always yours"，可自由选择/组合/创造方法。

| 改动 | 文件 |
|---|---|
| 15 renderer 声明式注册表，`spec.renderer ==` 分支归零 | program_compiler.py |
| 声明式方法选择器（新文件） | method_selector.py |
| 先验注入 planner prompt（不替代决策） | hera/planner.py |
| 指标计算/随机基线/资源规则全查表 | pact/evaluator.py、stage_controller.py、hera/portfolio.py、pact/deterministic.py |
| 8 组 83 项离线断言（AST 扫描禁止 if/else 路由） | test_v2_250.py |

测试：`python test_v2_250.py` → `RESULT=PASS ok=83 fail=0`；v239 及前序 16 套回归全 PASS。
安装/运行命令见上节 v2.3.9 模板（`V2_PKG_DIR=...v250`）。详见 `docs/V2_ARCHITECTURE_V250.md`。
