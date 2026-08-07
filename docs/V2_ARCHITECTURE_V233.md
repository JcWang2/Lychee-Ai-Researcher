# V2.3.3 架构说明：从「LLM 写脚本」到「模板编译执行」+ 数据形态驱动泛化

## 1. 问题：v2.2 的执行范式太贵

v2.2 每个 child 都由 LLM 现场写训练脚本（15k prompt，流式 codegen）：

- 每个 child 3–10 分钟 LLM 生成时间，24h 实际只能跑 ~16 个 grant/任务；
- 生成质量不稳定，失败后进入静默/重试，研究节奏被打断；
- 脚本不可重放，实验无法 bit-for-bit 审计。

v2.3.x 改为**决策与执行分离**：

```
HERA 研究决策（LLM）                    执行层（确定性编译，0 次 LLM）
─────────────────────────              ─────────────────────────────
选 method_id + params                    ProgramCompiler.validate()
选 preprocessing / axis                  ProgramCompiler.normalize()
选 research_intent                       ProgramCompiler.render()
（可选）Phase C 能力合成                 ProgramCompiler.patch_params()
                                        ProgramCompiler.render_ephemeral()
```

## 2. 数据流

1. `_agent_proposer`（v2_closed_loop.py）四路径兜底：
   - 合法 LLM 响应 → 编译通过 → grant + seal（钉 `code_hash + invocation_hash + template_hash`）；
   - 未知/非法/崩溃 → 确定性修复（`COMPILE_PATCH`），不重走 codegen；
   - capability registry 无匹配 → Phase C 合成（`SYNTHESIS`，`MAX_SYNTHESIS_ACTIONS` 默认 2）；
   - 全部失败 → legacy implementer 兜底（旧 15k prompt 路径，仅此场景触发）。
2. HostSupervisor 收到编译后的 TrialSpec → `COMPILED proposal=... method=... template_hash=...`（0 次 LLM）→ 模板编译脚本在 exec 镜像执行。
3. 结果写入 outcomes → receipt → ledger，`experiment_ledger.jsonl` 全程可审计。

## 3. 数据形态判定（全部内容驱动、无竞赛名）

`hera/analyzer.py` 从 manifest + 实际数据采样构造 `AnalysisProfile`：

### image（可直接识别 + 预缓存）
- `data_layout.resolve_dataset_layout` 按泛化目录名（train/train_images/images_train/train_imgs/images…）解析图像目录，覆盖 flat / class 子目录 / 深层嵌套；
- `iter_image_files` 做 **magic 校验**（读文件头解析 JPEG SOF / PNG IHDR，不只靠扩展名）；
- `_probe_image_dims` 扫描最多 1MB 找 SOF 标记（EXIF/ICC 后置也能解析）；探测失败回退 64×64 默认；
- modality=image 条件：可解析尺寸 **或** magic 校验文件数 ≥50（`IMAGE_FILE_MODALITY_THRESHOLD`）；
- 预缓存：`pact/data_cache.ensure_image_caches` 一次 docker pass 建 64/128/192/256 四档（大尺寸解码一次、内存下采样），内容 key + 行数校验防 stale，失败非致命，模板内 cache unreadable → raw decode 兜底。

### text（内容证据 + 强散文/占比规则）
- `_column_text_signal`：采样 30 行，非数值比例 ≥0.6 且（平均词数≥3 / 空格密集≥50% / 长串≥40%）；近唯一短 token（id/编码/哈希）守卫；
- 列名 hint 只降门槛，**内容证据必需**（`description`/`title` 不再误判）；
- **v2.3.3 mixed 规则**：`Name` 这类"空格密集但平均词数低（≈2）"的弱散文列，只有在**强散文证据**（某文本列平均词数≥4 或长串≥40%）**或文本列占可用特征多数**时才判 text；否则 modality=tabular（文本列仍记录，供审计与 HERA 参考）；
- id 列识别：sample_submission 首列 → id/index 命名 → 否则留空（不把表头首列误当 id）。

### timeseries（prompt + 日期证据）
- `time_column`：列名 hint（date/datetime/timestamp/time/month/year/…）+ `datetime.strptime` 解析，纯数字 year/month 先被 float 解析排除；
- task_type：强提示词（timeseries/time series/predict next/temporal）或（forecast/future sales/future demand + 日期列证据）；纯日期列不触发。

### tabular（默认 + metric 全覆盖）
- 无图像/文本证据时回退 tabular；`metrics_registry.py` 覆盖全部 MLE-Bench 竞赛的 metric（exact/proxy/inferred），未知竞赛自动推导，**不是决策硬编码**。

## 4. 模板与能力注册表

内置模板（`program_compiler.py` 由 `make_compiler.py` + `parts/*.txt` 构建）：

| method_id | 适用 | 说明 |
|---|---|---|
| tabular.linear.logistic.v1 | 表格/时序兜底 | 逻辑回归/岭回归，含缺失/类别处理 |
| tabular.gbdt.histgb.v1 | 表格/时序兜底 | HistGradientBoosting |
| tabular.neural.mlp.v1 | 表格/时序兜底 | sklearn MLP |
| image.embedding.timm.v1 | 图像 | timm 主干 + embedding + 头部，权重走本地缓存；缓存缺失→raw decode 兜底 |
| image.finetune.timm.v1 | 图像 | timm 主干 finetune；缓存缺失→raw decode 兜底 |
| ensemble.sklearn_soft_vote.v1 | 通用 | 已有模型软投票集成 |
| text.embedding.tfidf.v1 | 文本 | 逐列 TfidfVectorizer + scipy 稀疏拼接 + LogReg/Ridge |
| text.neural.mlp.v1 | 文本 | 逐列 TF-IDF + MLP（_DenseAdapter 转稠密） |
| timeseries.lag_histgb.v1 | 时序 | 按日期排序，lag{1,2,3,max}/rmean/rstd + HistGB；严格 time_holdout |

共享 harness（`parts/tabular_harness.txt`）：

- `MODALITY` token 烘焙进编译脚本：text 分支只在 modality=text 启用；**tabular 模式自动丢弃杂散文本列**（`log("tabular mode: dropping ...")`），避免 sparse 输入喂给 HistGB/MLP；
- lag 分支：`LAG_COLUMN` 排序，特征列遍历排除 target/id/date/**text 列**（否则全 NaN lag 会清空所有行），无特征列时仅 ordinal 列；强制 `VALIDATION="time_holdout"`；
- 旧 manifest 无 modality 字段（v231 恢复）时，烘焙了 TEXT_COLUMNS 即默认 text 分支，恢复行为不回退；
- `_run_cv` 支持 `time_holdout`：按时间序最后 25% 验证。

`capability_registry.py`：

- 内置 9 个能力，metric_outputs 全量覆盖 MLE-Bench；
- `set_broken`：失败模板标记 broken，后续 grant 自动排除；
- **实例隔离**：`__init__` 为每个实例持有 spec 副本，built-in broken 只对本实例生效；ephemeral broken 仍写入 `<state_dir>/capabilities/ephemeral_specs.json`，重启后继续排除；
- **无竞赛名硬编码**：只按 modality / task / metric / 资源画像过滤。

## 5. Phase C：能力合成

- 触发：HERA 选出的方法不在 registry、且无兼容能力可用；
- 预算：`MAX_SYNTHESIS_ACTIONS`（默认 2）每任务持久化于 `<state_dir>/capabilities/synthesis_usage.json`，重启不超发；
- 产物：LLM 生成 sklearn 适配器（必须定义 `build_model`），注册为 `ephemeral.*`，后续 grant 可直接复用（参数可 patch，不重复生成）。

## 6. 预算语义（硬约束）

| 预算 | 值 | 权威 |
|---|---|---|
| MAX_GRANTS | 128 | 外层研究决策次数上限（安全上限，非 KPI） |
| MAX_TOTAL_TRIALS | 256 | child trial 总数上限 |
| TOTAL_WALL_CLOCK | 86400s | **最高权威**，到点即收 |

child 数随研究意图动态决定：feasibility/repair 1–2、cheap probe 2–4、local exploitation 2–3、expensive structural 1–2、confirmation 2–3、final training 1。

## 7. 崩溃恢复语义

- seal 时钉住三个 hash，重启后按 receipt 恢复 scientific state（`restored scientific state: best=... round=...`）；
- 预算不超发：committed / reserved 与 kill 前一致；
- stage 不回落（`stage_state.json` 持久化）；
- 已编译 spec（invocation_hash 已设）走确定性修复，不重新 codegen。

## 8. 测试

- `test_v2_hera.py`：内容级 text 判定、id 编码列保持 tabular、timeseries 日期证据、**mixed 数据（Name 列）保持 tabular + 记录 text 列**；
- `test_v2_23.py`：proposer 四路径、合成预算、0-LLM materialize、seal hash、bit-for-bit 重放、patch 减半、broken 排除、无竞赛名硬编码、text/timeseries 能力空间与编译端到端、**mixed tabular 渲染 tabular 模式**、**timeseries + 杂散文本列不崩**、builtin broken 不跨实例 / ephemeral broken 跨重启存活；
- 其余基线（metrics/contracts/pact/stage_controller/resource_profiler/l1_transactional/closed_loop）全 PASS；
- 安装脚本 `--run-tests` 串行执行 9 套，任一失败即 FAIL。

## 9. 关键文件

```
v2_contracts.py        MethodInvocationV1 / TrialSpec / seal_record（三 hash）+ text_columns/time_column
hera/analyzer.py       内容驱动的 modality/text_columns/time_column/task_type 判定 + mixed 占比/散文强度规则
program_compiler.py    validate / normalize / render / patch_params / render_ephemeral（MODALITY token）
capability_registry.py 能力过滤 + broken 持久化 + 合成预算 + 实例隔离
v2_closed_loop.py      _agent_proposer 四路径 + Phase C 合成 + 恢复语义
pact/host_supervisor.py COMPILED 主路径 + legacy implementer 兜底
v2_host_daemon.py      registry + compiler 装配注入
```

## 10. 已知边界（不做误判承诺）

- modality 判定依赖采样内容与 prompt 证据；zip/tar 内嵌图片、无列 CSV、纯二进制等极端布局仍可能落到 tabular 兜底或 Phase C 合成；
- text 方法按词级 TF-IDF（英文/分词文本友好）；中文等无空格语言若无分词，效果依赖原始列文本的 n-gram；
- 弱散文列（如 Name）在 tabular 数据中被**丢弃**（不进模型特征），只保留审计记录；
- timeseries 方法要求显式日期/时间列可解析；无日期列的纯序列任务走 tabular 兜底。