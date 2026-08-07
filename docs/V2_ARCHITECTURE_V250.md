# V2.5.0 — 声明式方法架构（Declarative Method Architecture，全量 MLE-Bench 治本泛化）

> 本版本 **不入 git 仓库**：只构建交付包 + 部署脚本，用于服务器重测后验收。
> v2.4 已被另一会话占位；本会话做大版本 v2.5（v2.5.0）。

## 一、为什么做这版

v2.3.x 已经做到「HERA 选方法 + ProgramCompiler 确定性编译」，但方法选择与资源/指标/基线
规则仍散落在 `if/elif` 链里。每加一种模态、一个指标、一条经验，就要改代码——这就是
"按任务打补丁"的根源，也无法真正冲 SOTA：

1. **方法路由靠分支**：compiler 里 `spec.renderer == "xxx"` 的 15 段 if/elif、evaluator 里按
   指标名的 if/elif、stage_controller 里按 metric 的基线分支、portfolio 里按模态的预算分支；
2. **先验知识无处安放**：跨任务经验（"这类数据用什么方法通常更好"）没有数据载体，每次
   都靠 LLM 从头猜；
3. **最终决策权模糊**：注册表/调度器偶尔会"替"分析器做决定，违背 AI 自主科研原则。

v2.5.0 的答案是：**一切方法/资源/指标规则声明化（注册表 + 元数据表），检索与决策分离**。

## 二、三层原则（冻结契约）

```
┌─────────────────────────────────────────────────────────────┐
│  第 3 层  Analyzer / Planner（HERA LLM）                      │
│         最终决策权：自由选择 / 组合 / 创造方法，可拒绝一切先验 │
│         prompt 中明示 "final research decision is always yours"│
├─────────────────────────────────────────────────────────────┤
│  第 2 层  MethodSelector（method_selector.py）                │
│         只做检索与排序：元数据兼容过滤 + 成本表 + 经验先验打分 │
│         返回候选列表（ScoredMethod），永远不"选"             │
├─────────────────────────────────────────────────────────────┤
│  第 1 层  注册表 / 数据表（先验知识库）                       │
│         capability_registry：方法元数据（模态/任务/指标/参数/  │
│           成本模型/GPU 要求）                                 │
│         _TEMPLATE_REGISTRY：15 个 renderer 声明式条目          │
│         _COST_MODELS / _RANDOM_BASELINE_KIND / 指标分发表 /   │
│           资源规则表 / ExperienceTable（跨任务经验 JSON）      │
└─────────────────────────────────────────────────────────────┘
```

**纪律（写进测试）**：
- 禁止 `if/elif` 直接指定方法、模态、指标、资源规则（`test_v2_250.py` 用 AST 扫描锁定）；
- 新方法 = 注册表加一条目；新知识 = 经验表加一行；新模态 = 词表加一个词 + 注册表条目；
- 任何代码路径不得读取竞赛名做路由（`metrics_registry.py` 是唯一的官方数据表）。

## 三、本版改造清单

| 文件 | 改动 | 说明 |
|---|---|---|
| `program_compiler.py` | `_TEMPLATE_REGISTRY` 声明式注册表 | 15 个 renderer 全部声明式；`spec.renderer ==` 分支数 = 0；`TEMPLATE_SCHEMA="v250_template_v1"` |
| `method_selector.py`（新） | `DatasetContract` / `ExperienceTable` / `MethodSelector` | 元数据过滤 + `_COST_MODELS` 成本表 + 经验先验打分；无值分支；`V2_EXPERIENCE_JSON` 可持久化跨任务经验 |
| `hera/planner.py` | `prior_block()` 注入 prompt | 把候选 + 先验分数作为 **PRIOR KNOWLEDGE** 注入，明示"仅供参考，最终决策权在你"；`_has_pretrained()` 兼容 count/list 两种资源表示 |
| `pact/evaluator.py` | `_COMPUTE_HANDLERS` / `_PROBABILITY_HANDLERS` | 指标计算全查表，metric 名 if/elif 归零 |
| `stage_controller.py` | `_RANDOM_BASELINE_KIND` | 随机基线按 metric → kind（zero/half/one_over_k/log_k/target_scale）查表 |
| `hera/portfolio.py` | 资源规则全表化 | `_MODALITY_BASE_RESOURCE` / `_BATCH_GPU_TABLE` / `_EPOCHS_BONUS_*` / `_SEED_MODEL_BY_TASK`；模态分支归零 |
| `pact/deterministic.py` | `_BINARY_POS_FREQ_METRICS` | binary_logloss 正类频次基线查表 |
| `test_v2_250.py`（新） | 8 组 83 项离线断言 | 无竞赛名 / 无 renderer 分支 / selector 声明式 / evaluator 查表 / stage 查表 / registry 完整性+确定性渲染 / planner 先验注入 / portfolio 表 |

## 四、决策链路（数据流）

1. Analyzer 产出 `AnalysisProfile`（数据形态 + 指标，无竞赛名）；
2. Planner 构造 `DatasetContract`（模态/任务/指标族/行数/类别数/GPU/预算/预训练/缓存）；
3. `MethodSelector.candidates(contract)` 返回排序候选（成本 fit + 经验先验，**不决策**）；
4. `prior_block()` 把前 6 名候选写进 prompt，标注 PRIOR KNOWLEDGE；
5. LLM（Analyzer/Planner）自由选择/组合/创造方法 → `MethodInvocationV1`；
6. ProgramCompiler 按注册表条目确定性渲染 → PACT 执行/冻结/回读。

## 五、测试

```bash
cd <payload>/agents/aisci
python test_v2_250.py          # RESULT=PASS ok=83 fail=0
# 全量回归（v239 及其前序全部套件，确保声明式改造不破坏行为）：
python test_v2_metrics.py && python test_v2_contracts.py && \
python test_v2_resource_profiler.py && python test_v2_hera.py && \
python test_v2_stage_controller.py && python test_v2_l1_transactional.py && \
python test_v2_23.py && python test_v2_234.py && python test_v2_235.py && \
python test_v2_236.py && python test_v2_237.py && python test_v2_238.py && \
python test_v2_239.py && python test_v2_240.py && python test_v2_pact.py && \
python test_v2_closed_loop.py
```

验收：17 套全 PASS（16 套回归 + v250 新增）。

## 六、经验表（可选，治本闭环）

`MethodSelector` 支持 `V2_EXPERIENCE_JSON` 指向一个跨任务经验文件：
键 = `modality|metric_family|scale_bucket|method_family`，值为 `{n, mean_lift, mean_cost_ratio}`。
服务器每次 trial 结束后可回写该表，下一轮候选排序自动带上数据驱动的先验——
**先验只影响排序，不锁定选择**。

## 七、服务器运行（同 v2.3.9 模板）

```bash
export LLM_MODEL=qwen3.8-max STATE_ROOT=/mnt/data/v2_state_lite \
       V2_EXEC_IMAGE=pact-stage42-p8:20260727T112909Z_legacy_l1 \
       V2_EXEC_PYTHON=/opt/conda/envs/agent/bin/python3 V2_TORCH_CACHE=/mnt/data/v2_torch_cache
export V2_PKG_DIR=ai_scientist_execution_layer_v2_20260807_v250 \
       V2_PKG_TAR=ai_scientist_execution_layer_v2_20260807_v250.tar.gz
nohup bash run_v2_a100_lite_v250.sh 24h-mle > run_v2_lite_v250_outer.log 2>&1 &
```

验收：`V2_INSTALL_VERIFY=PASS`（含 `test_v2_250.py`）；日志出现 `profile:` 后
`grant:` → `receipt: verdict=success` → `NEW BEST`。
