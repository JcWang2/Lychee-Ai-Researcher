## v2.5.5 — 数据布局治本修复：sample 列规则 + 前缀图片标签（2026-08-08）

交付包 `ai_scientist_execution_layer_v2_20260808_v255`（含 v2.5.1–v2.5.4 全部改动）。

### 治本目标：MLE-Bench prepare 布局怪癖不再启动即崩

21 个任务实测暴露三类布局问题：text-normalization 前缀 zip 表、detecting-insults
无 id sample、dogs-vs-cats 前缀图片标签。本版全部做通用化处理（无竞赛名硬编码、无 if-else 路由）：

1. **本地化前缀表物化（text-normalization en_/ru_ 前缀 zip）**
   - data_layout 新增 `_materialize_localized_tables`：扫描 `<prefix>_train.csv(.zip)`
     及其 `<prefix>_test*` / `<prefix>_sample_submission*` 三件套，自动解压/复制为
     标准 `train.csv / test.csv / sample_submission.csv`，幂等；
   - 任何符合该形态的竞赛（不限 text-normalization）都会自动解析，不新增 if-else。

2. **无 id sample 的目标列推断（detecting-insults 的 `Insult,Date,Comment`）**
   - 目标列 = sample 中**不在 test.csv** 的列；id/透传列 = sample 中**在 test.csv**
     的列。提交按 sample 表头顺序写：透传列原样复制、目标列查预测值，
     无 id sample（如 insults 的 `Comment`）也能正确产出提交，不再出现 `0,0.0,0.0` 全零；
   - program_compiler 5 个模板（tabular + 4 个 image）+ hera/analyzer 统一改为
     `_sample_driven_target` / `_sample_target_columns` / `_id_column`，
     TSV 表头用真实分隔符读取，`id,label` 形态（如 nomad）不受影响。

3. **前缀图片标签合成（cat.0.jpg / dog.1.jpg 形态）**
   - data_layout 新增 `_synthesize_prefix_label_table`，resolver 检测平铺图片目录后
     按文件名前缀合成真实标签 train.csv（≥50 图、2..64 个不同前缀、非纯数字护栏），
     dogs-vs-cats 等任务不再因缺标签表而解析失败。

4. **启动预检（V2_DATA_PREFLIGHT=1）**
   - run 脚本在拉起闭环前用 `resolve_dataset_layout` 预检；`TASK_DATA_OK` 表示
     “布局可解析”，布局不可解析时打印 `TASK_DATA_FAIL` 并跳过该任务，
     不再出现“进程直接退出、无日志”的静默崩溃（`V2_DATA_PREFLIGHT=0` 可关闭）。

5. **离线覆盖**
   - `test_v2_255.py`：71 项断言（SKIP=1 时 64 项）：zip/plain 三件套解析、幂等、
     坏布局抛错、前缀图片真实标签、纯数字不误合成、no-id sample analyzer 规则、
     no-id sample 端到端 harness（提交表头/行数/透传原样/目标数值/oof 标签）、
     三代码文件无竞赛名硬编码。

测试：`python test_v2_255.py` → `RESULT=PASS ok=71 fail=0`（SKIP=1 时 ok=64）；
metrics/contracts/pact/hera/stage_controller/resource_profiler/l1/closed_loop/
23/234/235/236/237/238/239/240/250/251/252/254/255 全套回归 PASS。
安装/运行：见 `A100_DEPLOYMENT.md`；监控：`bash monitor_v2_v255_live.sh`。

---

## v2.1.0 — 声明式方法架构（2026-08-07）

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
