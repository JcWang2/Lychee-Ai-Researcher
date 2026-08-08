## v2.5.5 — 泛化数据布局修复（2026-08-08）

交付包 `ai_scientist_execution_layer_v2_20260808_v255`（含 v2.5.1–v2.5.4 全部改动）。

### 治本目标：MLE-Bench prepare 布局怪癖不再启动即崩

text-normalization（en_/ru_ 前缀 + zip 表）等任务此前在 `resolve_dataset_layout`
阶段直接抛 `DatasetLayoutError`，闭环进程静默退出、无日志可查。本版做通用化处理：

1. **本地化前缀表物化（前缀无关，无竞赛名硬编码）**
   - data_layout 新增 `_materialize_localized_tables`：扫描 `<prefix>_train.csv(.zip)`
     及其 `<prefix>_test*` / `<prefix>_sample_submission*` 三件套，自动解压/复制为
     标准 `train.csv / test.csv / sample_submission.csv`，幂等；
   - 任何符合该形态的竞赛（不限 text-normalization）都会自动解析，不新增 if-else。

2. **启动预检（V2_DATA_PREFLIGHT=1）**
   - run 脚本在拉起闭环前用 `resolve_dataset_layout` 预检；`TASK_DATA_OK` 现在表示
     “布局可解析”；布局不可解析时打印 `TASK_DATA_FAIL` 并跳过该任务，
     不再出现“进程直接退出、无日志”的静默崩溃（`V2_DATA_PREFLIGHT=0` 可关闭）。

3. **离线覆盖**
   - `test_v2_255.py`：18 项断言（zip/plain 三件套解析、幂等、坏布局抛错、
     代码无竞赛名硬编码）。

测试：`python test_v2_255.py` → `RESULT=PASS ok=18 fail=0`；本机完整 install 验证 →
`V2_PACKAGE_MANIFEST=PASS / V2_PYCOMPILE=PASS / V2_OFFLINE_TESTS=PASS / V2_INSTALL_VERIFY=PASS`。
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
