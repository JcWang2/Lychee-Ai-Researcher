

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
