# V2.3.6 — 逐指标最小提升门槛（Per-Metric min_delta）+ 恢复一致性

## 根因（真实线上观测）
旧 24h 图像 run 出现“指标明明更高却不 NEW BEST”：

```
[05:54:15] NEW BEST: 0.9971914860384965        (aerial, round 1)
...之后 receipt: verdict=success metric=0.9997 ...
但 best 永远停在 0.9971840584934031
```

全系统硬编码 `min_delta = 0.01`（绝对量）：
- `v2_closed_loop._is_better`：higher 需 `candidate > incumbent + 0.01`
- `pact/host_supervisor.py`：`self.min_delta = 0.01`
- `pact/promotion.py`：`min_delta: float = 0.01`
- `pact/verifier.py`：`metric > best_metric + 0.01`

对 [0,1] 有界、越高越好指标（AUC/QWK/accuracy）是灾难性的：
aerial `0.9997 - 0.99718 = 0.00252 < 0.01` → 永远无法触发 NEW BEST；
aptos（best=0.8919）需要 >0.9019 才算提升，几乎不可能。
logloss/RMSE 的小步真实提升同样被压住。

为什么 receipt 却显示 success：`host_supervisor._verdict(metric, best_child)` 的
`best_child` 是 **grant 内**第一个 child 为 None → “First verified trial” success；
全局判定在 closed loop `_absorb_receipt -> _is_better`，被 0.01 卡死。
副作用：`incumbent_best.json` / `best_code_*.py` 停在旧代码——指标新、代码旧。

## 修复：门槛是 metric family 的属性，不是 run 的属性

`metrics_registry.py` 新增 `METRIC_MIN_DELTA` + `DEFAULT_MIN_DELTA`：

| 族 | min_delta | 理由 |
|---|---|---|
| auc / accuracy / f1_* / mcc / qwk / mean_auc_multilabel / spearman / pearson / kendall_tau / dice / iou_mean / map_at_k / label_ranking_ap / jaccard / mean_angular_error | 1e-4 | 有界/分数型指标，1e-4 即真实可观测提升 |
| logloss / binary_logloss / weighted_logloss / kl_div | 1e-4 | 概率空间损失，量级 ~0.3..2.0 |
| rmse / mae / log_mae / rmsle / haversine / levenshtein | 1e-3 | 无界误差指标，低于 1e-3 视为噪声 |
| 未知族 | 0.01 | 保持旧行为，不冒险 |

接入点（四处判定全部统一走 spec 的 `min_delta`）：
1. `get_metric_spec()` / `infer_metric_spec()` 返回 `min_delta` 字段；
   `apply_metric_to_profile()` 写 `profile.metric_min_delta`。
2. `v2_closed_loop._is_better` 用 `self.metric_spec.get("min_delta", 0.01)`；
   manifest 写入 `metric_min_delta`。
3. `HostSupervisorService(metric_min_delta=...)`、`PromotionManager(min_delta=...)`
   （closed loop 与 v2_host_daemon 都由 spec/manifest 传入）。
4. `Verifier(min_delta=...)`（默认 0.01，兼容旧测试/旧调用方）。

## 恢复一致性（重启后指标与代码不漂移）

`_recover_scientific_state` 现在恢复 **certified 指针与 ledger 中较好者**：
- 新增 `_ledger_best_record()`（方向感知，只认 rc==0 的 verified 记录）。
- ledger best 严格优于 certified best（旧门槛拒绝过真实提升的形态）→
  `best_metric = ledger_best`、`best_receipt_id = ledger trial_id`，
  并调用 `_sync_incumbent_from_ledger()`：
  - 通过 host receipt store（`pact_control_host/.../receipts/receipt_*.json`）
    把 `receipt_id -> spec_id`，取 `ws_code/trial_<spec_id>.py`；
  - 兜底：按 `code_hash` 内容匹配 `ws_code/trial_*.py`；
  - 命中则 `_save_incumbent` 落盘 `incumbent_best.json` + `best_code_<round>.py`。
- 非致命：找不到代码只记日志，不阻塞启动。

## 回归测试
新增 `test_v2_236.py`（6 组断言）：
- registry：auc/qwk/logloss=1e-4、mae/rmse=1e-3、未知族 0.01、infer 分支。
- `_is_better` 生产形态：aerial 0.9997>0.99718 True、aptos 0.896>0.8919 True、
  logloss 0.3749<0.37517 True、rmse 1.0<1.005 True，且 sub-delta 全部 False。
- host_supervisor / promotion / verifier 同门槛一致性 + 默认 0.01 旧行为保留。
- 恢复一致性：certified=0.99718 + ledger=0.9997 → 恢复 0.9997 且 incumbent
  代码资产同步为 ledger 对应代码。

离线套件：metrics / contracts / pact / hera / stage_controller / resource_profiler /
l1_transactional / closed_loop / v2_23 / v2_234 / v2_235 / v2_236 全 PASS。

## 部署
`install_v2_execution_layer.sh --target ... --run-tests` → `run_v2_a100_lite_v236.sh`。
24h run 重启（新 STAMP）即吃到新逻辑：0.99x AUC 的小步提升也能 NEW BEST。