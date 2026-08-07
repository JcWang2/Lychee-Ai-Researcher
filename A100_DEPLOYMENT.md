# A100 部署与运行手册（v2.3.3）
## 0. 前置环境

- 部署根：`/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2`
- exec 镜像：`pact-stage42-p8:20260727T112909Z_legacy_l1`
- 数据：`/mnt/data/mle-bench/data/<competition>`
- torch 缓存：`/mnt/data/v2_torch_cache`（已预下载，HF mirror 环境变量已就绪）
- LLM：`qwen3.8-max`（DashScope 兼容端点），env 文件 `/mnt/data/stage42_delivery/latest_ai_scientist_v6.env`；**文件里没有 LLM_MODEL，必须显式导出**

## 1. 上传（Windows 工作站）

```powershell
cd C:\Users\Administrator\Documents\Ai科学家\deliveries
powershell -ExecutionPolicy Bypass -File upload_v2_a100_v233.ps1
# 期望 UPLOAD_VERIFIED=YES（远端校验 sha256 + bash -n）
```

## 2. 解包 + 安装 + 离线测试

```bash
cd /mnt/data/stage42_delivery/incoming
tar -xzf ai_scientist_execution_layer_v2_20260806_v233.tar.gz
cd ai_scientist_execution_layer_v2_20260806_v233
sha256sum -c MANIFEST.sha256
export DEPLOY_ROOT=/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2
bash install_v2_execution_layer.sh --target $DEPLOY_ROOT/MLE-bench/agents/aisci --run-tests
# 期望 V2_PACKAGE_MANIFEST=PASS / V2_PYCOMPILE=PASS / V2_OFFLINE_TESTS=PASS / V2_INSTALL_VERIFY=PASS
```

## 3. 清理旧 v2 进程（如有）

```bash
pkill -TERM -f "run_v2_a100_3tasks" 2>/dev/null; sleep 5
pkill -TERM -f "v2_closed_loop.py" 2>/dev/null; sleep 5
pkill -9 -f "v2_closed_loop.py" 2>/dev/null
pkill -9 -f "v2_host_daemon.py" 2>/dev/null
docker ps --format '{{.Names}}' | grep '^v2_' | xargs -r -n1 docker rm -f
ps aux | grep -E "v2_closed_loop|v2_host_daemon|run_v2_a100_3tasks" | grep -v grep || echo ALL_V2_STOPPED
```

## 4. smoke（必须先过再跑 24h）

```bash
cd /mnt/data/stage42_delivery/incoming
export LLM_MODEL=qwen3.8-max        # 必须显式导出（env 文件里没有）
export STATE_ROOT=/mnt/data/v2_state
export V2_EXEC_IMAGE=pact-stage42-p8:20260727T112909Z_legacy_l1
export V2_EXEC_PYTHON=/opt/conda/envs/agent/bin/python3
export V2_TORCH_CACHE=/mnt/data/v2_torch_cache
V2_PKG_DIR=ai_scientist_execution_layer_v2_20260806_v233 \
V2_PKG_TAR=ai_scientist_execution_layer_v2_20260806_v233.tar.gz \
V2_LLM_ENV=/mnt/data/stage42_delivery/latest_ai_scientist_v6.env \
  nohup bash run_v2_a100_3tasks_v23.sh smoke > run_v2_smoke_v233_outer.log 2>&1 &
bash monitor_v2_v233_live.sh
```

smoke 验收：
1. `grep LLM_STATUS run_v2_smoke_v233_outer.log` → `READY model=qwen3.8-max`
2. 日志出现 `COMPILED proposal=... method=... template_hash=...`（模板编译主路径，0 次 LLM codegen）
3. `grep -cE "\[llm\] FAIL role=codegen" run_v2_*_<STAMP>.log` = 0（legacy 只作兜底）
4. `grant: ... branch=<HERA 选择> axis=... intent=...` 非空、非 baseline 兜底
5. 出现 `NEW BEST` / `receipt: verdict=success`，且有 submission 落盘
6. 崩溃恢复抽查：在 grant 2–3 时 `kill -9` 对应 v2_closed_loop，用同一 STATE_DIR 重启，确认 `restored scientific state` 出现、预算不超发、stage 不回落

## 5. 24h-mle（正式）

```bash
cd /mnt/data/stage42_delivery/incoming
export LLM_MODEL=qwen3.8-max
export STATE_ROOT=/mnt/data/v2_state
export V2_EXEC_IMAGE=pact-stage42-p8:20260727T112909Z_legacy_l1
export V2_EXEC_PYTHON=/opt/conda/envs/agent/bin/python3
export V2_TORCH_CACHE=/mnt/data/v2_torch_cache
# 可选：每任务能力合成次数（默认 2，已持久化预算）
# export MAX_SYNTHESIS_ACTIONS=2
V2_PKG_DIR=ai_scientist_execution_layer_v2_20260806_v233 \
V2_PKG_TAR=ai_scientist_execution_layer_v2_20260806_v233.tar.gz \
V2_LLM_ENV=/mnt/data/stage42_delivery/latest_ai_scientist_v6.env \
  nohup bash run_v2_a100_3tasks_v23.sh 24h-mle > run_v2_24h_mle_v233_outer.log 2>&1 &
bash monitor_v2_v233_live.sh
```

## 6. 观测命令

```bash
STAMP=<run 时间戳>
grep -E "COMPILED|COMPILE_PATCH|SYNTHESIS|grant plan:|grant:|NEW BEST|receipt: verdict" run_v2_*_${STAMP}.log | tail -30
grep -cE "\[llm\] FAIL role=codegen" run_v2_*_${STAMP}.log
grep -E "LLM chose unknown branch|fallback baseline" run_v2_*_${STAMP}.log | head   # 期望 0
ls -t /mnt/data/v2_state/run_v2_*_${STAMP}/capabilities/   # Phase C 合成产物
```

## 7. v2.3.3 泛化验收（内容驱动 + mixed 判定）

```bash
# 1) profile 行：modality/text_cols/time_col 全内容驱动
grep -hE "profile: task_type=.*modality=" run_v2_*_${STAMP}.log | tail -3
# 2) image 任务：预缓存必须建成四档
grep -hE "image cache: sizes=" run_v2_*_${STAMP}.log | tail -3        # sizes=[64, 128, 192, 256] rows>0
# 3) text 任务：TEXT_COLUMNS 烘焙进编译脚本；timeseries 任务：LAG_COLUMN + time_holdout
grep -hE "TEXT_COLUMNS = |LAG_COLUMN = " run_v2_*_${STAMP}.log | tail -5
# 4) mixed 数据（杂散 Name 类列）：应出现 Mixed data 说明且 modality=tabular
grep -hE "Mixed data:" run_v2_*_${STAMP}.log | tail -3
# 5) HERA 能选到对应能力（非空能力退化）
grep -hE "HERA wrote new branch|grant plan:" run_v2_*_${STAMP}.log | tail -12
```

## 8. 原则提醒

- LLM 只做**研究决策**（HERA：选方法/轴/意图）和**能力合成**，不做每轮训练脚本生成；训练脚本全部来自模板编译。
- 失败 trial 会标记 broken 并排除；修复走确定性 patch（`COMPILE_PATCH`），不重新走 codegen。
- built-in 的 broken 标记只对本进程实例生效（registry 实例隔离），ephemeral broken 落盘持久化。
- 中途停机恢复：`docker rm -f v2_*` + 用同一 STATE_DIR 重启即可，预算/stage 均持久化。