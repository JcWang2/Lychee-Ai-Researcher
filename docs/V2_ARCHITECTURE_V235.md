# V2.3.5 — MLE-Bench 全量布局泛化（Layout Generalization）

## 目标
对 MLE-Bench 全量 competition 做静态扫描（82 个 prepare.py/config.yaml），修复所有
“标准布局之外”的真实盲区，保证 modality / task_type / target / 图像预缓存对
**所有任务类型**都正确识别，且不依赖任何 competition 名称（治本、泛化）。

## 扫描发现的 4 个盲区与修复

| 盲区 | 代表 case | v2.3.4 之前 | v2.3.5 修复 |
|---|---|---|---|
| `train_labels.csv` 训练表 | histopathologic / rsna-miccai / seti / invasive-species / bms | `unsupported dataset layout` 直接失败 | `_TRAIN_FILE_NAMES` 覆盖 train.csv / labels.csv / train_labels.csv / training.csv 及全部 .tsv 变体 |
| `.tsv` 训练/测试表 | movie-review（train.tsv/test.tsv + sampleSubmission.csv） | 解析失败；sample 文件名大小写不匹配 | `_resolve_test_path` 支持 test.tsv；`table_delimiter()` 让 analyzer/sanitize 共享 csv/tsv 分隔符探测；`_SAMPLE_FILE_NAMES` 覆盖 sampleSubmission.csv / samplesubmission.csv / kaggle_* / 本地化前缀等 |
| 无 public test.csv 的图片 case | aerial / dog / plant-seedlings / cassava / paddy ... | `sanitize_test_csv` 只认 train.csv/labels.csv + private/test.csv，且 target 用 header[-1]（taxi/histopathologic 会错） | private 候选扩展为 test.csv / answers.csv / gold_submission.csv（+tsv）；target 探测改为 sample_submission 驱动（同 analyzer 规则）；目标名不在 private 表时不再误删真实特征列 |
| mixed 数据强文本短路 | spaceship-titanic / 任何含 Name 列的表 | `strong_text`（≥4 词）直接绕过 dominance 规则 → Name 列劫持 modality=text | dominance 规则强化：可用特征排除 sample_submission 目标列（多标签文本任务保持 text）+ 排除 *id 后缀元数据列（PhraseId/SentenceId）；strong_text 门槛提高到“文档级散文”（≥8 词/行或 ≥40% 长串），人名/短标签永远够不到 |

## 额外泛化：无训练标签表的图片 case
- **class-dir 合成**：plant-seedlings 式 `train/<species>/*.png`（无任何 train CSV）→
  `synthesize_train_labels()` 按子目录生成 `train.csv(file,label)`，列名取自 sample_submission。
- **flat-prefix 合成**：dogs-vs-cats 式 `cat.0.jpg / dog.1.jpg`（≥50 张、2..64 个前缀 token）→
  按文件名前缀生成标签。
- 两者都只在“不存在任何训练表”时触发，幂等，失败静默跳过（non-fatal）。

## 图像预缓存（不变，已实测）
- `manifest["train_images"]` 命中 → `ensure_image_caches()` 自动建 64/128/192/256 四档缓存；
  aerial / aptos / dog 已在服务器实测 `image cache: sizes=[64,128,192,256] rows=...`。
- modality=image 判定：magic-number 校验（非扩展名）+ 递归计数 ≥50 或可解析尺寸。

## 回归测试
- 新增 `test_v2_235.py`（7 场景 27 断言）：train_labels、tsv+sampleSubmission、mixed 保持
  tabular、多标签文本保持 text、class-dir 合成、flat-prefix 合成、taxi 中段 target 的
  sanitize 只删标签列。
- `batch_layout_test.py`（本机开发脚本）12 场景全 PASS：image 平铺/命名/类目录/无标签表、
  text、多输出文本、timeseries、tabular、多输出回归、mixed、tsv。
- 离线套件 11 个文件全 PASS：metrics / contracts / pact / hera / stage_controller /
  resource_profiler / l1_transactional / closed_loop / v2_23 / v2_234 / v2_235。

## 部署
与 v2.3.x 相同：`install_v2_execution_layer.sh --target ... --run-tests`，
然后 `run_v2_a100_lite_v235.sh`（GPU 5 跑 taxi/nomad/spooky 三任务）。
