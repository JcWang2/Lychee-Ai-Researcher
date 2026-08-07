# V2.3.8 通用任务识别（RLE 掩码 / bbox 检测 / 音频）+ 确定性基线

## 背景
v2.3.7 之后仍有 26 个 MLE-Bench 竞赛无法正确识别（被当成 tabular / 报布局错误）。
v2.3.8 的目标是**治本**：不做任何竞赛名特判，只用结构信号（列内容 + 文件形态）
把任务类型识别对，并为每个新 modality 提供**可提交的确定性基线**。

## 新增结构识别（hera/analyzer.py）
按"测量，不猜测"原则新增 4 类信号，全部基于内容/形态：

| 信号 | 判定规则 | 输出 |
|---|---|---|
| RLE 掩码目标列 | 目标列抽样 200 个非空值，>=60% 匹配 `\d+( \d+)+`（空 / `-` 行是合法"无掩码"，不计入分母） | `mask_target` → `task_type=segmentation` → `modality=image_mask` |
| bbox 坐标列集 | 表头完整出现 6 组坐标名之一（x,y,w,h / x1,y1,x2,y2 / x_min,y_min,x_max,y_max / xmin,ymin,xmax,ymax / left,top,right,bottom / x,y,width,height） | `bbox_columns` → `task_type=detection` → `modality=image_detection` |
| JSON box 列 | **任意列** >=50% 非空值是以 `[`/`{` 开头且含 `"bbox"` 或 x/y/w/h 键（kuzushiji 的 labels、siim 的 boxes 等） | 同 bbox → detection |
| 音频文件 | 数据树递归扫描 wav/mp3/flac/ogg/m4a/aac/opus/wma >= 50 个 | `audio_file_count` → `modality=audio` |

覆盖优先级（在 `_guess_modality` 之后强制覆盖）：
`image_pixel`（v2.3.7 保留） > `image_mask` > `image_detection` > `audio`。
文本/表格信号永远不能劫持 mask/bbox/audio。

## 布局合成（data_layout.py）
部分竞赛的 public 目录**根本没有 train.csv / test.csv**（vesuvius、contrails、
tensorflow-speech、mlsp 形状）。新增泛型合成路径：

1. **目录标签表**：`train/audio/<label>/*.wav`（或 `train/<label>/*.wav`）→ 写
   `public/train.csv` = (fname, label)，fname 相对 public 根。
2. **样本拷贝表**：无 train 表但有 sample_submission → 拷贝为 `public/train.csv`
   （占位目标值让 RLE / JSON sniff 能分类任务；确定性基线不拿占位值训练）。
3. **id-only test 表**：public/private 都没有测试表时，从 sample 写 `test.csv`（仅 id 列）。
4. **守卫**：只有 sample 直接位于该候选 public_dir 内才合成——避免 flat 候选目录
   劫持 `prepared/public` 布局（回归测试覆盖）。

另：`_TRAIN_FILE_NAMES` 增加 `train_image_level.csv` / `train_study_level.csv`
（siim-covid19 形状，image-level 优先）；`materialize_dataset` 支持
`train_images.zip` / `test_images.zip`（kuzushiji 形状）。

## 新能力（capability_registry + program_compiler）
三个确定性基线（全部 `gpu=False`、无预处理、single_holdout）：

| method_id | renderer | 提交策略 | OOF 契约 |
|---|---|---|---|
| `image.mask.rle.baseline.v1` | `image_mask_rle_baseline` | 空掩码（`-`/空串，按训练列内容嗅探）或全图掩码（RLE 编码，列优先） | 每像素 true/pred 0/1 → dice / iou_mean |
| `image.detection.bbox.baseline.v1` | `image_detection_bbox_baseline` | 全部相同占位值（vinbigdata `14 1 0 0 1 1`）→ 多数类+单位框（siim）→ 空串（kuzushiji 跳过空预测） | (query,true,pred) 类频率排序 → map_at_k 有定义且可被超越 |
| `audio.tabular.baseline.v1` | `audio_tabular_baseline` | 多数类（单标签）或每类频率（多标签） | 单标签 (true,pred)；多标签 true_<c>/pred_<c> → label_ranking_ap |

模板执行环境新增 env：`MASK_TARGET`、`BBOX_COLUMNS`（JSON）、`MULTI_ROW_TARGET`、
`AUDIO_FILE_COUNT`（manifest 同步新增对应字段）。

## 指标推断（metrics_registry.py）
- 未知竞赛 + `task_type=segmentation` → `dice`（不再默认 accuracy）
- 未知竞赛 + `task_type=detection` → `map_at_k`
- 修复 `apply_metric_to_profile`：未知竞赛时按实测 task_type 重推断（此前
  profile 默认 alignment="exact" 导致永远停留 accuracy，mask/detection 模板会被拒）。

## 覆盖到的竞赛族（结构识别，非名称特判）
- RLE 掩码：hubmap-kidney-segmentation、tgs-salt-identification-challenge、
  google-research-identify-contrails-reduce-global-warming、
  vesuvius-challenge-ink-detection、uw-madison-gi-tract-image-segmentation
- bbox / JSON box：vinbigdata-chest-xray-abnormalities-detection、
  kuzushiji-recognition、siim-covid19-detection
- 音频：freesound-audio-tagging-2019、tensorflow-speech-recognition-challenge、
  mlsp-2013-birds
- 布局合成：vesuvius / contrails / tensorflow-speech / mlsp 的缺表形态

## 已知边界（诚实说明）
- 3D 目标检测（3d-object-detection-for-autonomous-vehicles）坐标在 JSON answers，
  无 bbox 列集 → 保持 tabular 路径；nfl-player-contact-detection 是视频 tracking
  （mp4），不识别为图像；序列/QA 类（tensorflow2-question-answering、chaii、
  tweet-sentiment-extraction、text-normalization、billion-word-imputation、
  bms-molecular-translation 等）仍走 tabular/text 路径——它们的提交不是简单
  多数类/空值可覆盖的，留待 v2.3.9+。
- detection 基线内部 map@k 是"类频率排序"代理指标（不预测框坐标），真实分数
  由外部 grader 决定；闭环内它只是可被超越的起点。
- vesuvius 的训练标签在 private（public 不可见），基线只输出合法空掩码提交。

## 验证
`python test_v2_238.py`：嗅探 5 类信号 + 布局合成（含 flat 守卫）+ 指标推断 +
registry/compiler 契约 + 3 个模板端到端执行（oof.csv/submission.csv/metric 打印）。