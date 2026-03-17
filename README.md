# SARcoherenceDPM

一个面向研究与工程落地的 InSAR 灾损代理图（Damage Proxy Mapping, DPM）流程库。仓库把原本偏 Notebook 的流程整理为可复用的 Python 包与分步 CLI，支持 **RNN / ViT / CCD** 三类路径，并保留可视化与地理编码输出能力。

---

## 1. 项目目标

SARcoherenceDPM 关注以下问题：

- 从多时相干涉产品构建像素级时间序列；
- 在“无灾害基线”假设下预测震后期望值；
- 将观测值与预测值比较，生成损伤代理分数；
- 输出可用于 GIS 分析的地理编码产品。

核心原则：**模块化、可复现、可扩展（不绑定单一模型/单一路径）**。

---

## 2. 功能全景（Feature Matrix）

| 模块 | 已实现能力 | 对应 CLI `--step` |
|---|---|---|
| 数据裁剪 | 裁剪 `cor/int/full` 与 `lat/lon` | `crop` |
| 数据集构建 | `cor` 直读或 `stack_int` 推导相干性 | `build_dataset` |
| 时序训练预测（RNN） | LSTM / GRU，支持时间特征与 zscore 分支 | `train_predict` |
| 分数生成 | `auto/direct/ndi/zscore`，支持 metric-aware 符号 | `score` |
| 地理编码输出 | geocode / subset / save_gdal 导出 | `output` |
| 可视化 | matplotlib 与 MintPy `view.py` 风格 | `visualize` |
| ViT 时序矩阵 | 矩阵构建 + 训练预测 | `vit_build_dataset`, `vit_train_predict` |
| 全流程一键（RNN） | crop→dataset→train→score→output | `full` |
| 全流程一键（ViT） | crop→dataset→vit train→score→output | `vit_full` |
| 时序退相干 CCD | SLC 栈构建与变更检测（Jung et al., 2016） | `ccd_build_stack`, `ccd_run`, `ccd_full` |

---

## 3. 代码结构

```text
insar_pipeline/
├── app.py                # CLI 入口：参数解析与分步调度
├── pipeline.py           # full / vit_full 高层编排
├── preprocess.py         # 裁剪逻辑与文件搜集
├── dataset_builder.py    # 时序样本构建与序列化
├── isce_stack.py         # ISCE stack 对读取与索引
├── coherence.py          # 相干性估计与映射工具
├── modeling.py           # RNN(LSTM/GRU) 训练预测
├── vit_modeling.py       # ViT 时序矩阵建模
├── scoring.py            # score 计算
├── output_products.py    # 地理编码产品导出
├── temporal_ccd.py       # 时序退相干 CCD
└── visualization.py      # matplotlib / MintPy 可视化
```

---

## 4. CLI 步骤总览

```bash
python -m insar_pipeline.app --step <STEP> [args...]
```

支持步骤：

- `load_data`
- `crop`
- `build_dataset`
- `train_predict`
- `score`
- `output`
- `visualize`
- `full`
- `vit_build_dataset`
- `vit_train_predict`
- `vit_full`
- `ccd_build_stack`
- `ccd_run`
- `ccd_full`

---

## 5. 典型流程（RNN 路径）

### 5.1 裁剪

```bash
python -m insar_pipeline.app --step crop \
  --base-dir /data/.../merged/interferograms \
  --geom-reference-dir /data/.../merged/geom_reference
```

### 5.2 构建数据集

```bash
python -m insar_pipeline.app --step build_dataset \
  --base-dir /data/.../merged/interferograms \
  --output-dir /data/.../merged/interferograms/cropped \
  --input-source cor
```

`--input-source stack_int` 时可再指定：

- `--coherence-source isce`
- `--coherence-source computed_phsig`
- `--coherence-source computed_crlb`

### 5.3 训练与预测

```bash
python -m insar_pipeline.app --step train_predict \
  --output-dir /data/.../cropped \
  --timeseries-metric phase_std \
  --ts-model lstm \
  --use-zscore
```

### 5.4 计算 score

```bash
python -m insar_pipeline.app --step score \
  --output-dir /data/.../cropped \
  --score-mode auto
```

### 5.5 地理编码输出

```bash
python -m insar_pipeline.app --step output \
  --output-dir /data/.../cropped \
  --lat-file /data/.../cropped/lat_cropped.rdr \
  --lon-file /data/.../cropped/lon_cropped.rdr
```

---

## 6. ViT 路径

```bash
python -m insar_pipeline.app --step vit_build_dataset \
  --output-dir /data/.../cropped \
  --timeseries-metric coherence \
  --vit-matrix-mode similarity

python -m insar_pipeline.app --step vit_train_predict \
  --output-dir /data/.../cropped \
  --timeseries-metric coherence \
  --vit-matrix-mode similarity \
  --vit-patch-size 2 \
  --vit-depth 4

python -m insar_pipeline.app --step score \
  --dataset-dir /data/.../cropped/dataset_vit \
  --output-dir /data/.../cropped \
  --use-zscore
```

---

## 7. CCD 路径（Jung et al., 2016）

```bash
python -m insar_pipeline.app --step ccd_build_stack \
  --base-dir /data/.../merged/SLC \
  --geom-reference-dir /data/.../merged/geom_reference

python -m insar_pipeline.app --step ccd_run \
  --output-dir /data/.../merged/SLC/cropped \
  --event-date 20160824 \
  --ccd-max-temporal-baseline 84 \
  --ccd-threshold 0.75
```

会在 `predict/` 输出：

- `ccd_temporal_probability.npy`
- `ccd_temporal_change.npy`

> 如果只有 `.slc.full.vrt`，请先转 ENVI 实体文件：
>
> `gdal_translate -of envi 20160821.slc.full.vrt 20160821.slc.full`

---

## 8. 可视化

### 8.1 matplotlib 后端

```bash
python -m insar_pipeline.app --step visualize \
  --output-dir /data/.../cropped \
  --visualize-input /data/.../cropped/predict/rnn_lstm_coherence_zscore_time_score.npy \
  --visualize-mode matplotlib \
  --visualize-output /data/.../cropped/predict/rnn_score.png \
  --visualize-nodisplay
```

### 8.2 MintPy `view.py` 风格

```bash
python -m insar_pipeline.app --step visualize \
  --visualize-input /data/.../merged/SLC/lon.rdr \
  --visualize-mode mintpy \
  --visualize-output /tmp/lon_view.png \
  --visualize-nodisplay
```

不传 `--visualize-input` 时，CLI 会在 `predict/` 中按优先级自动选取：

1. `*score.npy`
2. `*_probability.npy`
3. `*.npy`

---

## 9. 参数文件（推荐）

支持通过 JSON 统一管理参数，例如：`configs/model_params.example.json`。

```bash
python -m insar_pipeline.app --step train_predict \
  --output-dir /data/.../cropped \
  --param-file configs/model_params.example.json
```

参数文件可覆盖 `global / dataset / rnn / vit` 四个段落。

---

## 10. 关键产物目录

- 裁剪产物：`<output_dir>/`
- 数据集产物：`<output_dir>/dataset_rnn` 或 `<output_dir>/dataset_vit`
- 预测与分数：`<output_dir>/predict`
- 地理编码导出：`<output_dir>/output`

---

## 11. 文档导航

- 中文全功能 Notebook 指南：`docs/JUPYTER_NOTEBOOK_GUIDE_ZH.md`
- 示例参数文件：`configs/model_params.example.json`
- 包入口与导出：`insar_pipeline/__init__.py`

---

## 12. 运行环境

常见依赖：

- `numpy`
- `matplotlib`
- `torch`
- `osgeo.gdal`
- MintPy（`geocode.py` / `subset.py` / `save_gdal.py` / `view.py`）

建议在已配置 InSAR/MintPy 的环境中运行。
