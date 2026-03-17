# SARcoherenceDPM 全功能 Jupyter Notebook 指南（中文）

> 目标：用一个 Notebook 覆盖并验证本仓库 **已实现的主要功能**，包含：
>
> - 基础分步流程（`load_data/crop/build_dataset/train_predict/score/output/visualize`）
> - 一键流程（`full` / `vit_full` / `ccd_full`）
> - RNN 路径（LSTM/GRU）
> - ViT 路径
> - CCD 路径
> - 参数文件驱动

---

## 0. 使用说明与覆盖清单

### 0.1 本文档覆盖哪些 CLI step

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

### 0.2 执行建议

- 每个 Cell 独立可跑；首次建议按顺序跑。
- 体量较大步骤（训练/地理编码）可先做“冒烟参数”再做正式参数。
- 在 Notebook 中统一通过 `run_cmd(...)` 运行 CLI，便于记录日志。

---

## 1. Notebook 初始化

### Cell 1：路径与环境变量

```python
from pathlib import Path

# ====== 按你的机器修改 ======
BASE_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/interferograms')
GEOM_REF_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference')
CROPPED_DIR = BASE_DIR / 'cropped'

SLC_BASE_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/SLC')
SLC_CROPPED_DIR = SLC_BASE_DIR / 'cropped'

EVENT_DATE = '20160824'
NEXT_DATE = '20160821_20160902'
```

### Cell 2：通用命令执行函数

```python
import subprocess
import shlex

def run_cmd(cmd: str, check: bool = True):
    print('>>>', cmd)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        if check:
            raise RuntimeError(f'Command failed: {cmd}')
    return p
```

### Cell 3：可选参数文件（推荐）

```python
PARAM_FILE = Path('configs/model_params.example.json')
print('Param file exists:', PARAM_FILE.exists(), PARAM_FILE)
```

---

## 2. 基础数据流：分步流程（RNN 主线）

### Cell 4：`load_data`（快速检查输入序列可读）

```python
run_cmd(
    f"python -m insar_pipeline.app --step load_data "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE}"
)
```

### Cell 5：`crop`

```python
run_cmd(
    f"python -m insar_pipeline.app --step crop "
    f"--base-dir {BASE_DIR} "
    f"--geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {CROPPED_DIR}"
)
```

### Cell 6：`build_dataset`（`input_source=cor`）

```python
run_cmd(
    f"python -m insar_pipeline.app --step build_dataset "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--input-source cor"
)
```

### Cell 7：`build_dataset`（`input_source=stack_int` + 不同 coherence_source）

```python
for coh_src in ['isce', 'computed_phsig', 'computed_crlb']:
    run_cmd(
        f"python -m insar_pipeline.app --step build_dataset "
        f"--base-dir {BASE_DIR} "
        f"--output-dir {CROPPED_DIR} "
        f"--event-date {EVENT_DATE} "
        f"--input-source stack_int "
        f"--stack-root {BASE_DIR} "
        f"--coherence-source {coh_src}"
    )
```

### Cell 8：`train_predict`（LSTM + zscore）

```python
run_cmd(
    f"python -m insar_pipeline.app --step train_predict "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--next-date {NEXT_DATE} "
    f"--timeseries-metric coherence "
    f"--ts-model lstm "
    f"--use-zscore "
    f"--param-file {PARAM_FILE}"
)
```

### Cell 9：`train_predict`（GRU + 时间特征 + 非 zscore）

```python
run_cmd(
    f"python -m insar_pipeline.app --step train_predict "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--next-date {NEXT_DATE} "
    f"--timeseries-metric phase_std "
    f"--ts-model gru"
)
```

### Cell 10：`score`（测试 4 种模式）

```python
for mode in ['auto', 'direct', 'ndi', 'zscore']:
    run_cmd(
        f"python -m insar_pipeline.app --step score "
        f"--output-dir {CROPPED_DIR} "
        f"--score-mode {mode} "
        f"--use-zscore"
    )
```

### Cell 11：`output`（地理编码导出）

```python
run_cmd(
    f"python -m insar_pipeline.app --step output "
    f"--output-dir {CROPPED_DIR} "
    f"--lat-file {CROPPED_DIR / 'lat_cropped.rdr'} "
    f"--lon-file {CROPPED_DIR / 'lon_cropped.rdr'}"
)
```

### Cell 12：`visualize`（matplotlib + mintpy）

```python
# A) matplotlib
run_cmd(
    f"python -m insar_pipeline.app --step visualize "
    f"--output-dir {CROPPED_DIR} "
    f"--visualize-mode matplotlib "
    f"--visualize-input {CROPPED_DIR / 'predict' / 'score.npy'} "
    f"--visualize-output {CROPPED_DIR / 'predict' / 'score_matplotlib.png'} "
    f"--visualize-nodisplay"
)

# B) mintpy view.py 风格
run_cmd(
    f"python -m insar_pipeline.app --step visualize "
    f"--visualize-mode mintpy "
    f"--visualize-input {CROPPED_DIR / 'lon_cropped.rdr'} "
    f"--visualize-output {CROPPED_DIR / 'predict' / 'lon_view.png'} "
    f"--visualize-nodisplay"
)
```

---

## 3. ViT 路径（完整覆盖）

### Cell 13：`vit_build_dataset`

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_build_dataset "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity"
)
```

### Cell 14：`vit_train_predict`

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_train_predict "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--next-date {NEXT_DATE} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity "
    f"--vit-patch-size 2 "
    f"--vit-depth 4 "
    f"--use-zscore"
)
```

### Cell 15：ViT score 与输出

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {CROPPED_DIR / 'dataset_vit'} "
    f"--use-zscore "
    f"--score-mode auto"
)
```

---

## 4. CCD 路径（完整覆盖）

### Cell 16：`ccd_build_stack`

```python
run_cmd(
    f"python -m insar_pipeline.app --step ccd_build_stack "
    f"--base-dir {SLC_BASE_DIR} "
    f"--geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {SLC_CROPPED_DIR}"
)
```

### Cell 17：`ccd_run`

```python
run_cmd(
    f"python -m insar_pipeline.app --step ccd_run "
    f"--output-dir {SLC_CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--ccd-max-temporal-baseline 84 "
    f"--ccd-threshold 0.75"
)
```

### Cell 18：`ccd_full`（一键）

```python
run_cmd(
    f"python -m insar_pipeline.app --step ccd_full "
    f"--base-dir {SLC_BASE_DIR} "
    f"--geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {SLC_CROPPED_DIR} "
    f"--event-date {EVENT_DATE}"
)
```

---

## 5. 一键流程覆盖（full / vit_full）

### Cell 19：`full`

```python
run_cmd(
    f"python -m insar_pipeline.app --step full "
    f"--base-dir {BASE_DIR} "
    f"--geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--next-date {NEXT_DATE}"
)
```

### Cell 20：`vit_full`

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_full "
    f"--base-dir {BASE_DIR} "
    f"--geom-reference-dir {GEOM_REF_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--next-date {NEXT_DATE} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity"
)
```

---

## 6. 功能验收清单（建议勾选）

执行完上述单元后，至少确认：

- `crop` 产出 `lat_cropped.rdr` / `lon_cropped.rdr`
- `dataset_rnn/` 与 `dataset_vit/` 目录存在
- `predict/` 下存在以下一类或多类：
  - `rnn_*_future_predictions.npy`
  - `vit_*_future_predictions.npy`
  - `*_score.npy`
  - `ccd_temporal_probability.npy`
- `output/` 下存在导出的地理编码结果（若 MintPy 工具可用）
- 可视化 PNG 成功输出

可用如下检查：

```python
run_cmd(f"find {CROPPED_DIR} -maxdepth 3 -type f | head -n 80")
```

---

## 7. 常见问题

1. `ModuleNotFoundError`：请在正确的 Python 环境安装依赖（`numpy/torch/gdal/mintpy`）。
2. `output` 步骤失败：通常是 MintPy 或 ISCE 环境未配置完全。
3. `zscore` 失败：需要对应模型路径先生成 `future_prediction_std.npy`。
4. `visualize --mode mintpy` 失败：检查 `mintpy.cli.view` 是否可导入。

---

## 8. 推荐分享方式

对外展示时建议附带：

- 本文档（全功能覆盖）
- README（整体架构与入口）
- `configs/model_params.example.json`（统一参数模板）

这样新用户可直接定位：**原理 → 命令 → 产物 → 验证**。
