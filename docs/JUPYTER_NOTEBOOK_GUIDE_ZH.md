# Jupyter Notebook 操作指南（基于裁剪后数据）

本指南面向 **在 Jupyter Notebook 中分步骤执行** 的使用方式，与你当前需求一致：

- 使用裁剪后的数据构建数据集；
- 相干性来源为 **直接读取 ISCE 生成的 `filt_fine.cor`**；
- 三条实验路径：
  1. LSTM + zscore
  2. GRU + 时间信息 + 普通 score（非 zscore）
  3. 时间矩阵 ViT + zscore

> 说明：以下所有单元格都在仓库根目录执行（你当前为 `/data6/WORKDIR/amatrice2025/merged/SARcoherenceDPM-main`）。

---

## 1. Notebook 初始化

### Cell 1：设置路径变量

```python
import os
from pathlib import Path

REPO_DIR = Path('/data6/WORKDIR/amatrice2025/merged/SARcoherenceDPM-main')
BASE_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/interferograms')
CROPPED_DIR = BASE_DIR / 'cropped'
GEOM_DIR = Path('/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference')
EVENT_DATE = '20160824'

# 可配置参数（与你新需求对应）
SEQUENCE_LENGTH = 10
MATRIX_SIZE = 10

# ViT 参数（建议统一在这里改）
VIT_PATCH_SIZE = 1
VIT_DEPTH = 4

RNN_DATASET_DIR = CROPPED_DIR / 'dataset_rnn'
VIT_DATASET_DIR = CROPPED_DIR / 'vit_dataset'
PARAM_FILE = REPO_DIR / 'configs' / 'model_params.example.json'

os.chdir(REPO_DIR)

print('REPO_DIR =', REPO_DIR)
print('BASE_DIR =', BASE_DIR)
print('CROPPED_DIR =', CROPPED_DIR)
print('GEOM_DIR =', GEOM_DIR)
print('RNN_DATASET_DIR =', RNN_DATASET_DIR)
print('VIT_DATASET_DIR =', VIT_DATASET_DIR)
print('PARAM_FILE =', PARAM_FILE)

# ViT 关键约束：内部使用 seq_len = SEQUENCE_LENGTH - 1
# 需满足：(SEQUENCE_LENGTH - 1) % VIT_PATCH_SIZE == 0
print('ViT divisibility check =', (SEQUENCE_LENGTH - 1) % VIT_PATCH_SIZE)
```

### Cell 2：定义通用命令执行函数

```python
import subprocess
import shlex

def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'Command failed with code {result.returncode}')
```

---

## 2. 数据准备（裁剪 + 构建数据集）

### Cell 2.5：可选（推荐）参数文件统一管理

```python
# 将网络结构、训练超参数、sequence_length/matrix_size 统一放在 JSON 文件中维护
# 可复制 configs/model_params.example.json 另存为你的实验配置
print('param file exists =', PARAM_FILE.exists())
```

### Cell 3：执行裁剪（已裁剪可跳过）

```python
run_cmd(
    f"python -m insar_pipeline.app --step crop "
    f"--base-dir {BASE_DIR} "
    f"--geom-reference-dir {GEOM_DIR} "
    f"--cropped-dir {CROPPED_DIR} "
    f"--output-dir {CROPPED_DIR}"
)
```

### Cell 4：基于 `filt_fine.cor` 构建数据集

```python
run_cmd(
    f"python -m insar_pipeline.app --step build_dataset "
    f"--base-dir {BASE_DIR} "
    f"--cropped-dir {CROPPED_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--event-date {EVENT_DATE} "
    f"--input-source cor "
    f"--timeseries-metric coherence "
    f"--sequence-length {SEQUENCE_LENGTH} "
    f"--matrix-size {MATRIX_SIZE} "
    f"--param-file {PARAM_FILE}"
)
```

### Cell 5：检查数据集产物

```python
dataset_dir = RNN_DATASET_DIR
print('Dataset dir:', dataset_dir)
for name in ['rnn_data.npy', 'score_observation.npy', 'dates.pkl', 'matrix_dates.pkl', 'matrix_pairs.pkl']:
    p = dataset_dir / name
    print(name, 'exists=', p.exists())
```

---

## 3. 路径一：LSTM + zscore

### Cell 6：训练与预测（LSTM + zscore）

```python
run_cmd(
    f"python -m insar_pipeline.app --step train_predict "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {RNN_DATASET_DIR} "
    f"--timeseries-metric coherence "
    f"--ts-model lstm "
    f"--use-zscore "
    f"--param-file {PARAM_FILE}"
)
```

### Cell 7：计算 zscore 分数

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {RNN_DATASET_DIR} "
    f"--predict-dir {CROPPED_DIR / 'predict'} "
    f"--timeseries-metric coherence "
    f"--use-zscore "
    f"--param-file {PARAM_FILE}"
)
```

---

## 4. 路径二：GRU + 时间信息 + 普通 score

> 关键点：默认就是“带时间信息”，因此 **不要传 `--disable-timestamp`**。

### Cell 8：训练与预测（GRU）

```python
run_cmd(
    f"python -m insar_pipeline.app --step train_predict "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {RNN_DATASET_DIR} "
    f"--timeseries-metric coherence "
    f"--ts-model gru "
    f"--param-file {PARAM_FILE}"
)
```

### Cell 9：计算普通 score（不加 `--use-zscore`）

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {RNN_DATASET_DIR} "
    f"--predict-dir {CROPPED_DIR / 'predict'} "
    f"--timeseries-metric coherence "
    f"--param-file {PARAM_FILE}"
)
```

---

## 5. 路径三：时间矩阵 ViT + zscore

### Cell 10：构建 ViT 矩阵数据

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_build_dataset "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {RNN_DATASET_DIR} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity "
    f"--param-file {PARAM_FILE}"
)
```

> 说明：这一步会在 `{CROPPED_DIR}/vit_dataset` 下生成独立 ViT 数据集，包含：
> `vit_matrix_data.npy`、`rnn_data.npy`、`score_observation.npy`、`dates.pkl`。
> 从本版开始，Cell 11（训练）和 Cell 12（score）都使用 `--dataset-dir {VIT_DATASET_DIR}`，
> 从目录与文件名层面彻底与 RNN 路径解耦，避免 `data.npy` 混淆。

### Cell 11：ViT 训练与预测

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_train_predict "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {VIT_DATASET_DIR} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity "
    f"--vit-patch-size {VIT_PATCH_SIZE} "
    f"--vit-depth {VIT_DEPTH} "
    f"--param-file {PARAM_FILE}"
)
```

> 若出现你遇到的 `RuntimeError: size of tensor a (...) must match tensor b (...)`，常见原因是：
> 训练阶段输入长度是 `t-1`（预测最后一个历史点），而推理阶段输入长度是 `t`（预测下一时刻），
> 导致 patch token 数变化，`pos_embed` 长度不匹配。
>
> 本仓库已修复为**自动插值位置编码**（`_resize_pos_embed`），可兼容训练/推理 token 数不一致。
> 当前实现也与“遮蔽对角线自监督 + 高斯(mean,std)输出 + z-score评分”流程对齐。
> 仍建议保持参数可整除：`(SEQUENCE_LENGTH - 1) % VIT_PATCH_SIZE == 0`，以避免 patch 切分异常。

### Cell 11.2：先做快速冒烟（可选，强烈建议）

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_train_predict "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {VIT_DATASET_DIR} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity "
    f"--vit-patch-size {VIT_PATCH_SIZE} "
    f"--vit-depth {VIT_DEPTH} "
    f"--epochs 1 "
    f"--param-file {PARAM_FILE}"
)
```

### Cell 11.5：快速确认 ViT 产物（建议）

```python
print('vit_dataset exists =', VIT_DATASET_DIR.exists())
print('rnn_data in vit_dataset exists =', (VIT_DATASET_DIR / 'rnn_data.npy').exists())
print('predict file exists =', (CROPPED_DIR / 'predict' / 'future_predictions.npy').exists())
```

### Cell 12：score（zscore）

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {VIT_DATASET_DIR} "
    f"--predict-dir {CROPPED_DIR / 'predict'} "
    f"--timeseries-metric coherence "
    f"--use-zscore"
)
```

---

## 6. 输出地理编码产品（任一路径跑完后执行）

### Cell 13：输出结果

```python
run_cmd(
    f"python -m insar_pipeline.app --step output "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--predict-dir {CROPPED_DIR / 'predict'} "
    f"--lat-file {CROPPED_DIR / 'lat_cropped.rdr'} "
    f"--lon-file {CROPPED_DIR / 'lon_cropped.rdr'}"
)
```

---

## 7. Notebook 场景下的实用建议

1. **避免结果互相覆盖**：
   - 建议每条路径单独设置 `output_dir`（例如 `cropped_lstm_zscore/`、`cropped_gru_raw/`、`cropped_vit_zscore/`），或在每条路径结束后备份 `predict/`。
2. **快速调试**：
   - 先把 `SEQUENCE_LENGTH` 与 `MATRIX_SIZE` 设小一点，确认流程跑通后再增大。
3. **显式检查输出**：
   - 每跑完一条路径，检查 `future_predictions.npy`、`score.npy`、（zscore 分支下）`future_prediction_std.npy`（并确认其所在目录与当前路径对应：RNN 用 dataset_rnn，ViT 用 vit_dataset） 是否生成。

---

## 8. 可选：在 Notebook 内快速可视化 score

### Cell 14：显示 `score.npy`

```python
import numpy as np
import matplotlib.pyplot as plt

score = np.load(CROPPED_DIR / 'predict' / 'score.npy')
plt.figure(figsize=(6, 5))
plt.imshow(score, cmap='turbo')
plt.colorbar(label='score')
plt.title('Damage Proxy Score')
plt.tight_layout()
plt.show()
```
