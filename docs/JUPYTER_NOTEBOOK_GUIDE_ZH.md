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

os.chdir(REPO_DIR)

print('REPO_DIR =', REPO_DIR)
print('BASE_DIR =', BASE_DIR)
print('CROPPED_DIR =', CROPPED_DIR)
print('GEOM_DIR =', GEOM_DIR)

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
    f"--matrix-size {MATRIX_SIZE}"
)
```

### Cell 5：检查数据集产物

```python
dataset_dir = CROPPED_DIR / 'dataset'
print('Dataset dir:', dataset_dir)
for name in ['data.npy', 'dates.pkl', 'matrix_dates.pkl', 'matrix_pairs.pkl']:
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
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
    f"--timeseries-metric coherence "
    f"--ts-model lstm "
    f"--use-zscore"
)
```

### Cell 7：计算 zscore 分数

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
    f"--predict-dir {CROPPED_DIR / 'predict'} "
    f"--timeseries-metric coherence "
    f"--use-zscore"
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
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
    f"--timeseries-metric coherence "
    f"--ts-model gru"
)
```

### Cell 9：计算普通 score（不加 `--use-zscore`）

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
    f"--predict-dir {CROPPED_DIR / 'predict'} "
    f"--timeseries-metric coherence"
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
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity"
)
```

> 说明：这一步会在 `{CROPPED_DIR}/vit_dataset` 下生成矩阵版本数据（用于检查/扩展）。
> 当前代码中 `vit_train_predict` 训练时仍读取 `{CROPPED_DIR}/dataset/data.npy`，
> 所以 Cell 11 的 `--dataset-dir` 依旧填写 `{CROPPED_DIR / 'dataset'}`（不是 `vit_dataset`）。

### Cell 11：ViT 训练与预测

```python
run_cmd(
    f"python -m insar_pipeline.app --step vit_train_predict "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
    f"--timeseries-metric coherence "
    f"--vit-matrix-mode similarity "
    f"--vit-patch-size {VIT_PATCH_SIZE} "
    f"--vit-depth {VIT_DEPTH}"
)
```

> 若出现你遇到的 `RuntimeError: size of tensor a (...) must match tensor b (...)`，优先检查两点：
> 1) 代码版本是否包含本次修复（`vit_modeling.py` 中先 `cat` 再加 `pos_embed`）；
> 2) 参数是否满足 `(SEQUENCE_LENGTH - 1) % VIT_PATCH_SIZE == 0`。
>    例如 `SEQUENCE_LENGTH=10` 时建议 `VIT_PATCH_SIZE=1/3/9`；若 `VIT_PATCH_SIZE=2`，则建议把 `SEQUENCE_LENGTH` 改为 `9/11/13/...`。

### Cell 11.5：快速确认 ViT 产物（建议）

```python
print('vit_dataset exists =', (CROPPED_DIR / 'vit_dataset').exists())
print('train dataset exists =', (CROPPED_DIR / 'dataset' / 'data.npy').exists())
print('predict file exists =', (CROPPED_DIR / 'predict' / 'future_predictions.npy').exists())
```

### Cell 12：score（zscore）

```python
run_cmd(
    f"python -m insar_pipeline.app --step score "
    f"--base-dir {BASE_DIR} "
    f"--output-dir {CROPPED_DIR} "
    f"--dataset-dir {CROPPED_DIR / 'dataset'} "
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
   - 每跑完一条路径，检查 `future_predictions.npy`、`score.npy`、（zscore 分支下）`future_prediction_std.npy` 是否生成。

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
