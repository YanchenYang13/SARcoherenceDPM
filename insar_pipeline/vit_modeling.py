from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class PixelMatrixDataset(Dataset):
    def __init__(self, data: np.ndarray, is_prediction: bool = False, matrix_mode: str = "similarity"):
        self.data = data.astype(np.float32)
        self.h, self.w, self.t = self.data.shape
        self.is_prediction = is_prediction
        self.matrix_mode = matrix_mode
        self.samples = [(i, j) for i in range(self.h) for j in range(self.w)]

    def __len__(self) -> int:
        return len(self.samples)

    def _to_matrix(self, seq: np.ndarray) -> np.ndarray:
        si = seq[:, None]
        sj = seq[None, :]
        if self.matrix_mode == "outer":
            mat = si * sj
        elif self.matrix_mode == "difference":
            mat = np.abs(si - sj)
        else:
            # similarity matrix in [0, 1]
            mat = np.exp(-np.abs(si - sj))
        np.fill_diagonal(mat, seq)
        return mat.astype(np.float32)

    def __getitem__(self, idx: int):
        i, j = self.samples[idx]
        ts = self.data[i, j, :]

        if self.is_prediction:
            x = ts
            y = 0.0
        else:
            x = ts[:-1]
            y = ts[-1]

        matrix = self._to_matrix(x)
        diag = np.diag(matrix).copy()

        return {
            "pixel_coords": torch.tensor([i, j]),
            "matrix": torch.tensor(matrix[None, :, :], dtype=torch.float32),
            "diag": torch.tensor(diag, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }


class MaskedDiagonalViT(nn.Module):
    def __init__(self, seq_len: int, patch_size: int = 2, dim: int = 64, depth: int = 4, heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        grid = seq_len // patch_size
        self.num_tokens = grid * grid

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, dim))

        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.reg_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.diag_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, seq_len))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, L, L]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]
        pred = self.reg_head(cls_out).squeeze(-1)
        diag_pred = self.diag_head(cls_out)
        return pred, diag_pred


@dataclass
class ViTConfig:
    dataset_dir: Path
    output_dir: Path
    metric: Literal["phase_std", "coherence"] = "coherence"
    epochs: int = 20
    train_batch_size: int = 128
    pred_batch_size: int = 256
    lr: float = 1e-3
    matrix_mode: Literal["similarity", "outer", "difference"] = "similarity"
    patch_size: int = 2
    hidden_dim: int = 64
    depth: int = 4
    heads: int = 4
    diag_mask_ratio: float = 0.5
    diag_loss_weight: float = 0.3


@dataclass
class ViTDatasetBuildConfig:
    dataset_dir: Path
    output_dir: Path
    metric: Literal["phase_std", "coherence"] = "coherence"
    matrix_mode: Literal["similarity", "outer", "difference"] = "similarity"


def _load_data(dataset_dir: Path, metric: str) -> np.ndarray:
    data_filename = "data_std.npy" if metric == "phase_std" else "data.npy"
    return np.load(dataset_dir / data_filename).astype(np.float32)


def build_and_save_vit_matrix_dataset(config: ViTDatasetBuildConfig) -> Path:
    data = _load_data(config.dataset_dir, config.metric)
    base_ds = PixelMatrixDataset(data, is_prediction=True, matrix_mode=config.matrix_mode)
    matrices = np.zeros((base_ds.h, base_ds.w, base_ds.t, base_ds.t), dtype=np.float32)

    for i in range(base_ds.h):
        for j in range(base_ds.w):
            matrices[i, j] = base_ds._to_matrix(data[i, j, :])

    out_dir = config.output_dir / "vit_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "matrix_data.npy", matrices)
    with open(config.dataset_dir / "dates.pkl", "rb") as f:
        dates = pickle.load(f)
    with open(out_dir / "dates.pkl", "wb") as f:
        pickle.dump(dates, f)
    return out_dir


def _apply_diag_mask(matrix: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    # matrix: [B, 1, L, L]
    masked = matrix.clone()
    b, _, l, _ = matrix.shape
    mask = torch.zeros((b, l), device=matrix.device, dtype=torch.bool)
    num_mask = max(1, int(l * ratio))
    for bi in range(b):
        idx = torch.randperm(l, device=matrix.device)[:num_mask]
        mask[bi, idx] = True
        masked[bi, 0, idx, idx] = 0.0
    return masked, mask


def run_vit_training_and_prediction(config: ViTConfig) -> Path:
    data = _load_data(config.dataset_dir, config.metric)

    train_ds = PixelMatrixDataset(data, is_prediction=False, matrix_mode=config.matrix_mode)
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.train_batch_size, shuffle=False)

    seq_len = data.shape[-1] - 1
    if seq_len % config.patch_size != 0:
        raise ValueError("For ViT, sequence length (t-1) must be divisible by patch_size")

    model = MaskedDiagonalViT(
        seq_len=seq_len,
        patch_size=config.patch_size,
        dim=config.hidden_dim,
        depth=config.depth,
        heads=config.heads,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val = float("inf")
    for _ in range(config.epochs):
        model.train()
        for batch in train_loader:
            matrix = batch["matrix"].to(device)
            y = batch["y"].to(device)
            diag = batch["diag"].to(device)

            matrix_masked, diag_mask = _apply_diag_mask(matrix, config.diag_mask_ratio)
            pred, diag_pred = model(matrix_masked)
            reg_loss = mse(pred, y)
            diag_loss = ((diag_pred - diag) ** 2)[diag_mask].mean()
            loss = reg_loss + config.diag_loss_weight * diag_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                matrix = batch["matrix"].to(device)
                y = batch["y"].to(device)
                pred, _ = model(matrix)
                val_loss += mse(pred, y).item() * y.size(0)
        val_loss /= len(val_subset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_vit_model.pth")

    model.load_state_dict(torch.load("best_vit_model.pth", map_location=device))
    model.eval()

    pred_ds = PixelMatrixDataset(data, is_prediction=True, matrix_mode=config.matrix_mode)
    pred_loader = DataLoader(pred_ds, batch_size=config.pred_batch_size, shuffle=False)
    predictions = np.zeros((pred_ds.h, pred_ds.w), dtype=np.float32)

    with torch.no_grad():
        for batch in pred_loader:
            coords = batch["pixel_coords"].numpy()
            matrix = batch["matrix"].to(device)
            pred, _ = model(matrix)
            for k, (i, j) in enumerate(coords):
                predictions[i, j] = pred[k].item()

    predict_dir = config.output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    np.save(predict_dir / "future_predictions.npy", predictions)
    torch.save(model.state_dict(), predict_dir / "best_vit_model.pth")
    return predict_dir
