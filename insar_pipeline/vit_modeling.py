from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.mu_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.logvar_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.diag_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, seq_len))

    def _resize_pos_embed(self, token_count_with_cls: int) -> torch.Tensor:
        if token_count_with_cls == self.pos_embed.size(1):
            return self.pos_embed[:, :token_count_with_cls, :]

        cls_pos = self.pos_embed[:, :1, :]
        spatial_pos = self.pos_embed[:, 1:, :]  # [1, N, C]

        old_grid = int((spatial_pos.size(1)) ** 0.5)
        new_grid = int((token_count_with_cls - 1) ** 0.5)
        if new_grid * new_grid != (token_count_with_cls - 1):
            raise ValueError(
                f"Token count without cls must be a perfect square, got {token_count_with_cls - 1}"
            )

        spatial_pos = spatial_pos.reshape(1, old_grid, old_grid, self.dim).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(spatial_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, self.dim)
        return torch.cat([cls_pos, spatial_pos], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 1, L, L]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        pos_embed = self._resize_pos_embed(tokens.size(1)).to(tokens.dtype).to(tokens.device)
        tokens = tokens + pos_embed
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]
        pred_mu = self.mu_head(cls_out).squeeze(-1)
        pred_logvar = self.logvar_head(cls_out).squeeze(-1).clamp(min=-10.0, max=5.0)
        diag_pred = self.diag_head(cls_out)
        return pred_mu, pred_logvar, diag_pred


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
    use_zscore: bool = False
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
    artifact_prefix: str = ""


@dataclass
class ViTDatasetBuildConfig:
    dataset_dir: Path
    output_dir: Path
    metric: Literal["phase_std", "coherence"] = "coherence"
    matrix_mode: Literal["similarity", "outer", "difference"] = "similarity"


def _artifact_name(prefix: str, base_name: str) -> str:
    """Return ``{prefix}_{base_name}`` when a prefix is set, else ``base_name``."""
    return f"{prefix}_{base_name}" if prefix else base_name


def _load_sequence_data(dataset_dir: Path, metric: str) -> np.ndarray:
    candidates = ["rnn_data_std.npy", "data_std.npy"] if metric == "phase_std" else ["rnn_data.npy", "data.npy"]
    for name in candidates:
        path = dataset_dir / name
        if path.exists():
            return np.load(path).astype(np.float32)
    raise FileNotFoundError(f"No sequence data found in {dataset_dir}. Tried: {candidates}")




def build_and_save_vit_matrix_dataset(config: ViTDatasetBuildConfig) -> Path:
    data = _load_sequence_data(config.dataset_dir, config.metric)
    base_ds = PixelMatrixDataset(data, is_prediction=True, matrix_mode=config.matrix_mode)
    matrices = np.zeros((base_ds.h, base_ds.w, base_ds.t, base_ds.t), dtype=np.float32)

    for i in range(base_ds.h):
        for j in range(base_ds.w):
            matrices[i, j] = base_ds._to_matrix(data[i, j, :])

    out_dir = config.output_dir / "vit_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vit_matrix_data.npy", matrices)
    np.save(out_dir / "matrix_data.npy", matrices)  # backward-compatible alias

    with open(config.dataset_dir / "dates.pkl", "rb") as f:
        dates = pickle.load(f)
    with open(out_dir / "dates.pkl", "wb") as f:
        pickle.dump(dates, f)

    obs_name = "score_observation_std.npy" if config.metric == "phase_std" else "score_observation.npy"
    obs_candidates = [obs_name, "geninue_std.npy" if config.metric == "phase_std" else "geninue.npy"]
    for name in obs_candidates:
        p = config.dataset_dir / name
        if p.exists():
            obs = np.load(p).astype(np.float32)
            np.save(out_dir / obs_name, obs)
            # backward-compatible alias
            np.save(out_dir / ("geninue_std.npy" if config.metric == "phase_std" else "geninue.npy"), obs)
            break

    # copy sequence inputs so vit_train_predict can read from vit_dataset directly
    seq_name = "rnn_data_std.npy" if config.metric == "phase_std" else "rnn_data.npy"
    seq_alias = "data_std.npy" if config.metric == "phase_std" else "data.npy"
    seq_candidates = [seq_name, seq_alias]
    for name in seq_candidates:
        p = config.dataset_dir / name
        if p.exists():
            seq = np.load(p).astype(np.float32)
            np.save(out_dir / seq_name, seq)
            np.save(out_dir / seq_alias, seq)
            break

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



def _normal_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar).clamp(min=1e-6)
    return (0.5 * (logvar + ((target - mu) ** 2) / var)).mean()

def run_vit_training_and_prediction(config: ViTConfig) -> Path:
    data = _load_sequence_data(config.dataset_dir, config.metric)

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
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val = float("inf")
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            matrix = batch["matrix"].to(device)
            y = batch["y"].to(device)
            diag = batch["diag"].to(device)

            matrix_masked, diag_mask = _apply_diag_mask(matrix, config.diag_mask_ratio)
            pred_mu, pred_logvar, diag_pred = model(matrix_masked)
            reg_loss = _normal_nll(pred_mu, pred_logvar, y) if config.use_zscore else mse(pred_mu, y)
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
                pred_mu, pred_logvar, _ = model(matrix)
                batch_loss = _normal_nll(pred_mu, pred_logvar, y) if config.use_zscore else mse(pred_mu, y)
                val_loss += batch_loss.item() * y.size(0)
        val_loss /= len(val_subset)

        print(f"[ViT][epoch {epoch + 1}/{config.epochs}] val_loss={val_loss:.6f} best={best_val:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_vit_model.pth")

    try:
        state_dict = torch.load("best_vit_model.pth", map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load("best_vit_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    pred_ds = PixelMatrixDataset(data, is_prediction=True, matrix_mode=config.matrix_mode)
    pred_loader = DataLoader(pred_ds, batch_size=config.pred_batch_size, shuffle=False)
    predictions = np.zeros((pred_ds.h, pred_ds.w), dtype=np.float32)
    pred_std = np.zeros((pred_ds.h, pred_ds.w), dtype=np.float32) if config.use_zscore else None

    with torch.no_grad():
        for batch in pred_loader:
            coords = batch["pixel_coords"].numpy()
            matrix = batch["matrix"].to(device)
            pred_mu, pred_logvar, _ = model(matrix)
            pred_sigma = torch.exp(0.5 * pred_logvar)
            for k, (i, j) in enumerate(coords):
                predictions[i, j] = pred_mu[k].item()
                if pred_std is not None:
                    pred_std[i, j] = pred_sigma[k].item()

    predict_dir = config.output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    np.save(predict_dir / _artifact_name(config.artifact_prefix, "future_predictions.npy"), predictions)
    if pred_std is not None:
        np.save(predict_dir / _artifact_name(config.artifact_prefix, "future_prediction_std.npy"), pred_std)
    torch.save(model.state_dict(), predict_dir / _artifact_name(config.artifact_prefix, "best_vit_model.pth"))
    return predict_dir
