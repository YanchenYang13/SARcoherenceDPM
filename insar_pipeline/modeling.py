from __future__ import annotations

import datetime
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class InSARDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        dates: list[str],
        is_prediction: bool = False,
        use_timestamp: bool = True,
        use_zscore: bool = False,
        metric: Literal["phase_std", "coherence"] = "phase_std",
        scaler_type: str = "robust",
        prefit_scaler_center: np.ndarray | None = None,
        prefit_scaler_scale: np.ndarray | None = None,
    ):
        self.data = data
        self.height, self.width, self.time_steps = data.shape
        self.is_prediction = is_prediction
        self.use_timestamp = use_timestamp
        self.use_zscore = use_zscore
        self.metric = metric

        # -----------------------------------------------------------------
        # IMPROVED time-feature encoding (dim=8)
        #   [doy_sin, doy_cos, month_sin, month_cos, day_sin, day_cos,
        #    norm_interval, seq_pos]
        #
        # Features:
        # 1) doy_sin/cos (period=365.25): captures seasonal decorrelation
        #    (vegetation, snow cover, soil moisture).
        # 2) month_sin/cos, day_sin/cos: fine-grained calendar position.
        # 3) norm_interval: temporal baseline normalised to [0,1]; distinguishes
        #    6-day from 12-day acquisitions.
        # 4) seq_pos: normalised sequence position in [0,1] (0=oldest pair in
        #    the dataset, 1=most recent / prediction target).  This gives the
        #    LSTM an explicit recency cue so it can weight recent pre-event
        #    coherence more heavily when anticipating the post-event scene.
        # -----------------------------------------------------------------
        intervals = []
        self.time_features = []
        for date_str in dates:
            start_date, end_date = date_str.split("_")
            start = datetime.datetime.strptime(start_date, "%Y%m%d")
            end = datetime.datetime.strptime(end_date, "%Y%m%d")

            doy = start.timetuple().tm_yday
            doy_sin = np.sin(2 * np.pi * doy / 365.25)
            doy_cos = np.cos(2 * np.pi * doy / 365.25)
            month_sin = np.sin(2 * np.pi * start.month / 12)
            month_cos = np.cos(2 * np.pi * start.month / 12)
            day_sin = np.sin(2 * np.pi * start.day / 31)
            day_cos = np.cos(2 * np.pi * start.day / 31)
            interval = (end - start).days
            intervals.append(interval)
            self.time_features.append([doy_sin, doy_cos, month_sin, month_cos, day_sin, day_cos, interval])

        self.time_features = np.array(self.time_features, dtype=np.float32)

        # Normalize interval to [0, 1]
        max_interval = max(intervals) if intervals else 1.0
        if max_interval == 0:
            max_interval = 1.0
        self.time_features[:, 6] /= max_interval

        # Append normalised sequence position [0, 1] as 8th feature.
        # The position reflects temporal order: 0 = oldest pair in the
        # current dataset window, 1 = most recent entry (or the prediction
        # target date when is_prediction=True).
        n_dates = len(self.time_features)
        seq_pos = np.arange(n_dates, dtype=np.float32) / max(n_dates - 1, 1)
        self.time_features = np.column_stack([self.time_features, seq_pos])

        # ------------------------------------------------------------------
        # Vectorised per-pixel scaling (replaces the former per-pixel loop
        # over sklearn scalers, which was the main performance bottleneck).
        #
        # Convention: forward  X_scaled = (X - center) / scale
        #             inverse  X        = X_scaled * scale + center
        #
        # For RobustScaler : center = median, scale = IQR (Q75 − Q25)
        # For MinMaxScaler(-1,1): center = (min+max)/2,  scale = (max−min)/2
        #
        # If pre-fitted arrays are supplied (prediction dataset reusing the
        # training scaler), they are used directly and no fitting is done.
        # ------------------------------------------------------------------
        if prefit_scaler_center is not None and prefit_scaler_scale is not None:
            self.scaler_center = prefit_scaler_center
            self.scaler_scale = prefit_scaler_scale
        else:
            # Apply zscore transform before computing scaler statistics so
            # that the scaler operates in logit-coherence space.
            fit_data = (
                to_zscore_training_space(data[:, :, : self.time_steps], self.metric)
                if use_zscore
                else data[:, :, : self.time_steps].astype(np.float32)
            )

            if scaler_type == "robust":
                self.scaler_center = np.median(fit_data, axis=2).astype(np.float32)
                q25 = np.percentile(fit_data, 25, axis=2).astype(np.float32)
                q75 = np.percentile(fit_data, 75, axis=2).astype(np.float32)
                self.scaler_scale = (q75 - q25).astype(np.float32)
            else:
                # MinMaxScaler equivalent with feature_range=(-1, 1)
                data_min = fit_data.min(axis=2).astype(np.float32)
                data_max = fit_data.max(axis=2).astype(np.float32)
                self.scaler_center = ((data_min + data_max) / 2.0).astype(np.float32)
                self.scaler_scale = ((data_max - data_min) / 2.0).astype(np.float32)

            # Constant pixels (zero IQR / zero range) get scale = 1 so that
            # scaled output is 0 everywhere instead of NaN.
            self.scaler_scale = np.where(
                self.scaler_scale == 0, np.float32(1.0), self.scaler_scale
            )

        # Apply the scaling to the whole data cube in one vectorised pass.
        transformed = (
            to_zscore_training_space(data, self.metric)
            if use_zscore
            else data.astype(np.float32)
        )
        self.scaled_data = (
            (transformed - self.scaler_center[:, :, np.newaxis])
            / self.scaler_scale[:, :, np.newaxis]
        ).astype(np.float32)

        self.samples = [(i, j) for i in range(self.height) for j in range(self.width)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j = self.samples[idx]
        pixel_ts = self.scaled_data[i, j, :]

        if not self.is_prediction:
            x = pixel_ts[:-1]
            y = pixel_ts[-1]
            time_feat = self.time_features[:-1]
            target_time_feat = self.time_features[-1]
        else:
            x = pixel_ts
            y = 0.0
            time_feat = self.time_features[: self.time_steps]
            target_time_feat = self.time_features[-1]

        if not self.use_timestamp:
            time_feat = np.zeros_like(time_feat, dtype=np.float32)
            target_time_feat = np.zeros_like(target_time_feat, dtype=np.float32)

        return {
            "pixel_coords": torch.tensor([i, j]),
            "x": torch.tensor(x, dtype=torch.float32),
            "time_features": torch.tensor(time_feat, dtype=torch.float32),
            "target_time_features": torch.tensor(target_time_feat, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }


class InSARLSTM(nn.Module):
    """LSTM with gated time-feature fusion and true notime bypass."""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, time_feat_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Linear(time_feat_dim, hidden_dim)

        # Gated fusion learns HOW MUCH time info to inject
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.target_time_proj = nn.Linear(time_feat_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src, time_features, target_time_features):
        src = src.unsqueeze(-1)
        src_embed = self.input_embedding(src)

        # True bypass: when time features are all zero, skip fusion entirely
        time_active = time_features.abs().sum() > 0

        if time_active:
            time_embed = self.time_embedding(time_features)
            gate_input = torch.cat([src_embed, time_embed], dim=-1)
            gate = self.fusion_gate(gate_input)
            combined_input = gate * src_embed + (1 - gate) * time_embed
            combined_input = self.layer_norm(combined_input)
        else:
            combined_input = self.layer_norm(src_embed)

        _, (h_n, _) = self.lstm(combined_input)
        seq_repr = h_n[-1]

        if time_active:
            target_time_embed = self.target_time_proj(target_time_features)
            combined = torch.cat([seq_repr, target_time_embed], dim=-1)
        else:
            combined = torch.cat([seq_repr, torch.zeros_like(seq_repr)], dim=-1)

        return self.output_layer(combined).squeeze(-1)


class InSARGRU(nn.Module):
    """GRU variant with the same gated-fusion improvements."""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, time_feat_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Linear(time_feat_dim, hidden_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.target_time_proj = nn.Linear(time_feat_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src, time_features, target_time_features):
        src = src.unsqueeze(-1)
        src_embed = self.input_embedding(src)

        time_active = time_features.abs().sum() > 0

        if time_active:
            time_embed = self.time_embedding(time_features)
            gate_input = torch.cat([src_embed, time_embed], dim=-1)
            gate = self.fusion_gate(gate_input)
            combined_input = gate * src_embed + (1 - gate) * time_embed
            combined_input = self.layer_norm(combined_input)
        else:
            combined_input = self.layer_norm(src_embed)

        _, h_n = self.gru(combined_input)
        seq_repr = h_n[-1]

        if time_active:
            target_time_embed = self.target_time_proj(target_time_features)
            combined = torch.cat([seq_repr, target_time_embed], dim=-1)
        else:
            combined = torch.cat([seq_repr, torch.zeros_like(seq_repr)], dim=-1)

        return self.output_layer(combined).squeeze(-1)


class InSARDistributionHead(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.mu_head = nn.Linear(1, 1)
        self.logvar_head = nn.Linear(1, 1)

    def forward(self, src, time_features, target_time_features):
        base_out = self.base_model(src, time_features, target_time_features).unsqueeze(-1)
        mu = self.mu_head(base_out).squeeze(-1)
        logvar = self.logvar_head(base_out).squeeze(-1).clamp(min=-10.0, max=5.0)
        return mu, logvar


@dataclass
class TrainingConfig:
    dataset_dir: Path
    output_dir: Path
    next_date: str
    epochs: int = 15
    train_batch_size: int = 128
    pred_batch_size: int = 256
    lr: float = 1e-3
    metric: Literal["phase_std", "coherence"] = "phase_std"
    model_type: Literal["lstm", "gru"] = "lstm"
    use_timestamp: bool = True
    use_zscore: bool = False
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    artifact_prefix: str = ""


def _artifact_name(prefix: str, base_name: str) -> str:
    """Return ``{prefix}_{base_name}`` when a prefix is set, else ``base_name``."""
    return f"{prefix}_{base_name}" if prefix else base_name


def _safe_logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(values, eps, 1 - eps)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _phase_std_to_coherence(phase_std: np.ndarray, eps: float = 1e-6, looks: float = 3.0) -> np.ndarray:
    std = np.clip(phase_std.astype(np.float32), eps, None)
    denom = np.sqrt(1.0 + 2.0 * looks * (std**2))
    coh = 1.0 / denom
    return np.clip(coh, eps, 1 - eps).astype(np.float32)


def _coherence_to_phase_std(coh: np.ndarray, eps: float = 1e-6, looks: float = 3.0) -> np.ndarray:
    c = np.clip(coh.astype(np.float32), eps, 1 - eps)
    return np.sqrt((1.0 - c**2) / (2.0 * looks * c**2 + eps)).astype(np.float32)


def to_zscore_training_space(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "coherence":
        return _safe_logit(values)
    if metric == "phase_std":
        # phase_std is already approximately Gaussian-distributed (range ~0 to 2.5).
        # Empirical distribution analysis shows log() introduces left skew and
        # logit(coherence) introduces severe right skew with extreme tails.
        # The raw phase_std values are the best latent space for Gaussian NLL training.
        # Robust scaler handles centering and scaling downstream.
        return values.astype(np.float32)
    raise ValueError(f"Unsupported metric for zscore space: {metric}")


def _from_zscore_training_space(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "coherence":
        return _safe_sigmoid(values).astype(np.float32)
    if metric == "phase_std":
        # Identity inverse: no nonlinear transform was applied.
        return values.astype(np.float32)
    raise ValueError(f"Unsupported metric for zscore space: {metric}")


def _zscore_space_to_metric_std(std_in_zspace: float, z_value: float, metric: str) -> float:
    coh = float(_safe_sigmoid(np.array([z_value], dtype=np.float32))[0])
    dcoh_dz = coh * (1.0 - coh)
    if metric == "coherence":
        return float(abs(dcoh_dz) * std_in_zspace)

    c = max(coh, 1e-6)
    one_minus_c2 = max(1.0 - c * c, 1e-8)
    dstd_dc = -1.0 / (np.sqrt(2.0) * c * c * np.sqrt(one_minus_c2))
    dstd_dz = dstd_dc * dcoh_dz
    return float(abs(dstd_dz) * std_in_zspace)


def _normal_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Gaussian negative log-likelihood (constant −0.5·log(2π) omitted).
    # NLL = 0.5 * (logvar + (target − μ)² / σ²)
    # This value can be negative when the model learns a tight predictive
    # distribution (small logvar) and fits the data well — that is expected
    # and correct behaviour, not a sign of numerical error.
    var = torch.exp(logvar).clamp(min=1e-6)
    return (0.5 * (logvar + ((target - mu) ** 2) / var)).mean()


def _zscore_space_to_metric_std_vec(
    std_in_zspace: np.ndarray, z_value: np.ndarray, metric: str
) -> np.ndarray:
    """Vectorised version of _zscore_space_to_metric_std for batch prediction."""
    coh = _safe_sigmoid(z_value)  # shape matches input
    dcoh_dz = coh * (1.0 - coh)
    if metric == "coherence":
        return np.abs(dcoh_dz) * std_in_zspace

    c = np.maximum(coh, 1e-6)
    one_minus_c2 = np.maximum(1.0 - c * c, 1e-8)
    dstd_dc = -1.0 / (np.sqrt(2.0) * c * c * np.sqrt(one_minus_c2))
    dstd_dz = dstd_dc * dcoh_dz
    return np.abs(dstd_dz) * std_in_zspace


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    checkpoint_path: Path,
    num_epochs=50,
    device="cpu",
    max_grad_norm: float | None = None,
):
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    best_val_loss = float("inf")
    prev_val_loss = float("inf")

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Early stopping
    patience = max(num_epochs // 3, 5)
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        first_batch_logged = False
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=device.type == "cuda")
            time_features = batch["time_features"].to(device, non_blocking=device.type == "cuda")
            target_time_features = batch["target_time_features"].to(device, non_blocking=device.type == "cuda")
            y = batch["y"].to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            outputs = model(x, time_features, target_time_features)
            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], outputs[1], y)
            else:
                loss = criterion(outputs, y)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            if not first_batch_logged:
                first_batch_logged = True
                print(
                    f"[RNN] first_train_batch device={x.device} "
                    f"batch_shape={tuple(x.shape)} target_shape={tuple(y.shape)}",
                    flush=True,
                )

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=device.type == "cuda")
                time_features = batch["time_features"].to(device, non_blocking=device.type == "cuda")
                target_time_features = batch["target_time_features"].to(device, non_blocking=device.type == "cuda")
                y = batch["y"].to(device, non_blocking=device.type == "cuda")
                outputs = model(x, time_features, target_time_features)
                if isinstance(outputs, tuple):
                    loss = criterion(outputs[0], outputs[1], y)
                else:
                    loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        current_lr = scheduler.get_last_lr()[0]
        val_loss_delta = prev_val_loss - val_loss
        # Relative improvement < 0.01 % signals practical convergence.
        converged = (
            prev_val_loss != float("inf")
            and abs(val_loss_delta / (abs(prev_val_loss) + 1e-12)) < 1e-4
        )
        print(
            f"[RNN][epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={epoch_loss / max(len(train_loader), 1):.6f} "
            f"val_loss={val_loss:.6f} best={best_val_loss:.6f} "
            f"delta={val_loss_delta:+.6f} lr={current_lr:.2e}"
            + (" [converged]" if converged else ""),
            flush=True,
        )
        prev_val_loss = val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[RNN] saved_best_model={checkpoint_path}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"[RNN] early_stopping epoch={epoch + 1} patience={patience} "
                    f"best_val_loss={best_val_loss:.6f}",
                    flush=True,
                )
                break

    return model


def predict_future(
    model,
    dataset: InSARDataset,
    batch_size: int = 256,
    device="cpu",
    use_zscore: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Run inference and return predictions.

    Returns
    -------
    predictions : ndarray (H, W)
        Predicted values in the original metric space (phase_std or coherence).
    pred_metric_unc : ndarray (H, W) or None
        Predictive uncertainty propagated to metric space via the delta method.
        Available only when ``use_zscore=True`` and the model outputs a
        distribution (mean + logvar).  NOTE: this approximation degrades near
        extreme coherence values; prefer ``pred_latent_std`` for z-score
        computation.
    pred_latent : ndarray (H, W) or None
        Model mean prediction in logit-coherence (latent / z) space.
        The Gaussian NLL training loss is defined in this space, so z-scores
        computed here are statistically well-defined.
    pred_latent_std : ndarray (H, W) or None
        Model distribution std in logit-coherence (latent / z) space.
        This is the ``σ`` of the predicted Gaussian, NOT a phase_std value.
    """
    model.eval()
    predictions = np.zeros((dataset.height, dataset.width), dtype=np.float32)
    # pred_metric_unc: delta-method uncertainty in metric space (kept for compat)
    pred_metric_unc: np.ndarray | None = (
        np.zeros((dataset.height, dataset.width), dtype=np.float32) if use_zscore else None
    )
    # pred_latent / pred_latent_std: prediction mean and std in logit-coherence space
    pred_latent_arr: np.ndarray | None = (
        np.zeros((dataset.height, dataset.width), dtype=np.float32) if use_zscore else None
    )
    pred_latent_std_arr: np.ndarray | None = (
        np.zeros((dataset.height, dataset.width), dtype=np.float32) if use_zscore else None
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            coords = batch["pixel_coords"].numpy()  # (B, 2)
            pixel_is = coords[:, 0]
            pixel_js = coords[:, 1]

            x = batch["x"].to(device)
            time_features = batch["time_features"].to(device)
            target_time_features = batch["target_time_features"].to(device)
            outputs = model(x, time_features, target_time_features)

            if isinstance(outputs, tuple):
                pred_mean, pred_logvar = outputs
                mean_np = pred_mean.cpu().numpy()  # (B,) — scaled-space mean
                # pred_dist_std: distribution std in *scaled* space (not phase_std!)
                pred_dist_std_np = torch.exp(0.5 * pred_logvar).cpu().numpy()  # (B,)
            else:
                mean_np = outputs.cpu().numpy()  # (B,)
                pred_dist_std_np = None

            # ----------------------------------------------------------
            # Vectorised inverse scaling: X_latent = X_scaled * scale + center
            # Convention matches InSARDataset: X_scaled = (X - center)/scale
            # ----------------------------------------------------------
            scale = dataset.scaler_scale[pixel_is, pixel_js]   # (B,)
            center = dataset.scaler_center[pixel_is, pixel_js]  # (B,)
            pred_latent = mean_np * scale + center  # (B,) — value in logit-coherence space

            if dataset.use_zscore:
                pred_values = _from_zscore_training_space(pred_latent, dataset.metric)
            else:
                pred_values = pred_latent

            predictions[pixel_is, pixel_js] = pred_values

            if pred_dist_std_np is not None:
                # Propagate distribution std from scaled space to logit-coherence space:
                #   X_latent = X_scaled * scale  →  σ_latent = σ_scaled * scale
                # σ_latent is the Gaussian std in logit-coherence space.
                # This is conceptually separate from phase_std data values.
                latent_std = pred_dist_std_np * scale  # (B,) — std in logit-coh space

                if pred_latent_arr is not None:
                    pred_latent_arr[pixel_is, pixel_js] = pred_latent
                if pred_latent_std_arr is not None:
                    pred_latent_std_arr[pixel_is, pixel_js] = latent_std

                # Also propagate to metric space via delta method (kept for compat).
                # NOTE: this approximation breaks down near extreme coherence values
                # (coh ≈ 0 or coh ≈ 1) where the Jacobian vanishes, causing
                # metric-space std ≈ 0 and hence exploding z-scores.
                # Use latent-space z-scores instead for robust anomaly detection.
                if pred_metric_unc is not None:
                    pred_metric_unc[pixel_is, pixel_js] = _zscore_space_to_metric_std_vec(
                        latent_std, pred_latent, dataset.metric
                    )

    return predictions, pred_metric_unc, pred_latent_arr, pred_latent_std_arr


def _build_model(config: TrainingConfig) -> nn.Module:
    model_kwargs = dict(hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "gru":
        return InSARGRU(**model_kwargs)
    return InSARLSTM(**model_kwargs)


def _resolve_timeseries_filename(dataset_dir: Path, metric: str) -> str:
    candidates = ["rnn_data_std.npy", "data_std.npy"] if metric == "phase_std" else ["rnn_data.npy", "data.npy"]
    for name in candidates:
        if (dataset_dir / name).exists():
            return name
    raise FileNotFoundError(f"No timeseries dataset file found in {dataset_dir}. Tried: {candidates}")


def run_training_and_prediction(config: TrainingConfig) -> Path:
    data_filename = _resolve_timeseries_filename(config.dataset_dir, config.metric)
    print(f"[RNN] loading_timeseries={config.dataset_dir / data_filename}", flush=True)
    data = np.load(config.dataset_dir / data_filename)
    with open(config.dataset_dir / "dates.pkl", "rb") as f:
        dates = pickle.load(f)

    scaler_type = "robust"

    prep_start = time.perf_counter()
    print("[RNN] preparing_dataset_scalers=start", flush=True)
    full_dataset = InSARDataset(
        data,
        dates,
        use_timestamp=config.use_timestamp,
        use_zscore=config.use_zscore,
        metric=config.metric,
        scaler_type=scaler_type,
    )
    print(f"[RNN] preparing_dataset_scalers=done elapsed_sec={time.perf_counter() - prep_start:.2f}", flush=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=False, pin_memory=use_pin_memory)

    print("[RNN] Training configuration", flush=True)
    print(f"[RNN] dataset_dir={config.dataset_dir}", flush=True)
    print(f"[RNN] data_file={data_filename} shape={data.shape} metric={config.metric}", flush=True)
    print(
        f"[RNN] model={config.model_type} use_timestamp={config.use_timestamp} "
        f"use_zscore={config.use_zscore} hidden_dim={config.hidden_dim} "
        f"num_layers={config.num_layers} dropout={config.dropout}",
        flush=True,
    )
    print(
        f"[RNN] epochs={config.epochs} train_batch_size={config.train_batch_size} "
        f"pred_batch_size={config.pred_batch_size} lr={config.lr} optimizer={config.optimizer} "
        f"weight_decay={config.weight_decay} max_grad_norm={config.max_grad_norm}",
        flush=True,
    )
    print(f"[RNN] scaler_type={scaler_type}", flush=True)
    print(f"[RNN] train_samples={len(train_dataset)} val_samples={len(val_dataset)}", flush=True)
    if config.use_zscore:
        print(
            "[RNN] loss=NLL (Gaussian negative log-likelihood in logit-coherence space); "
            "negative values are expected once the model learns a tight predictive distribution",
            flush=True,
        )

    base_model = _build_model(config)
    model = InSARDistributionHead(base_model) if config.use_zscore else base_model
    criterion = _normal_nll if config.use_zscore else nn.MSELoss()
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[RNN] device=cuda ({gpu_name})", flush=True)
    else:
        print("[RNN] device=cpu (CUDA not available)", flush=True)

    predict_dir = config.output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = predict_dir / f"{config.artifact_prefix or 'rnn'}_best_model_training.pth"

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        checkpoint_path=checkpoint_path,
        num_epochs=config.epochs,
        device=device,
        max_grad_norm=config.max_grad_norm,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Reuse the scaler statistics fitted on training data — the prediction
    # dataset covers the same spatial pixels so refitting is redundant and
    # would be as slow as the initial fit.
    all_dates = dates + [config.next_date]
    predict_dataset = InSARDataset(
        data,
        all_dates,
        is_prediction=True,
        use_timestamp=config.use_timestamp,
        use_zscore=config.use_zscore,
        metric=config.metric,
        scaler_type=scaler_type,
        prefit_scaler_center=full_dataset.scaler_center,
        prefit_scaler_scale=full_dataset.scaler_scale,
    )
    future_predictions, future_pred_metric_unc, future_pred_latent, future_pred_latent_std = predict_future(
        model,
        predict_dataset,
        batch_size=config.pred_batch_size,
        device=device,
        use_zscore=config.use_zscore,
    )

    np.save(predict_dir / _artifact_name(config.artifact_prefix, "future_predictions.npy"), future_predictions)
    if future_pred_metric_unc is not None:
        # Metric-space uncertainty via delta method (kept for backward compat).
        # For z-score computation use the latent-space files below instead.
        np.save(predict_dir / _artifact_name(config.artifact_prefix, "future_prediction_std.npy"), future_pred_metric_unc)
    if future_pred_latent is not None:
        # Mean prediction in logit-coherence (latent / z) space.  The model's
        # Gaussian NLL loss is defined here, making latent-space z-scores
        # statistically well-defined and numerically stable.
        np.save(predict_dir / _artifact_name(config.artifact_prefix, "future_predictions_latent.npy"), future_pred_latent)
    if future_pred_latent_std is not None:
        # Distribution std in logit-coherence space (σ of the predicted
        # Gaussian).  Distinct from phase_std data values — this quantifies
        # model uncertainty, not InSAR phase noise.
        np.save(predict_dir / _artifact_name(config.artifact_prefix, "future_prediction_latent_std.npy"), future_pred_latent_std)
    torch.save(model.state_dict(), predict_dir / _artifact_name(config.artifact_prefix, "best_model.pth"))
    return predict_dir
