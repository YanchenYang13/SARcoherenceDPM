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
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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
    ):
        self.data = data
        self.height, self.width, self.time_steps = data.shape
        self.is_prediction = is_prediction
        self.use_timestamp = use_timestamp
        self.use_zscore = use_zscore
        self.metric = metric

        # -----------------------------------------------------------------
        # IMPROVED time-feature encoding (dim=7, same as before)
        #   OLD: [year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, raw_interval]
        #   NEW: [doy_sin,  doy_cos,  month_sin, month_cos, day_sin, day_cos, norm_interval]
        #
        # Key changes:
        # 1) year_sin/cos (period=2100) was essentially constant → replaced
        #    with day-of-year sin/cos (period=365.25) that captures seasonal
        #    decorrelation (vegetation, snow cover).
        # 2) interval (temporal baseline in days) is now normalized to [0,1]
        #    so it has the same scale as sin/cos features.
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

        # Per-pixel scaling
        self.scalers = {}
        self.scaled_data = np.zeros_like(self.data, dtype=np.float32)
        for i in range(self.height):
            for j in range(self.width):
                if scaler_type == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                pixel_ts = self.data[i, j, :]
                if self.use_zscore:
                    pixel_ts = _to_zscore_training_space(pixel_ts, self.metric)
                self.scaled_data[i, j, :] = scaler.fit_transform(pixel_ts.reshape(-1, 1)).flatten()
                self.scalers[(i, j)] = scaler

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

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, time_feat_dim=7):
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

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, time_feat_dim=7):
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


def _safe_logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(values, eps, 1 - eps)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _phase_std_to_coherence(phase_std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = np.clip(phase_std.astype(np.float32), eps, None)
    denom = np.sqrt(1.0 + 2.0 * (std**2))
    coh = 1.0 / denom
    return np.clip(coh, eps, 1 - eps).astype(np.float32)


def _coherence_to_phase_std(coh: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    c = np.clip(coh.astype(np.float32), eps, 1 - eps)
    return np.sqrt((1.0 - c**2) / (2.0 * c**2 + eps)).astype(np.float32)


def _to_zscore_training_space(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "coherence":
        return _safe_logit(values)
    if metric == "phase_std":
        return _safe_logit(_phase_std_to_coherence(values))
    raise ValueError(f"Unsupported metric for zscore space: {metric}")


def _from_zscore_training_space(values: np.ndarray, metric: str) -> np.ndarray:
    coh = _safe_sigmoid(values)
    if metric == "coherence":
        return coh.astype(np.float32)
    if metric == "phase_std":
        return _coherence_to_phase_std(coh)
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
    var = torch.exp(logvar).clamp(min=1e-6)
    return (0.5 * (logvar + ((target - mu) ** 2) / var)).mean()


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
        print(
            f"[RNN][epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={epoch_loss / max(len(train_loader), 1):.6f} "
            f"val_loss={val_loss:.6f} best={best_val_loss:.6f} lr={current_lr:.2e}",
            flush=True,
        )
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
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    predictions = np.zeros((dataset.height, dataset.width), dtype=np.float32)
    pred_std = np.zeros((dataset.height, dataset.width), dtype=np.float32) if use_zscore else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            coords = batch["pixel_coords"].numpy()
            x = batch["x"].to(device)
            time_features = batch["time_features"].to(device)
            target_time_features = batch["target_time_features"].to(device)
            outputs = model(x, time_features, target_time_features)
            if isinstance(outputs, tuple):
                pred_mean, pred_logvar = outputs
                model_output = pred_mean
                model_std = torch.exp(0.5 * pred_logvar)
            else:
                model_output = outputs
                model_std = None
            for i in range(len(coords)):
                pixel_i, pixel_j = coords[i]
                scaler = dataset.scalers[(pixel_i, pixel_j)]
                pred_value = scaler.inverse_transform(model_output[i].cpu().numpy().reshape(-1, 1))[0, 0]
                if dataset.use_zscore:
                    pred_value = _from_zscore_training_space(np.array([pred_value], dtype=np.float32), dataset.metric)[0]
                predictions[pixel_i, pixel_j] = pred_value
                if pred_std is not None and model_std is not None:
                    latent_std = model_std[i].item() / (scaler.scale_[0] + 1e-12)
                    latent_mean = scaler.inverse_transform(model_output[i].cpu().numpy().reshape(-1, 1))[0, 0]
                    pred_std[pixel_i, pixel_j] = _zscore_space_to_metric_std(latent_std, latent_mean, dataset.metric)

    return predictions, pred_std


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

    all_dates = dates + [config.next_date]
    predict_dataset = InSARDataset(
        data,
        all_dates,
        is_prediction=True,
        use_timestamp=config.use_timestamp,
        use_zscore=config.use_zscore,
        metric=config.metric,
        scaler_type=scaler_type,
    )
    future_predictions, future_pred_std = predict_future(
        model,
        predict_dataset,
        batch_size=config.pred_batch_size,
        device=device,
        use_zscore=config.use_zscore,
    )

    np.save(predict_dir / "future_predictions.npy", future_predictions)
    if config.artifact_prefix:
        np.save(predict_dir / f"{config.artifact_prefix}_future_predictions.npy", future_predictions)
    if future_pred_std is not None:
        np.save(predict_dir / "future_prediction_std.npy", future_pred_std)
        if config.artifact_prefix:
            np.save(predict_dir / f"{config.artifact_prefix}_future_prediction_std.npy", future_pred_std)
    torch.save(model.state_dict(), predict_dir / "best_model.pth")
    if config.artifact_prefix:
        torch.save(model.state_dict(), predict_dir / f"{config.artifact_prefix}_best_model.pth")
    return predict_dir
