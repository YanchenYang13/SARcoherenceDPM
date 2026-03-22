from __future__ import annotations

import datetime
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
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
    ):
        self.data = data
        self.height, self.width, self.time_steps = data.shape
        self.is_prediction = is_prediction
        self.use_timestamp = use_timestamp
        self.use_zscore = use_zscore
        self.metric = metric

        self.time_features = []
        for date_str in dates:
            start_date, end_date = date_str.split("_")
            start = datetime.datetime.strptime(start_date, "%Y%m%d")
            end = datetime.datetime.strptime(end_date, "%Y%m%d")

            year_sin = np.sin(2 * np.pi * start.year / 2100)
            year_cos = np.cos(2 * np.pi * start.year / 2100)
            month_sin = np.sin(2 * np.pi * start.month / 12)
            month_cos = np.cos(2 * np.pi * start.month / 12)
            day_sin = np.sin(2 * np.pi * start.day / 31)
            day_cos = np.cos(2 * np.pi * start.day / 31)
            interval = (end - start).days
            self.time_features.append([year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, interval])
        self.time_features = np.array(self.time_features)

        self.scalers = {}
        self.scaled_data = np.zeros_like(self.data, dtype=np.float32)
        for i in range(self.height):
            for j in range(self.width):
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
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, time_feat_dim=7):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Linear(time_feat_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.target_time_proj = nn.Linear(time_feat_dim, hidden_dim)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, src, time_features, target_time_features):
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        time_embed = self.time_embedding(time_features)
        combined_input = src + time_embed
        _, (h_n, _) = self.lstm(combined_input)
        seq_repr = h_n[-1]
        target_time_embed = self.target_time_proj(target_time_features)
        combined = torch.cat([seq_repr, target_time_embed], dim=-1)
        return self.output_layer(combined).squeeze(-1)


class InSARGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, time_feat_dim=7):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Linear(time_feat_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.target_time_proj = nn.Linear(time_feat_dim, hidden_dim)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, src, time_features, target_time_features):
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        time_embed = self.time_embedding(time_features)
        combined_input = src + time_embed
        _, h_n = self.gru(combined_input)
        seq_repr = h_n[-1]
        target_time_embed = self.target_time_proj(target_time_features)
        combined = torch.cat([seq_repr, target_time_embed], dim=-1)
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

    # metric == "phase_std", where phase_std = sqrt((1-c^2)/(2*c^2))
    c = max(coh, 1e-6)
    one_minus_c2 = max(1.0 - c * c, 1e-8)
    dstd_dc = -1.0 / (np.sqrt(2.0) * c * c * np.sqrt(one_minus_c2))
    dstd_dz = dstd_dc * dcoh_dz
    return float(abs(dstd_dz) * std_in_zspace)


def _normal_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar).clamp(min=1e-6)
    return (0.5 * (logvar + ((target - mu) ** 2) / var)).mean()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device="cpu", max_grad_norm: float | None = None):
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            time_features = batch["time_features"].to(device)
            target_time_features = batch["target_time_features"].to(device)
            y = batch["y"].to(device)

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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                time_features = batch["time_features"].to(device)
                target_time_features = batch["target_time_features"].to(device)
                y = batch["y"].to(device)
                outputs = model(x, time_features, target_time_features)
                if isinstance(outputs, tuple):
                    loss = criterion(outputs[0], outputs[1], y)
                else:
                    loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"[RNN][epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={epoch_loss / max(len(train_loader), 1):.6f} "
            f"val_loss={val_loss:.6f} best={best_val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

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
    data = np.load(config.dataset_dir / data_filename)
    with open(config.dataset_dir / "dates.pkl", "rb") as f:
        dates = pickle.load(f)

    full_dataset = InSARDataset(
        data,
        dates,
        use_timestamp=config.use_timestamp,
        use_zscore=config.use_zscore,
        metric=config.metric,
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=False)

    print("[RNN] Training configuration")
    print(f"[RNN] dataset_dir={config.dataset_dir}")
    print(f"[RNN] data_file={data_filename} shape={data.shape} metric={config.metric}")
    print(
        f"[RNN] model={config.model_type} use_timestamp={config.use_timestamp} "
        f"use_zscore={config.use_zscore} hidden_dim={config.hidden_dim} "
        f"num_layers={config.num_layers} dropout={config.dropout}"
    )
    print(
        f"[RNN] epochs={config.epochs} train_batch_size={config.train_batch_size} "
        f"pred_batch_size={config.pred_batch_size} lr={config.lr} optimizer={config.optimizer} "
        f"weight_decay={config.weight_decay} max_grad_norm={config.max_grad_norm}"
    )
    print(f"[RNN] train_samples={len(train_dataset)} val_samples={len(val_dataset)}")

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
        print(f"[RNN] device=cuda ({gpu_name})")
    else:
        print("[RNN] device=cpu (CUDA not available)")

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=config.epochs,
        device=device,
        max_grad_norm=config.max_grad_norm,
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    all_dates = dates + [config.next_date]
    predict_dataset = InSARDataset(
        data,
        all_dates,
        is_prediction=True,
        use_timestamp=config.use_timestamp,
        use_zscore=config.use_zscore,
        metric=config.metric,
    )
    future_predictions, future_pred_std = predict_future(
        model,
        predict_dataset,
        batch_size=config.pred_batch_size,
        device=device,
        use_zscore=config.use_zscore,
    )

    predict_dir = config.output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
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
