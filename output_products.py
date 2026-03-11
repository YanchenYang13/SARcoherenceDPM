from __future__ import annotations

import datetime
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class InSARDataset(Dataset):
    def __init__(self, data: np.ndarray, dates: list[str], is_prediction: bool = False):
        self.data = data
        self.height, self.width, self.time_steps = data.shape
        self.is_prediction = is_prediction

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


@dataclass
class TrainingConfig:
    dataset_dir: Path
    output_dir: Path
    next_date: str
    epochs: int = 15
    train_batch_size: int = 128
    pred_batch_size: int = 256
    lr: float = 1e-3


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device="cpu"):
    model.to(device)
    best_val_loss = float("inf")

    for _ in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            time_features = batch["time_features"].to(device)
            target_time_features = batch["target_time_features"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            outputs = model(x, time_features, target_time_features)
            loss = criterion(outputs, y)
            loss.backward()
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
                val_loss += criterion(outputs, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    return model


def predict_future(model, dataset: InSARDataset, batch_size: int = 256, device="cpu") -> np.ndarray:
    model.eval()
    predictions = np.zeros((dataset.height, dataset.width), dtype=np.float32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            coords = batch["pixel_coords"].numpy()
            x = batch["x"].to(device)
            time_features = batch["time_features"].to(device)
            target_time_features = batch["target_time_features"].to(device)
            outputs = model(x, time_features, target_time_features)
            for i in range(len(coords)):
                pixel_i, pixel_j = coords[i]
                scaler = dataset.scalers[(pixel_i, pixel_j)]
                predictions[pixel_i, pixel_j] = scaler.inverse_transform(outputs[i].cpu().numpy().reshape(-1, 1))[0, 0]

    return predictions


def run_training_and_prediction(config: TrainingConfig) -> Path:
    data = np.load(config.dataset_dir / "data_std.npy")
    with open(config.dataset_dir / "dates.pkl", "rb") as f:
        dates = pickle.load(f)

    full_dataset = InSARDataset(data, dates)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=False)

    model = InSARLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.epochs, device=device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    all_dates = dates + [config.next_date]
    predict_dataset = InSARDataset(data, all_dates, is_prediction=True)
    future_predictions = predict_future(model, predict_dataset, batch_size=config.pred_batch_size, device=device)

    predict_dir = config.output_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    np.save(predict_dir / "future_predictions.npy", future_predictions)
    torch.save(model.state_dict(), predict_dir / "best_model.pth")
    return predict_dir
