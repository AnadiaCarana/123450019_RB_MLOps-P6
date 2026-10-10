"""Modul bersama untuk seluruh skrip praktikum (Fashion-MNIST + MLflow + Optuna + DVC)."""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "subset"
DVC_FILE = ROOT / "data" / "subset.dvc"
BEST_CONFIG_PATH = ROOT / "configs" / "best_config.json"

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "fmnist-cnn"
MODEL_NAME = "fmnist-cnn-classifier"
SEEDS = [11, 22, 33]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530  # statistik Fashion-MNIST (train set penuh)


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 10
    dropout: float = 0.0
    use_bn: bool = False
    augment: bool = False
    base_channels: int = 32
    weight_decay: float = 0.0
    seed: int = 42


def load_best_config() -> Config:
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError("configs/best_config.json belum ada. Jalankan src/tune_optuna.py dulu.")
    return Config(**json.loads(BEST_CONFIG_PATH.read_text()))


# --------------------------------------------------------------------------- data + DVC
def get_data_hash() -> str:
    """Ambil hash konten data dari file .dvc (dibuat oleh `dvc add data/subset`)."""
    if not DVC_FILE.exists():
        raise FileNotFoundError(
            "data/subset.dvc tidak ditemukan. Jalankan: python src/prepare_data.py && dvc add data/subset"
        )
    meta = yaml.safe_load(DVC_FILE.read_text())
    return str(meta["outs"][0]["md5"])


def load_data() -> dict[str, np.ndarray]:
    names = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
    if not all((DATA_DIR / f"{n}.npy").exists() for n in names):
        raise FileNotFoundError("data/subset belum lengkap. Jalankan src/prepare_data.py (atau `dvc pull`).")
    return {n: np.load(DATA_DIR / f"{n}.npy") for n in names}


class ArrayDataset(Dataset):
    """Gambar uint8 -> float [0,1]. Normalisasi dilakukan DI DALAM model agar model registry self-contained."""

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = torch.from_numpy(x).float().div_(255.0).unsqueeze(1)
        self.y = torch.from_numpy(y).long()
        self.tf = v2.Compose([v2.RandomCrop(28, padding=2), v2.RandomHorizontalFlip()]) if augment else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = self.x[i]
        if self.tf is not None:
            img = self.tf(img)
        return img, self.y[i]


# --------------------------------------------------------------------------- model
class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN))
        self.register_buffer("std", torch.tensor(STD))

    def forward(self, x):
        return (x - self.mean) / self.std


class CNN(nn.Module):
    def __init__(self, base_channels: int = 32, dropout: float = 0.0, use_bn: bool = False):
        super().__init__()

        def block(i: int, o: int):
            layers = [nn.Conv2d(i, o, 3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(o))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            return layers

        c = base_channels
        self.net = nn.Sequential(
            Normalize(),
            *block(1, c), *block(c, 2 * c),            # 28 -> 14 -> 7
            nn.Flatten(),
            nn.Linear(2 * c * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- training
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> dict[str, float]:
    model.eval()
    dl = DataLoader(ArrayDataset(x, y), batch_size=512)
    losses, preds = [], []
    for xb, yb in dl:
        out = model(xb.to(device))
        losses.append(nn.functional.cross_entropy(out, yb.to(device), reduction="sum").item())
        preds.append(out.argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return {
        "loss": float(sum(losses) / len(y)),
        "acc": float((p == y).mean()),
        "macro_f1": float(f1_score(y, p, average="macro")),
    }


def train_and_eval(cfg: Config, data: dict, eval_test: bool = False):
    """Latih di train split; metrik per epoch dari VALIDATION split. Test hanya bila eval_test=True."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    model = CNN(cfg.base_channels, cfg.dropout, cfg.use_bn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(ArrayDataset(data["x_train"], data["y_train"], cfg.augment),
                          batch_size=cfg.batch_size, shuffle=True, generator=g)

    val = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        val = evaluate(model, data["x_val"], data["y_val"], device)
        mlflow.log_metrics({"train_loss": total / n, "val_loss": val["loss"],
                            "val_acc": val["acc"], "val_macro_f1": val["macro_f1"]}, step=epoch)

    final = {"val_acc": val["acc"], "val_macro_f1": val["macro_f1"]}
    if eval_test:
        t = evaluate(model, data["x_test"], data["y_test"], device)
        final.update({"test_acc": t["acc"], "test_macro_f1": t["macro_f1"], "test_loss": t["loss"]})
        mlflow.log_metrics({k: v for k, v in final.items() if k.startswith("test_")})
    return model, final


# --------------------------------------------------------------------------- MLflow helpers
def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def start_run(run_name: str, stage: str, nested: bool = False, extra_tags: dict | None = None):
    """Setiap run (parent maupun child) membawa hash data DVC sebagai tag DAN parameter."""
    h = get_data_hash()
    tags = {"stage": stage, "data_md5": h, "dataset": "fashion-mnist-subset", **(extra_tags or {})}
    run = mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
    mlflow.log_param("data_md5", h)
    return run


def log_pytorch_model(model: nn.Module, data: dict):
    model = model.to("cpu").eval()
    x_ex = (data["x_val"][:5].astype(np.float32) / 255.0)[:, None, :, :]
    with torch.no_grad():
        y_ex = model(torch.from_numpy(x_ex)).numpy()
    return mlflow.pytorch.log_model(model, name="model",
                                    signature=infer_signature(x_ex, y_ex), input_example=x_ex)


def run_seeds(name: str, stage: str, cfg: Config, data: dict, seeds=SEEDS, eval_test: bool = False,
              log_model: bool = True, extra_tags: dict | None = None, note: str = ""):
    """Satu parent run (ringkasan mean ± std, ddof=1) + satu child run per seed."""
    collected = defaultdict(list)
    with start_run(name, stage, extra_tags=extra_tags) as parent:
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_params({k: v for k, v in asdict(cfg).items() if k != "seed"})
        mlflow.log_param("seeds", ",".join(map(str, seeds)))
        mlflow.log_dict(asdict(cfg), "config.json")
        for s in seeds:
            c = replace(cfg, seed=s)
            with start_run(f"{name}-seed{s}", f"{stage}-seed", nested=True, extra_tags=extra_tags):
                mlflow.log_params(asdict(c))
                model, m = train_and_eval(c, data, eval_test=eval_test)
                if log_model:
                    log_pytorch_model(model, data)
            for k, v in m.items():
                collected[k].append(v)
        summary = {}
        for k, vals in collected.items():
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals, ddof=1))  # sample std; n=3 -> estimasi kasar
        mlflow.log_metrics(summary)
        return parent.info.run_id, summary
