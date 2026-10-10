"""Unduh Fashion-MNIST, ambil subset stratified, simpan sebagai .npy deterministik di data/subset.
Setelah ini jalankan:  dvc add data/subset
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # TIDAK dilacak DVC/Git (hanya cache unduhan)
OUT = ROOT / "data" / "subset"       # dilacak DVC
SEED = 42
N_TRAIN, N_VAL, N_TEST = 10_000, 2_000, 2_000


def main() -> None:
    tr = FashionMNIST(RAW, train=True, download=True)
    te = FashionMNIST(RAW, train=False, download=True)
    x, y = tr.data.numpy(), tr.targets.numpy()
    xt, yt = te.data.numpy(), te.targets.numpy()

    x_pool, _, y_pool, _ = train_test_split(x, y, train_size=N_TRAIN + N_VAL, stratify=y, random_state=SEED)
    x_tr, x_va, y_tr, y_va = train_test_split(x_pool, y_pool, train_size=N_TRAIN, stratify=y_pool, random_state=SEED)
    x_te, _, y_te, _ = train_test_split(xt, yt, train_size=N_TEST, stratify=yt, random_state=SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, arr in dict(x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va, x_test=x_te, y_test=y_te).items():
        np.save(OUT / f"{name}.npy", arr)
    print("Tersimpan di", OUT, "| train/val/test =", len(y_tr), len(y_va), len(y_te))


if __name__ == "__main__":
    main()
