"""Langkah 6: ablasi leave-one-out pada 3 komponen (augmentasi, dropout, batch norm), masing-masing 3 seed.
Dibandingkan pada VALIDATION; test set tidak dipakai."""
from dataclasses import replace

from common import load_best_config, load_data, run_seeds, setup_mlflow

VARIANTS = {
    "full": {},
    "no_augmentation": {"augment": False},
    "no_dropout": {"dropout": 0.0},
    "no_batchnorm": {"use_bn": False},
}

if __name__ == "__main__":
    setup_mlflow()
    data = load_data()
    best = load_best_config()
    for name, change in VARIANTS.items():
        rid, s = run_seeds(f"ablation-{name}", "ablation", replace(best, **change), data,
                           eval_test=False, log_model=False,
                           extra_tags={"purpose": "ablation", "ablation_variant": name},
                           note=f"Ablasi: {name} ({change or 'konfigurasi penuh'}).")
        print(f"{name:18s} val_acc = {s['val_acc_mean']:.4f} ± {s['val_acc_std']:.4f}")
