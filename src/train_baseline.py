"""Langkah 3: baseline CNN sederhana (tanpa BN, dropout, augmentasi). 3 seed, metrik per epoch, model di-log.
Hanya VALIDATION yang dipakai; test set tidak disentuh di sini."""
from common import Config, load_data, run_seeds, setup_mlflow

if __name__ == "__main__":
    setup_mlflow()
    data = load_data()
    cfg = Config()  # default = baseline
    rid, s = run_seeds("baseline-cnn", "baseline", cfg, data, eval_test=False, log_model=True,
                       note="Baseline: 2 conv + FC, tanpa BN/dropout/augmentasi, Adam lr=1e-3.")
    print("run_id:", rid, "\n", s)
