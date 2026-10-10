"""Langkah 5: konfigurasi terbaik x 3 seed -> mean ± std. SATU-SATUNYA skrip yang menyentuh test set."""
from common import load_best_config, load_data, run_seeds, setup_mlflow

if __name__ == "__main__":
    setup_mlflow()
    data = load_data()
    cfg = load_best_config()
    rid, s = run_seeds("final-best-config", "final", cfg, data, eval_test=True, log_model=True,
                       note="Konfigurasi terbaik hasil Optuna, 3 seed. Test set dievaluasi sekali di sini.")
    print("run_id:", rid)
    for k, v in sorted(s.items()):
        print(f"{k:22s} {v:.4f}")
