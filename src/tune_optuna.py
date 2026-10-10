"""Langkah 4: >= 20 trial Optuna; 1 trial = 1 nested (child) run MLflow. Objektif = val_acc (BUKAN test)."""
import argparse
import json
from dataclasses import asdict

import mlflow
import optuna

from common import BEST_CONFIG_PATH, Config, load_data, setup_mlflow, start_run, train_and_eval

FIXED = dict(use_bn=True, augment=True, epochs=10, seed=42)


def main(n_trials: int) -> None:
    setup_mlflow()
    data = load_data()

    def objective(trial: optuna.Trial) -> float:
        cfg = Config(
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.1, 0.6),   # >0 agar ablasi "no_dropout" bermakna
            base_channels=trial.suggest_categorical("base_channels", [16, 32, 64]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            **FIXED,
        )
        with start_run(f"trial-{trial.number:02d}", "tuning-trial", nested=True):
            mlflow.log_params(asdict(cfg))
            mlflow.set_tag("optuna_trial_number", str(trial.number))
            _, m = train_and_eval(cfg, data)
            return m["val_acc"]

    with start_run("optuna-tuning", "tuning"):
        mlflow.set_tag("mlflow.note.content", "TPE sampler (seed=42), objektif: maksimalkan val_acc.")
        mlflow.log_dict({"lr": "loguniform[1e-4,1e-2]", "batch_size": [64, 128, 256], "dropout": "uniform[0.1,0.6]",
                         "base_channels": [16, 32, 64], "weight_decay": "loguniform[1e-6,1e-3]", "fixed": FIXED},
                        "search_space.json")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best = Config(**{**FIXED, **study.best_params})
        BEST_CONFIG_PATH.write_text(json.dumps(asdict(best), indent=2))
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"best_val_acc": study.best_value, "n_trials": len(study.trials)})
        mlflow.log_artifact(str(BEST_CONFIG_PATH))
        print("Best:", study.best_params, "val_acc =", study.best_value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    main(ap.parse_args().n_trials)
