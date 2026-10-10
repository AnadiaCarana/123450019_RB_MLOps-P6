"""Langkah 7a: pilih model terbaik berdasarkan VAL_ACC (bukan test), registrasi, beri alias @champion."""
import mlflow
from mlflow import MlflowClient

from common import EXPERIMENT, MODEL_NAME, get_data_hash, setup_mlflow

if __name__ == "__main__":
    setup_mlflow()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                              filter_string="tags.stage = 'final-seed'",
                              order_by=["metrics.val_acc DESC"])
    if runs.empty:
        raise SystemExit("Belum ada run final-seed. Jalankan src/final_seeds.py dulu.")
    best = runs.iloc[0]
    mv = mlflow.register_model(f"runs:/{best.run_id}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_model_version_tag(MODEL_NAME, mv.version, "data_md5", get_data_hash())
    client.set_model_version_tag(MODEL_NAME, mv.version, "selected_by", "max val_acc among final seeds")
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"Registered {MODEL_NAME} v{mv.version} dari run {best.run_id} -> alias @champion")
