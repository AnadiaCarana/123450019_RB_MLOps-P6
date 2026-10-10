"""Tarik hasil dari MLflow -> report/results.md (tabel siap tempel ke laporan). Tidak ada angka karangan."""
import mlflow

from common import EXPERIMENT, ROOT, setup_mlflow


def fmt(row, key):
    m, s = row.get(f"metrics.{key}_mean"), row.get(f"metrics.{key}_std")
    return "-" if m != m or m is None else f"{m:.4f} ± {s:.4f}"


if __name__ == "__main__":
    setup_mlflow()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"])
    lines = ["# Hasil (otomatis dari MLflow)\n"]

    def table(title, stage, keys, name_col="tags.mlflow.runName"):
        sub = df[df["tags.stage"] == stage].drop_duplicates(subset=[name_col])
        lines.append(f"\n## {title}\n")
        lines.append("| Run | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for _, r in sub.iterrows():
            lines.append(f"| {r[name_col]} | " + " | ".join(fmt(r, k) for k in keys) + " |")

    table("Baseline (3 seed, validation)", "baseline", ["val_acc", "val_macro_f1"])
    table("Konfigurasi terbaik (3 seed)", "final", ["val_acc", "val_macro_f1", "test_acc", "test_macro_f1"])
    table("Ablasi (3 seed, validation)", "ablation", ["val_acc", "val_macro_f1"])

    tuning = df[df["tags.stage"] == "tuning-trial"]
    lines.append(f"\n## Tuning\n\nJumlah trial tercatat: {len(tuning)}")
    out = ROOT / "report" / "results.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
