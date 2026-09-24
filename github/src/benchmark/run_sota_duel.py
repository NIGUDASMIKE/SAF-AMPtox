from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired SOTA dual-screening baselines on AMP/TOX hard tests.")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "data--final" / "sota_duel"))
    parser.add_argument("--venv-python", default=str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    parser.add_argument("--amp-hard-csv", default=str(REPO_ROOT / "data--final" / "splits" / "test_hard_amp.csv"))
    parser.add_argument("--tox-hard-csv", default=str(REPO_ROOT / "data--final" / "tox" / "splits" / "test_hard_tox.csv"))
    parser.add_argument("--amplify-root", default=str(REPO_ROOT / "sota" / "AMPlify-master"))
    parser.add_argument("--toxinpred-root", default=str(REPO_ROOT / "sota" / "toxinpred3-main"))
    parser.add_argument("--macrel-exe", default=str(REPO_ROOT / ".venv" / "Scripts" / "macrel.exe"))
    parser.add_argument(
        "--ours-summary",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_compare" / "reports" / "fusion_model_summary.csv"),
    )
    parser.add_argument("--skip-sota", action="store_true", help="Only regenerate tables from existing predictions.")
    return parser.parse_args()


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    paths = {
        "inputs": out_root / "inputs",
        "predictions": out_root / "predictions",
        "tables": out_root / "tables",
        "reports": out_root / "reports",
        "logs": out_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            seq = clean_sequence(row["sequence"])
            if set(seq) - STANDARD_AA:
                raise ValueError(f"Non-standard sequence in hard test: {row['fasta_id']} {seq}")
            handle.write(f">{row['fasta_id']}\n{seq}\n")


def unzip_toxinpred_model(toxinpred_root: Path) -> None:
    model_path = toxinpred_root / "model" / "toxinpred3.0_model.pkl"
    zip_path = toxinpred_root / "model" / "toxinpred3.0_model.pkl.zip"
    if model_path.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    shutil.unpack_archive(str(zip_path), str(zip_path.parent))
    if not model_path.exists():
        raise RuntimeError(f"Failed to unpack {zip_path} into {model_path}")


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    env = os.environ.copy()
    for key in ["PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME"]:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        proc = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See {log_path}")


def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[0]


def run_amplify(model_kind: str, fasta_path: Path, out_dir: Path, log_dir: Path, args: argparse.Namespace) -> Path:
    model_dir = Path(args.amplify_root)
    script = model_dir / "src" / "AMPlify.py"
    before = set(out_dir.glob(f"AMPlify_{model_kind}_results_*.tsv"))
    command = [
        str(Path(args.venv_python)),
        str(script),
        "-m",
        model_kind,
        "-s",
        str(fasta_path.resolve()),
        "-od",
        str(out_dir.resolve()),
        "-of",
        "tsv",
        "-sub",
        "off",
        "-att",
        "off",
    ]
    run_command(command, cwd=model_dir / "src", log_path=log_dir / f"amplify_{model_kind}.log")
    after = set(out_dir.glob(f"AMPlify_{model_kind}_results_*.tsv"))
    new_files = list(after - before)
    if new_files:
        return sorted(new_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return latest_file(out_dir, f"AMPlify_{model_kind}_results_*.tsv")


def run_toxinpred(model_id: int, fasta_path: Path, out_file: Path, log_dir: Path, args: argparse.Namespace) -> Path:
    toxinpred_root = Path(args.toxinpred_root)
    unzip_toxinpred_model(toxinpred_root)
    command = [
        str(Path(args.venv_python)),
        str((toxinpred_root / "toxinpred3.py").resolve()),
        "-i",
        str(fasta_path.resolve()),
        "-o",
        str(out_file.resolve()),
        "-m",
        str(model_id),
        "-d",
        "2",
    ]
    run_command(command, cwd=toxinpred_root, log_path=log_dir / f"toxinpred_model{model_id}.log")
    if not out_file.exists():
        raise FileNotFoundError(out_file)
    return out_file


def run_macrel(fasta_path: Path, out_dir: Path, tag: str, log_dir: Path, args: argparse.Namespace) -> Path:
    command = [
        str(Path(args.macrel_exe)),
        "peptides",
        "--fasta",
        str(fasta_path.resolve()),
        "--output",
        str(out_dir.resolve()),
        "--tag",
        tag,
        "--keep-negatives",
        "--force",
    ]
    run_command(command, cwd=REPO_ROOT, log_path=log_dir / f"macrel_{tag}.log")
    pred_path = out_dir / f"{tag}.prediction.gz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    return pred_path


def load_amplify_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    out = df[["Sequence_ID", "Probability_score", "Prediction"]].copy()
    out.columns = ["fasta_id", "score", "prediction"]
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out


def load_toxinpred_predictions(path: Path, model_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if model_id == 1:
        score_col = "ML Score"
    else:
        score_col = "Hybrid Score"
    id_col = "ID" if "ID" in df.columns else "Subject"
    out = df[[id_col, score_col, "Prediction"]].copy()
    out.columns = ["fasta_id", "score", "prediction"]
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out


def load_macrel_predictions(path: Path, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", compression="gzip")
    if "Access" not in df.columns or score_col not in df.columns:
        raise ValueError(f"Macrel output {path} lacks Access or {score_col}")
    out = df[["Access", score_col]].copy()
    out.columns = ["fasta_id", "score"]
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out


def metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_prediction(task_df: pd.DataFrame, pred_df: pd.DataFrame, threshold: float) -> dict[str, float]:
    merged = task_df[["fasta_id", "label_id"]].merge(pred_df[["fasta_id", "score"]], on="fasta_id", how="left")
    missing = merged["score"].isna().sum()
    if missing:
        raise RuntimeError(f"Missing predictions for {missing} sequences")
    return metrics_from_scores(merged["label_id"].to_numpy(dtype=int), merged["score"].to_numpy(dtype=float), threshold)


def ours_metrics(summary_path: Path) -> dict[str, dict[str, float]]:
    summary = pd.read_csv(summary_path)
    out: dict[str, dict[str, float]] = {}
    mapping = {"amp": "test_hard_amp", "tox": "test_hard_tox"}
    for task, split in mapping.items():
        row = summary[
            (summary["task"] == task)
            & (summary["split"] == split)
            & (summary["model"] == "cross_attention_residual")
        ].iloc[0]
        out[task] = {
            metric: float(row[f"mean_{metric}"])
            for metric in [
                "sensitivity",
                "specificity",
                "accuracy",
                "precision",
                "npv",
                "balanced_accuracy",
                "f1",
                "mcc",
                "roc_auc",
                "pr_auc",
            ]
        }
    return out


def paired_rows(
    amp_metrics: dict[str, dict[str, float]],
    tox_metrics: dict[str, dict[str, float]],
    ours: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for amp_name, amp_m in amp_metrics.items():
        for tox_name, tox_m in tox_metrics.items():
            model = f"{amp_name} + {tox_name}"
            rows.append(build_dual_row(model, "Paired SOTA baseline", amp_m, tox_m))
    rows.append(build_dual_row("Residual Cross-Attention", "Proposed model", ours["amp"], ours["tox"]))
    table = pd.DataFrame(rows)
    return table.sort_values(["macro_roc_auc", "macro_pr_auc"], ascending=False).reset_index(drop=True)


def build_dual_row(model: str, category: str, amp_m: dict[str, float], tox_m: dict[str, float]) -> dict[str, float | str]:
    return {
        "model": model,
        "category": category,
        "amp_sensitivity": amp_m["sensitivity"],
        "amp_specificity": amp_m["specificity"],
        "amp_accuracy": amp_m["accuracy"],
        "amp_precision": amp_m["precision"],
        "amp_roc_auc": amp_m["roc_auc"],
        "amp_pr_auc": amp_m["pr_auc"],
        "amp_f1": amp_m["f1"],
        "amp_mcc": amp_m["mcc"],
        "tox_sensitivity": tox_m["sensitivity"],
        "tox_specificity": tox_m["specificity"],
        "tox_accuracy": tox_m["accuracy"],
        "tox_precision": tox_m["precision"],
        "tox_roc_auc": tox_m["roc_auc"],
        "tox_pr_auc": tox_m["pr_auc"],
        "tox_f1": tox_m["f1"],
        "tox_mcc": tox_m["mcc"],
        "macro_sensitivity": (amp_m["sensitivity"] + tox_m["sensitivity"]) / 2,
        "macro_specificity": (amp_m["specificity"] + tox_m["specificity"]) / 2,
        "macro_accuracy": (amp_m["accuracy"] + tox_m["accuracy"]) / 2,
        "macro_precision": (amp_m["precision"] + tox_m["precision"]) / 2,
        "macro_roc_auc": (amp_m["roc_auc"] + tox_m["roc_auc"]) / 2,
        "macro_pr_auc": (amp_m["pr_auc"] + tox_m["pr_auc"]) / 2,
        "macro_f1": (amp_m["f1"] + tox_m["f1"]) / 2,
        "macro_mcc": (amp_m["mcc"] + tox_m["mcc"]) / 2,
        "worst_task_roc_auc": min(amp_m["roc_auc"], tox_m["roc_auc"]),
        "worst_task_pr_auc": min(amp_m["pr_auc"], tox_m["pr_auc"]),
    }


def fasta_sequences(path: Path) -> list[str]:
    seqs = []
    current = []
    if not path.exists():
        return seqs
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                seqs.append(clean_sequence("".join(current)))
                current = []
        else:
            current.append(line)
    if current:
        seqs.append(clean_sequence("".join(current)))
    return seqs


def csv_sequences(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for column in ["sequence", "Sequence", "seq", "Seq"]:
        if column in df.columns:
            return [clean_sequence(x) for x in df[column].dropna().astype(str).tolist()]
    if len(df.columns) == 1:
        headerless = pd.read_csv(path, header=None)
        return [clean_sequence(x) for x in headerless.iloc[:, 0].dropna().astype(str).tolist()]
    return []


def sequence_bias_stats(seqs: list[str]) -> dict[str, float | int]:
    seqs = [seq for seq in seqs if seq]
    n = len(seqs)
    if n == 0:
        return {
            "n": 0,
            "m_start_rate": math.nan,
            "contains_c_rate": math.nan,
            "mean_c_frequency": math.nan,
        }
    return {
        "n": n,
        "m_start_rate": sum(seq.startswith("M") for seq in seqs) / n,
        "contains_c_rate": sum("C" in seq for seq in seqs) / n,
        "mean_c_frequency": float(np.mean([seq.count("C") / len(seq) for seq in seqs])),
    }


def build_shortcut_bias_table(amp_hard: pd.DataFrame, tox_hard: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    amplify_root = Path(args.amplify_root)
    toxinpred_root = Path(args.toxinpred_root)

    amp_sets = [
        ("AMPlify train AMP", fasta_sequences(amplify_root / "data" / "AMPlify_AMP_train_common.fa"), "AMP", "positive"),
        ("AMPlify train non-AMP balanced", fasta_sequences(amplify_root / "data" / "AMPlify_non_AMP_train_balanced.fa"), "AMP", "negative"),
        ("AMPlify train non-AMP imbalanced", fasta_sequences(amplify_root / "data" / "AMPlify_non_AMP_train_imbalanced.fa"), "AMP", "negative"),
        ("AMPlify test AMP", fasta_sequences(amplify_root / "data" / "AMPlify_AMP_test_common.fa"), "AMP", "positive"),
        ("AMPlify test non-AMP balanced", fasta_sequences(amplify_root / "data" / "AMPlify_non_AMP_test_balanced.fa"), "AMP", "negative"),
        ("AMPlify test non-AMP imbalanced", fasta_sequences(amplify_root / "data" / "AMPlify_non_AMP_test_imbalanced.fa"), "AMP", "negative"),
        ("Our AMP hard positive", amp_hard.loc[amp_hard["label_id"] == 1, "sequence"].astype(str).tolist(), "AMP", "positive"),
        ("Our AMP hard negative", amp_hard.loc[amp_hard["label_id"] == 0, "sequence"].astype(str).tolist(), "AMP", "negative"),
    ]
    tox_sets = [
        ("ToxinPred3 train toxic", csv_sequences(toxinpred_root / "dataset" / "train_pos.csv"), "TOX", "positive"),
        ("ToxinPred3 train non-toxic", csv_sequences(toxinpred_root / "dataset" / "train_neg.csv"), "TOX", "negative"),
        ("ToxinPred3 test toxic", csv_sequences(toxinpred_root / "dataset" / "test_pos.csv"), "TOX", "positive"),
        ("ToxinPred3 test non-toxic", csv_sequences(toxinpred_root / "dataset" / "test_neg.csv"), "TOX", "negative"),
        ("Our TOX hard toxic", tox_hard.loc[tox_hard["label_id"] == 1, "sequence"].astype(str).tolist(), "TOX", "positive"),
        ("Our TOX hard non-toxic", tox_hard.loc[tox_hard["label_id"] == 0, "sequence"].astype(str).tolist(), "TOX", "negative"),
    ]
    for dataset, seqs, task, label in amp_sets + tox_sets:
        row = {"dataset": dataset, "task": task, "label_group": label}
        row.update(sequence_bias_stats(seqs))
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(path: Path, title: str, table: pd.DataFrame) -> None:
    path.write_text(f"# {title}\n\n{table.to_markdown(index=False)}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    paths = ensure_dirs(out_root)
    amp_hard = pd.read_csv(args.amp_hard_csv)
    tox_hard = pd.read_csv(args.tox_hard_csv)
    amp_fasta = paths["inputs"] / "amp_hard.fa"
    tox_fasta = paths["inputs"] / "tox_hard.fa"
    write_fasta(amp_hard, amp_fasta)
    write_fasta(tox_hard, tox_fasta)

    amp_pred_paths: dict[str, Path] = {}
    tox_pred_paths: dict[str, Path] = {}
    macrel_amp_pred_path: Path
    macrel_tox_pred_path: Path
    if not args.skip_sota:
        for model_kind in ["balanced", "imbalanced"]:
            amp_pred_paths[model_kind] = run_amplify(model_kind, amp_fasta, paths["predictions"], paths["logs"], args)
        for model_id in [1, 2]:
            tox_pred_paths[f"model{model_id}"] = run_toxinpred(
                model_id,
                tox_fasta,
                paths["predictions"] / f"toxinpred3_model{model_id}_tox_hard.csv",
                paths["logs"],
                args,
            )
        macrel_amp_pred_path = run_macrel(
            amp_fasta,
            paths["predictions"] / "macrel_amp_hard",
            "amp_hard",
            paths["logs"],
            args,
        )
        macrel_tox_pred_path = run_macrel(
            tox_fasta,
            paths["predictions"] / "macrel_tox_hard",
            "tox_hard",
            paths["logs"],
            args,
        )
    else:
        amp_pred_paths = {
            "balanced": latest_file(paths["predictions"], "AMPlify_balanced_results_*.tsv"),
            "imbalanced": latest_file(paths["predictions"], "AMPlify_imbalanced_results_*.tsv"),
        }
        tox_pred_paths = {
            "model1": paths["predictions"] / "toxinpred3_model1_tox_hard.csv",
            "model2": paths["predictions"] / "toxinpred3_model2_tox_hard.csv",
        }
        macrel_amp_pred_path = paths["predictions"] / "macrel_amp_hard" / "amp_hard.prediction.gz"
        macrel_tox_pred_path = paths["predictions"] / "macrel_tox_hard" / "tox_hard.prediction.gz"

    amp_metrics = {}
    for model_kind, pred_path in amp_pred_paths.items():
        pred = load_amplify_predictions(pred_path)
        metrics = evaluate_prediction(amp_hard, pred, threshold=0.5)
        amp_metrics[f"AMPlify-{model_kind}"] = metrics
        pred.to_csv(paths["predictions"] / f"parsed_amplify_{model_kind}_amp_hard.csv", index=False)
    if macrel_amp_pred_path.exists():
        pred = load_macrel_predictions(macrel_amp_pred_path, score_col="AMP_probability")
        amp_metrics["Macrel-AMP"] = evaluate_prediction(amp_hard, pred, threshold=0.5)
        pred.to_csv(paths["predictions"] / "parsed_macrel_amp_hard.csv", index=False)

    tox_metrics = {}
    for model_key, pred_path in tox_pred_paths.items():
        model_id = 1 if model_key == "model1" else 2
        pred = load_toxinpred_predictions(pred_path, model_id=model_id)
        threshold = 0.38
        metrics = evaluate_prediction(tox_hard, pred, threshold=threshold)
        label = "ToxinPred3-ML" if model_id == 1 else "ToxinPred3-Hybrid"
        tox_metrics[label] = metrics
        pred.to_csv(paths["predictions"] / f"parsed_toxinpred3_model{model_id}_tox_hard.csv", index=False)
    if macrel_tox_pred_path.exists():
        pred = load_macrel_predictions(macrel_tox_pred_path, score_col="Hemolytic_probability")
        tox_metrics["Macrel-Hemo"] = evaluate_prediction(tox_hard, pred, threshold=0.5)
        pred.to_csv(paths["predictions"] / "parsed_macrel_hemo_tox_hard.csv", index=False)

    ours = ours_metrics(Path(args.ours_summary))
    dual = paired_rows(amp_metrics, tox_metrics, ours)
    dual_rounded = dual.copy()
    for column in dual_rounded.columns:
        if pd.api.types.is_numeric_dtype(dual_rounded[column]):
            dual_rounded[column] = dual_rounded[column].map(lambda value: round(float(value), 4))

    amp_task_table = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in amp_metrics.items()]
        + [{"model": "Residual Cross-Attention", **ours["amp"]}]
    )
    tox_task_table = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in tox_metrics.items()]
        + [{"model": "Residual Cross-Attention", **ours["tox"]}]
    )
    shortcut = build_shortcut_bias_table(amp_hard, tox_hard, args)

    dual.to_csv(paths["tables"] / "sota_dual_screening_metrics.csv", index=False)
    dual_rounded.to_csv(paths["tables"] / "sota_dual_screening_metrics_rounded.csv", index=False)
    amp_task_table.to_csv(paths["tables"] / "sota_amp_hard_task_metrics.csv", index=False)
    tox_task_table.to_csv(paths["tables"] / "sota_tox_hard_task_metrics.csv", index=False)
    shortcut.to_csv(paths["tables"] / "sota_shortcut_bias_audit.csv", index=False)
    write_markdown(paths["tables"] / "sota_dual_screening_metrics.md", "SOTA dual-screening hard-test comparison", dual_rounded)
    write_markdown(paths["tables"] / "sota_shortcut_bias_audit.md", "Shortcut-bias audit for SOTA source datasets", shortcut)

    report = {
        "amp_prediction_files": {key: str(path) for key, path in amp_pred_paths.items()},
        "tox_prediction_files": {key: str(path) for key, path in tox_pred_paths.items()},
        "macrel_amp_prediction_file": str(macrel_amp_pred_path),
        "macrel_tox_prediction_file": str(macrel_tox_pred_path),
        "main_metric_sort": "macro_roc_auc desc, macro_pr_auc desc",
        "sota_rows": dual_rounded.to_dict(orient="records"),
    }
    (paths["reports"] / "sota_duel_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] dual table -> {paths['tables'] / 'sota_dual_screening_metrics.md'}")
    print(f"[DONE] shortcut audit -> {paths['tables'] / 'sota_shortcut_bias_audit.md'}")


if __name__ == "__main__":
    main()
