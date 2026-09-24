from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

from train_fusion_models import (
    REPO_ROOT,
    SplitArrays,
    build_model,
    grouped_feature_order,
    hard_split,
    load_split_arrays,
    predict,
    select_device,
    set_seed,
    standardize_splits,
)


MODEL_ORDER = ["ccd_mlp", "esm_mlp", "concat_mlp", "cross_attention_residual"]
MODEL_LABELS = {
    "ccd_mlp": "CCD-MLP",
    "esm_mlp": "ESM2-MLP",
    "concat_mlp": "Direct Concat",
    "cross_attention_residual": "Residual CA",
}
TASK_LABELS = {
    "amp": {0: "Non-AMP", 1: "AMP"},
    "tox": {0: "Non-toxic", 1: "Toxic"},
}
COLORS = {0: "#4DBBD5", 1: "#E64B35"}
METRIC_DIRECTIONS = {
    "silhouette": "higher",
    "davies_bouldin": "lower",
    "calinski_harabasz": "higher",
    "centroid_distance": "higher",
    "fisher_ratio": "higher",
    "linear_probe_auc": "higher",
    "linear_probe_auprc": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP and separation audit for CCD/ESM/fusion hard-test representations.")
    parser.add_argument("--tasks", nargs="+", default=["amp", "tox"], choices=["amp", "tox"])
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--plot-seed", type=int, default=13)
    parser.add_argument("--ccd-root", default=str(REPO_ROOT / "data--final" / "feature_physchem"))
    parser.add_argument("--ccd-union-subdir", default="union_final_groups")
    parser.add_argument(
        "--ccd-feature-json",
        default=str(REPO_ROOT / "data--final" / "feature_physchem" / "reports" / "final_lightgbm_compact_feature_union.json"),
    )
    parser.add_argument("--esm-root", default=str(REPO_ROOT / "data--final" / "feature_esm2"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data--final" / "fusion_umap_hard"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--esm-tokens", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2D333B",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_loader(arrays: SplitArrays, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays.ccd),
        torch.from_numpy(arrays.esm),
        torch.from_numpy(arrays.y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def train_best_model(
    task: str,
    model_name: str,
    seed: int,
    splits: dict[str, SplitArrays],
    group_dims: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, object]], float]:
    set_seed(seed)
    batch_size = min(args.batch_size, max(16, splits["train"].y.shape[0]))
    train_loader = make_loader(splits["train"], batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(splits["val"], batch_size, shuffle=False, num_workers=args.num_workers)
    ccd_dim = splits["train"].ccd.shape[1]
    esm_dim = splits["train"].esm.shape[1]
    model = build_model(model_name, ccd_dim, esm_dim, group_dims, args).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -math.inf
    patience_counter = 0
    max_epochs = 3 if args.smoke else args.epochs
    patience = 2 if args.smoke else args.patience
    history: list[dict[str, object]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for ccd, esm, y in train_loader:
            ccd = ccd.to(device, non_blocking=True)
            esm = esm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ccd, esm)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        y_val, p_val, val_loss = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(y_val, p_val))
        scheduler.step(val_auc)
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append(
            {
                "task": task,
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": val_auc,
            }
        )
        print(f"[TRAIN] {task}/{model_name}/seed{seed} epoch={epoch:02d} train_loss={train_loss:.4f} val_auc={val_auc:.4f}")
        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_val_auc


def extract_latent_batch(model: nn.Module, model_name: str, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
    if model_name == "ccd_mlp":
        return model.net[:-1](ccd)
    if model_name == "esm_mlp":
        return model.net[:-1](esm)
    if model_name == "concat_mlp":
        return model.net[:-1](torch.cat([ccd, esm], dim=1))
    if model_name == "cross_attention_residual":
        ccd_tokens = model.ccd_tokens_from_vector(ccd)
        esm_tokens = model.esm_tokens_from_vector(esm)
        esm_context, _ = model.esm_to_ccd(esm_tokens, ccd_tokens, ccd_tokens, need_weights=False)
        ccd_context, _ = model.ccd_to_esm(ccd_tokens, esm_tokens, esm_tokens, need_weights=False)
        esm_fused = model.esm_norm(esm_tokens + model.drop(esm_context))
        ccd_fused = model.ccd_norm(ccd_tokens + model.drop(ccd_context))
        cross_pooled = torch.cat(
            [
                esm_fused.mean(dim=1),
                esm_fused.amax(dim=1),
                ccd_fused.mean(dim=1),
                ccd_fused.amax(dim=1),
            ],
            dim=1,
        )
        fused = torch.cat(
            [
                model.cross_tower(cross_pooled),
                model.ccd_tower(ccd),
                model.esm_tower(esm),
            ],
            dim=1,
        )
        return model.head[:-1](fused)
    raise ValueError(f"Unsupported model: {model_name}")


def extract_latent(
    model: nn.Module,
    model_name: str,
    arrays: SplitArrays,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(arrays, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for ccd, esm, _ in loader:
            ccd = ccd.to(device, non_blocking=True)
            esm = esm.to(device, non_blocking=True)
            latent = extract_latent_batch(model, model_name, ccd, esm)
            chunks.append(latent.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def compute_separation_metrics(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    x_scaled = StandardScaler().fit_transform(x)
    y_int = y.astype(int)
    class0 = x_scaled[y_int == 0]
    class1 = x_scaled[y_int == 1]
    centroid0 = class0.mean(axis=0)
    centroid1 = class1.mean(axis=0)
    centroid_distance = float(np.linalg.norm(centroid1 - centroid0))
    within0 = float(np.mean(np.sum((class0 - centroid0) ** 2, axis=1))) if class0.size else 0.0
    within1 = float(np.mean(np.sum((class1 - centroid1) ** 2, axis=1))) if class1.size else 0.0
    fisher_ratio = float((centroid_distance**2) / max(within0 + within1, 1e-8))
    min_class = int(np.bincount(y_int).min())
    n_splits = min(5, min_class)
    probe_auc = np.nan
    probe_auprc = np.nan
    if n_splits >= 2:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        prob = cross_val_predict(clf, x, y_int, cv=cv, method="predict_proba")[:, 1]
        probe_auc = float(roc_auc_score(y_int, prob))
        probe_auprc = float(average_precision_score(y_int, prob))
    return {
        "silhouette": float(silhouette_score(x_scaled, y_int, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(x_scaled, y_int)),
        "calinski_harabasz": float(calinski_harabasz_score(x_scaled, y_int)),
        "centroid_distance": centroid_distance,
        "fisher_ratio": fisher_ratio,
        "linear_probe_auc": probe_auc,
        "linear_probe_auprc": probe_auprc,
    }


def make_umap_embedding(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if umap is None:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=args.umap_random_state).fit_transform(StandardScaler().fit_transform(x))
    n_neighbors = min(args.umap_neighbors, max(2, x.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.umap_random_state,
    )
    return reducer.fit_transform(StandardScaler().fit_transform(x))


def save_figure(fig: plt.Figure, stem: str, figure_dir: Path) -> None:
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(figure_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_umap(plot_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.8))
    for row_idx, task in enumerate(["amp", "tox"]):
        for col_idx, model_name in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = plot_df[(plot_df["task"] == task) & (plot_df["model"] == model_name)]
            for label_id in [0, 1]:
                part = sub[sub["label_id"] == label_id]
                ax.scatter(
                    part["umap_1"],
                    part["umap_2"],
                    s=4.0 if task == "amp" else 8.0,
                    c=COLORS[label_id],
                    alpha=0.72,
                    linewidths=0,
                    label=TASK_LABELS[task][label_id],
                )
            ax.set_title(f"{task.upper()} | {MODEL_LABELS[model_name]}", pad=5)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.tick_params(length=2.0, width=0.5)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc="upper left", markerscale=2.2)
            if row_idx == 1 and col_idx == 0:
                ax.legend(frameon=False, loc="upper left", markerscale=2.2)
    letters = list("ABCDEFGH")
    for ax, letter in zip(axes.ravel(), letters, strict=True):
        ax.text(-0.20, 1.12, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.42, hspace=0.58)
    save_figure(fig, "fig_umap_hard_test_representations", figure_dir)


def summarize_metrics(metric_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = list(METRIC_DIRECTIONS)
    rows = []
    for (task, model), sub in metric_df.groupby(["task", "model"], sort=False):
        row: dict[str, object] = {"task": task, "model": model, "model_label": MODEL_LABELS[model]}
        for metric in metric_cols:
            row[f"mean_{metric}"] = float(sub[metric].mean())
            row[f"std_{metric}"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def residual_vs_concat_summary(summary_df: pd.DataFrame, eps: float = 0.005) -> pd.DataFrame:
    rows = []
    for task in summary_df["task"].drop_duplicates().tolist():
        sub = summary_df[summary_df["task"] == task]
        if not {"cross_attention_residual", "concat_mlp"}.issubset(set(sub["model"])):
            continue
        residual = sub[sub["model"] == "cross_attention_residual"].iloc[0]
        concat = sub[sub["model"] == "concat_mlp"].iloc[0]
        for metric, direction in METRIC_DIRECTIONS.items():
            res = float(residual[f"mean_{metric}"])
            con = float(concat[f"mean_{metric}"])
            delta = res - con if direction == "higher" else con - res
            status = "tie" if abs(delta) <= eps else ("win" if delta > 0 else "loss")
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "direction": direction,
                    "residual_mean": res,
                    "concat_mean": con,
                    "beneficial_delta": delta,
                    "status_eps0005": status,
                }
            )
    return pd.DataFrame(rows)


def write_report(summary_df: pd.DataFrame, rvsc: pd.DataFrame, report_path: Path) -> None:
    status_counts = rvsc.groupby(["task", "status_eps0005"]).size().unstack(fill_value=0)
    lines = [
        "# UMAP Representation Audit Report",
        "",
        "Hard-test representation audit for CCD-MLP, ESM2-MLP, Direct Concat, and Residual CA.",
        "",
        "## Residual CA versus Direct Concat",
        "",
        "A +/-0.005 tolerance is used for near-tie calls. For Davies-Bouldin, lower is better; for all other metrics, higher is better.",
        "",
        status_counts.to_markdown(),
        "",
        "## Mean separation metrics",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Interpretation guardrail",
        "",
        "Use the UMAP panels as visual support and the separation metrics as the primary quantitative evidence. Avoid claiming that UMAP alone proves model superiority.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    device = select_device(args.device)
    output_root = Path(args.output_root).resolve()
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    source_dir = output_root / "source_data"
    report_dir = output_root / "reports"
    for path in [table_dir, figure_dir, source_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    ccd_features_raw = json.loads(Path(args.ccd_feature_json).read_text(encoding="utf-8"))
    ccd_features, ccd_groups, group_dims = grouped_feature_order(ccd_features_raw)

    all_metrics: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    plot_frames: list[pd.DataFrame] = []
    embedding_frames: list[pd.DataFrame] = []

    for task in args.tasks:
        raw_splits = {
            split: load_split_arrays(
                task=task,
                split=split,
                ccd_root=Path(args.ccd_root),
                ccd_union_subdir=args.ccd_union_subdir,
                ccd_features=ccd_features,
                esm_root=Path(args.esm_root),
            )
            for split in ("train", "val", hard_split(task))
        }
        splits = standardize_splits(raw_splits)
        hard_arrays = splits[hard_split(task)]
        batch_size = min(args.batch_size, max(16, hard_arrays.y.shape[0]))
        for model_name in args.models:
            for seed in args.seeds:
                model, history, best_val_auc = train_best_model(task, model_name, seed, splits, group_dims, args, device)
                all_history.extend(history)
                latent = extract_latent(model, model_name, hard_arrays, batch_size, device)
                metrics = compute_separation_metrics(latent, hard_arrays.y, seed=seed)
                all_metrics.append(
                    {
                        "task": task,
                        "split": hard_split(task),
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "seed": seed,
                        "best_val_auc": best_val_auc,
                        "n": int(hard_arrays.y.shape[0]),
                        "latent_dim": int(latent.shape[1]),
                        **metrics,
                    }
                )
                if seed == args.plot_seed:
                    latent_df = pd.DataFrame(latent, columns=[f"latent_{i:03d}" for i in range(latent.shape[1])])
                    latent_df.insert(0, "label_id", hard_arrays.y.astype(int))
                    latent_df.insert(0, "fasta_id", hard_arrays.fasta_id)
                    latent_df.insert(0, "seed", seed)
                    latent_df.insert(0, "model", model_name)
                    latent_df.insert(0, "task", task)
                    embedding_frames.append(latent_df)

                    xy = make_umap_embedding(latent, args)
                    plot_df = pd.DataFrame(
                        {
                            "task": task,
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "seed": seed,
                            "fasta_id": hard_arrays.fasta_id,
                            "label_id": hard_arrays.y.astype(int),
                            "label": [TASK_LABELS[task][int(v)] for v in hard_arrays.y],
                            "umap_1": xy[:, 0],
                            "umap_2": xy[:, 1],
                        }
                    )
                    plot_frames.append(plot_df)

    metric_df = pd.DataFrame(all_metrics)
    summary_df = summarize_metrics(metric_df)
    rvsc = residual_vs_concat_summary(summary_df)
    history_df = pd.DataFrame(all_history)
    plot_df = pd.concat(plot_frames, ignore_index=True)
    embedding_df = pd.concat(embedding_frames, ignore_index=True)

    metric_df.to_csv(table_dir / "umap_separation_metrics_by_seed.csv", index=False)
    summary_df.to_csv(table_dir / "umap_separation_metrics_summary.csv", index=False)
    rvsc.to_csv(table_dir / "umap_residual_vs_concat_audit.csv", index=False)
    history_df.to_csv(source_dir / "umap_training_history.csv", index=False)
    plot_df.to_csv(source_dir / "umap_embedding_2d_source.csv", index=False)
    embedding_df.to_parquet(source_dir / "hard_test_latent_embeddings_plot_seed.parquet", index=False)
    plot_umap(plot_df, figure_dir)
    write_report(summary_df, rvsc, report_dir / "umap_representation_report_20260611.md")

    manifest = {
        "device": str(device),
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "plot_seed": args.plot_seed,
        "umap_available": umap is not None,
        "ccd_groups": {group: len(features) for group, features in ccd_groups.items()},
        "args": vars(args),
    }
    (report_dir / "umap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] UMAP figure -> {figure_dir / 'fig_umap_hard_test_representations.svg'}")
    print(f"[DONE] metrics summary -> {table_dir / 'umap_separation_metrics_summary.csv'}")
    print(f"[DONE] report -> {report_dir / 'umap_representation_report_20260611.md'}")


if __name__ == "__main__":
    main()
