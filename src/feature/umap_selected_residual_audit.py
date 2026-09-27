from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_fusion_models import build_model, hard_split, load_split_arrays  # noqa: E402


MODEL_ORDER = ["ccd_mlp", "esm_mlp", "concat_mlp", "cross_attention_residual"]
MODEL_LABELS = {
    "ccd_mlp": "CCD-only",
    "esm_mlp": "ESM-2-only",
    "concat_mlp": "Direct Concat",
    "cross_attention_residual": "Residual CA",
}
MODEL_COLORS = {
    "ccd_mlp": "#8FA3D1",
    "esm_mlp": "#63BFA5",
    "concat_mlp": "#82919E",
    "cross_attention_residual": "#C93F3A",
}
CLASS_COLORS = {0: "#55BFD3", 1: "#E95C4A"}
CLASS_LABELS = {
    "amp": {0: "Non-AMP", 1: "AMP"},
    "tox": {0: "Non-toxic", 1: "Toxic"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP audit using fixed selected Residual CA checkpoints.")
    parser.add_argument(
        "--baseline-checkpoint-root",
        default=str(REPO_ROOT / "data--final" / "fusion_models_residual_sweep_v2" / "fair_ensemble_ablation_default" / "checkpoints"),
    )
    parser.add_argument(
        "--selected-residual-root",
        default=str(REPO_ROOT / "data--final" / "fusion_models_residual_sweep_v2" / "selected_validation_ranked_candidate" / "checkpoints"),
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "data--final" / "fusion_models_residual_sweep_v2" / "selected_umap_hard"),
    )
    parser.add_argument("--tasks", nargs="+", default=["amp", "tox"], choices=["amp", "tox"])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-random-state", type=int, default=42)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "figure.dpi": 160,
        }
    )


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_paths(task: str, model: str, baseline_root: Path, selected_root: Path) -> list[Path]:
    root = selected_root if model == "cross_attention_residual" else baseline_root
    paths = sorted(root.glob(f"{task}__{model}__seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints for {task}/{model} in {root}")
    return paths


def scaler_transform(x: np.ndarray, scaler_state: dict) -> np.ndarray:
    mean = np.asarray(scaler_state["mean"], dtype=np.float32)
    std = np.asarray(scaler_state["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def make_model(checkpoint: dict, ccd_dim: int, esm_dim: int) -> torch.nn.Module:
    hyper = checkpoint["model_hyperparameters"]
    args = SimpleNamespace(
        hidden_dim=int(hyper["hidden_dim"]),
        d_model=int(hyper["d_model"]),
        heads=int(hyper["heads"]),
        esm_tokens=int(hyper["esm_tokens"]),
        dropout=float(hyper["dropout"]),
    )
    model = build_model(
        checkpoint["model_name"],
        ccd_dim=ccd_dim,
        esm_dim=esm_dim,
        group_dims=list(checkpoint["group_dims"]),
        args=args,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def latent_batch(model: torch.nn.Module, model_name: str, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
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
    raise ValueError(model_name)


def extract_latent(arrays, checkpoint: dict, batch_size: int) -> np.ndarray:
    ccd = scaler_transform(arrays.ccd.astype(np.float32), checkpoint["ccd_scaler"])
    esm = scaler_transform(arrays.esm.astype(np.float32), checkpoint["esm_scaler"])
    model = make_model(checkpoint, ccd.shape[1], esm.shape[1])
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays.y), batch_size):
            end = start + batch_size
            latent = latent_batch(
                model,
                checkpoint["model_name"],
                torch.from_numpy(ccd[start:end]),
                torch.from_numpy(esm[start:end]),
            )
            chunks.append(latent.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def load_arrays_for_checkpoint(task: str, checkpoint: dict):
    sources = checkpoint["feature_sources"]
    return load_split_arrays(
        task=task,
        split=hard_split(task),
        ccd_root=Path(sources["ccd_root"]),
        ccd_union_subdir=sources["ccd_union_subdir"],
        ccd_features=checkpoint["ccd_features"],
        esm_root=Path(sources["esm_root"]),
    )


def separation_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_scaled = StandardScaler().fit_transform(x)
    y_int = y.astype(int)
    c0 = x_scaled[y_int == 0]
    c1 = x_scaled[y_int == 1]
    center0 = c0.mean(axis=0)
    center1 = c1.mean(axis=0)
    centroid_distance = float(np.linalg.norm(center1 - center0))
    within0 = float(np.mean(np.sum((c0 - center0) ** 2, axis=1)))
    within1 = float(np.mean(np.sum((c1 - center1) ** 2, axis=1)))
    fisher = float((centroid_distance**2) / max(within0 + within1, 1e-8))
    return {
        "silhouette": float(silhouette_score(x_scaled, y_int)),
        "davies_bouldin": float(davies_bouldin_score(x_scaled, y_int)),
        "calinski_harabasz": float(calinski_harabasz_score(x_scaled, y_int)),
        "centroid_distance": centroid_distance,
        "fisher_ratio": fisher,
    }


def embed_2d(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scaled = StandardScaler().fit_transform(x)
    if umap is None:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=args.umap_random_state).fit_transform(scaled)
    reducer = umap.UMAP(
        n_neighbors=min(args.umap_neighbors, max(2, x.shape[0] - 1)),
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.umap_random_state,
    )
    try:
        return reducer.fit_transform(scaled)
    except TypeError as exc:
        if "ensure_all_finite" not in str(exc):
            raise
        import sklearn.utils.validation as sk_validation
        import umap.umap_ as umap_core

        original_check_array = umap_core.check_array
        sklearn_check_array = sk_validation.check_array

        def compat_check_array(*check_args, ensure_all_finite=None, **check_kwargs):
            if ensure_all_finite is not None and "force_all_finite" not in check_kwargs:
                check_kwargs["force_all_finite"] = ensure_all_finite
            return sklearn_check_array(*check_args, **check_kwargs)

        umap_core.check_array = compat_check_array
        try:
            reducer = umap.UMAP(
                n_neighbors=min(args.umap_neighbors, max(2, x.shape[0] - 1)),
                min_dist=args.umap_min_dist,
                metric="cosine",
                random_state=args.umap_random_state,
            )
            return reducer.fit_transform(scaled)
        finally:
            umap_core.check_array = original_check_array


def save_all(fig: plt.Figure, stem: Path) -> None:
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def plot_umap(coords: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.75))
    panels = [
        ("amp", "ccd_mlp", "AMP | CCD-only"),
        ("amp", "esm_mlp", "AMP | ESM-2-only"),
        ("amp", "concat_mlp", "AMP | Direct Concat"),
        ("amp", "cross_attention_residual", "AMP | Residual CA"),
        ("tox", "ccd_mlp", "TOX | CCD-only"),
        ("tox", "esm_mlp", "TOX | ESM-2-only"),
        ("tox", "concat_mlp", "TOX | Direct Concat"),
        ("tox", "cross_attention_residual", "TOX | Residual CA"),
    ]
    for ax, (task, model, title), letter in zip(axes.flat, panels, list("ABCDEFGH"), strict=True):
        sub = coords[(coords["task"] == task) & (coords["model"] == model)]
        for label_id in (0, 1):
            part = sub[sub["label_id"] == label_id]
            ax.scatter(
                part["umap_1"],
                part["umap_2"],
                s=7 if task == "amp" else 14,
                color=CLASS_COLORS[label_id],
                alpha=0.72,
                linewidths=0,
                label=CLASS_LABELS[task][label_id],
            )
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(length=2.5, width=0.6)
        ax.text(-0.22, 1.13, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        if model == "ccd_mlp":
            ax.legend(frameon=False, loc="upper left", markerscale=1.6)
    fig.subplots_adjust(wspace=0.42, hspace=0.58)
    save_all(fig, figure_dir / "fig_selected_umap_hard_test_representations")


def plot_metric_bars(summary: pd.DataFrame, figure_dir: Path) -> None:
    metric_specs = [
        ("silhouette", "Silhouette\nhigher better", True),
        ("davies_bouldin", "Davies-Bouldin\nlower better", False),
        ("fisher_ratio", "Fisher ratio\nhigher better", True),
    ]
    title_specs = {
        "silhouette": ("Silhouette", "higher better"),
        "davies_bouldin": ("Davies-Bouldin", "lower better"),
        "fisher_ratio": ("Fisher ratio", "higher better"),
    }
    display_labels = ["CCD", "ESM-2", "Concat", "Residual CA"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.2))
    for row_idx, task in enumerate(["amp", "tox"]):
        task_df = summary[summary["task"] == task].set_index("model").loc[MODEL_ORDER].reset_index()
        for col_idx, (metric, _, higher_better) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = task_df[f"mean_{metric}"].to_numpy()
            errors = task_df[f"std_{metric}"].fillna(0).to_numpy()
            best = values.max() if higher_better else values.min()
            is_best = values >= best - 5e-4 if higher_better else values <= best + 5e-4
            x = np.arange(len(MODEL_ORDER))
            ax.bar(
                x,
                values,
                yerr=errors,
                capsize=2.2,
                color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                edgecolor=["black" if flag else "white" for flag in is_best],
                linewidth=[1.0 if flag else 0.5 for flag in is_best],
                zorder=3,
            )
            top = float(np.max(values + errors))
            bottom = float(min(0.0, np.min(values - errors)))
            y_span = max(top - bottom, 1e-6)
            label_offset = y_span * 0.045
            for xi, yi, err in zip(x, values, errors, strict=True):
                ax.text(
                    xi,
                    yi + err + label_offset,
                    f"{yi:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    rotation=0,
                )
            top = float(np.max(values + errors))
            ax.set_ylim(0, top + y_span * 0.30 if top > 0 else 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=7.2)
            title, subtitle = title_specs[metric]
            ax.set_title(f"{task.upper()} | {title}", fontsize=9.5, pad=13)
            ax.text(
                0.5,
                1.025,
                subtitle,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#5B6670",
            )
            if col_idx == 0:
                ax.set_ylabel("Score")
            ax.grid(axis="y", color="#DDE4EA", lw=0.5, alpha=0.9)
            ax.set_axisbelow(True)
            ax.text(
                -0.16,
                1.13,
                list("ABCDEF")[row_idx * 3 + col_idx],
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
            )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.10, wspace=0.30, hspace=0.58)
    save_all(fig, figure_dir / "fig_selected_umap_geometric_separation_bars")


def main() -> None:
    args = parse_args()
    setup_style()
    out_root = Path(args.output_root)
    source_dir = out_root / "source_data"
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    report_dir = out_root / "reports"
    for path in (source_dir, table_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(args.baseline_checkpoint_root)
    selected_root = Path(args.selected_residual_root)
    metric_rows: list[dict[str, object]] = []
    coord_rows: list[pd.DataFrame] = []
    manifest = {
        "baseline_checkpoint_root": str(baseline_root.resolve()),
        "selected_residual_root": str(selected_root.resolve()),
        "tasks": {},
    }

    for task in args.tasks:
        manifest["tasks"][task] = {}
        for model in MODEL_ORDER:
            paths = checkpoint_paths(task, model, baseline_root, selected_root)
            checkpoints = [load_checkpoint(path) for path in paths]
            arrays = load_arrays_for_checkpoint(task, checkpoints[0])
            latents = []
            seeds = []
            for checkpoint, path in zip(checkpoints, paths, strict=True):
                latent = extract_latent(arrays, checkpoint, args.batch_size)
                latents.append(latent)
                seeds.append(int(checkpoint["seed"]))
                metrics = separation_metrics(latent, arrays.y)
                metric_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": int(checkpoint["seed"]),
                        "checkpoint": str(path.resolve()),
                        **metrics,
                    }
                )
            mean_latent = np.mean(np.stack(latents, axis=0), axis=0)
            emb = embed_2d(mean_latent, args)
            coord_rows.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "fasta_id": arrays.fasta_id,
                        "label_id": arrays.y.astype(int),
                        "label": [CLASS_LABELS[task][int(v)] for v in arrays.y],
                        "umap_1": emb[:, 0],
                        "umap_2": emb[:, 1],
                    }
                )
            )
            manifest["tasks"][task][model] = {
                "seeds": seeds,
                "n_checkpoints": len(paths),
                "n_samples": int(len(arrays.y)),
            }

    metrics_by_seed = pd.DataFrame(metric_rows)
    metric_cols = ["silhouette", "davies_bouldin", "calinski_harabasz", "centroid_distance", "fisher_ratio"]
    summary_rows = []
    for (task, model), sub in metrics_by_seed.groupby(["task", "model"], sort=False):
        row = {"task": task, "model": model, "model_label": MODEL_LABELS[model], "n_checkpoints": int(sub["seed"].nunique())}
        for metric in metric_cols:
            row[f"mean_{metric}"] = float(sub[metric].mean())
            row[f"std_{metric}"] = float(sub[metric].std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    coords = pd.concat(coord_rows, ignore_index=True)

    metrics_by_seed.to_csv(table_dir / "umap_separation_metrics_by_checkpoint.csv", index=False)
    summary.to_csv(table_dir / "umap_separation_metrics_summary.csv", index=False)
    coords.to_csv(source_dir / "umap_embedding_2d_source.csv", index=False)
    (report_dir / "umap_selected_residual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plot_umap(coords, figure_dir)
    plot_metric_bars(summary, figure_dir)
    print(f"[DONE] UMAP coordinates -> {source_dir / 'umap_embedding_2d_source.csv'}")
    print(f"[DONE] metric summary -> {table_dir / 'umap_separation_metrics_summary.csv'}")
    print(f"[DONE] figures -> {figure_dir}")


if __name__ == "__main__":
    main()
