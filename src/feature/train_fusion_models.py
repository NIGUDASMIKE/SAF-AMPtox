from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_COLUMNS = {"fasta_id", "sequence", "length", "label", "label_id", "task_label", "task", "split"}


@dataclass
class SplitArrays:
    ccd: np.ndarray
    esm: np.ndarray
    y: np.ndarray
    fasta_id: list[str]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = x.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean.astype(np.float32),
            "std": self.std.astype(np.float32),
        }


class ConcatMLP(nn.Module):
    def __init__(self, ccd_dim: int, esm_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ccd_dim + esm_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ccd, esm], dim=1)).squeeze(1)


class SingleModalityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, modality: str):
        super().__init__()
        if modality not in {"ccd", "esm"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
        x = ccd if self.modality == "ccd" else esm
        return self.net(x).squeeze(1)


class GatedTwoTower(nn.Module):
    def __init__(self, ccd_dim: int, esm_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.ccd_tower = nn.Sequential(
            nn.Linear(ccd_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.esm_tower = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
        ccd_h = self.ccd_tower(ccd)
        esm_h = self.esm_tower(esm)
        gate = self.gate(torch.cat([ccd_h, esm_h], dim=1))
        fused = gate * ccd_h + (1.0 - gate) * esm_h
        return self.head(fused).squeeze(1)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        ccd_group_dims: list[int],
        esm_dim: int,
        esm_tokens: int,
        d_model: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        if esm_dim % esm_tokens != 0:
            raise ValueError(f"esm_dim={esm_dim} must be divisible by esm_tokens={esm_tokens}")
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        self.ccd_group_dims = ccd_group_dims
        self.esm_tokens = esm_tokens
        self.esm_chunk_dim = esm_dim // esm_tokens
        self.ccd_slices: list[slice] = []
        offset = 0
        for dim in ccd_group_dims:
            self.ccd_slices.append(slice(offset, offset + dim))
            offset += dim

        self.ccd_projectors = nn.ModuleList([nn.Linear(dim, d_model) for dim in ccd_group_dims])
        self.esm_projectors = nn.ModuleList([nn.Linear(self.esm_chunk_dim, d_model) for _ in range(esm_tokens)])
        self.ccd_token_embed = nn.Parameter(torch.zeros(1, len(ccd_group_dims), d_model))
        self.esm_token_embed = nn.Parameter(torch.zeros(1, esm_tokens, d_model))

        self.esm_to_ccd = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ccd_to_esm = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.esm_norm = nn.LayerNorm(d_model)
        self.ccd_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.ccd_token_embed, std=0.02)
        nn.init.normal_(self.esm_token_embed, std=0.02)

    def ccd_tokens_from_vector(self, ccd: torch.Tensor) -> torch.Tensor:
        tokens = []
        for projector, feature_slice in zip(self.ccd_projectors, self.ccd_slices):
            tokens.append(projector(ccd[:, feature_slice]))
        return torch.stack(tokens, dim=1) + self.ccd_token_embed

    def esm_tokens_from_vector(self, esm: torch.Tensor) -> torch.Tensor:
        chunks = esm.view(esm.shape[0], self.esm_tokens, self.esm_chunk_dim)
        tokens = [projector(chunks[:, index, :]) for index, projector in enumerate(self.esm_projectors)]
        return torch.stack(tokens, dim=1) + self.esm_token_embed

    def forward(self, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
        ccd_tokens = self.ccd_tokens_from_vector(ccd)
        esm_tokens = self.esm_tokens_from_vector(esm)
        esm_context, _ = self.esm_to_ccd(esm_tokens, ccd_tokens, ccd_tokens, need_weights=False)
        ccd_context, _ = self.ccd_to_esm(ccd_tokens, esm_tokens, esm_tokens, need_weights=False)
        esm_fused = self.esm_norm(esm_tokens + self.drop(esm_context))
        ccd_fused = self.ccd_norm(ccd_tokens + self.drop(ccd_context))
        pooled = torch.cat(
            [
                esm_fused.mean(dim=1),
                esm_fused.amax(dim=1),
                ccd_fused.mean(dim=1),
                ccd_fused.amax(dim=1),
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(1)


class ResidualCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        ccd_dim: int,
        esm_dim: int,
        ccd_group_dims: list[int],
        esm_tokens: int,
        d_model: int,
        heads: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if esm_dim % esm_tokens != 0:
            raise ValueError(f"esm_dim={esm_dim} must be divisible by esm_tokens={esm_tokens}")
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        self.ccd_group_dims = ccd_group_dims
        self.esm_tokens = esm_tokens
        self.esm_chunk_dim = esm_dim // esm_tokens
        self.ccd_slices: list[slice] = []
        offset = 0
        for dim in ccd_group_dims:
            self.ccd_slices.append(slice(offset, offset + dim))
            offset += dim

        self.ccd_projectors = nn.ModuleList([nn.Linear(dim, d_model) for dim in ccd_group_dims])
        self.esm_projectors = nn.ModuleList([nn.Linear(self.esm_chunk_dim, d_model) for _ in range(esm_tokens)])
        self.ccd_token_embed = nn.Parameter(torch.zeros(1, len(ccd_group_dims), d_model))
        self.esm_token_embed = nn.Parameter(torch.zeros(1, esm_tokens, d_model))
        self.esm_to_ccd = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ccd_to_esm = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.esm_norm = nn.LayerNorm(d_model)
        self.ccd_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.cross_tower = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.ccd_tower = nn.Sequential(
            nn.Linear(ccd_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.esm_tower = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ccd_token_embed, std=0.02)
        nn.init.normal_(self.esm_token_embed, std=0.02)

    def ccd_tokens_from_vector(self, ccd: torch.Tensor) -> torch.Tensor:
        tokens = []
        for projector, feature_slice in zip(self.ccd_projectors, self.ccd_slices):
            tokens.append(projector(ccd[:, feature_slice]))
        return torch.stack(tokens, dim=1) + self.ccd_token_embed

    def esm_tokens_from_vector(self, esm: torch.Tensor) -> torch.Tensor:
        chunks = esm.view(esm.shape[0], self.esm_tokens, self.esm_chunk_dim)
        tokens = [projector(chunks[:, index, :]) for index, projector in enumerate(self.esm_projectors)]
        return torch.stack(tokens, dim=1) + self.esm_token_embed

    def forward(self, ccd: torch.Tensor, esm: torch.Tensor) -> torch.Tensor:
        ccd_tokens = self.ccd_tokens_from_vector(ccd)
        esm_tokens = self.esm_tokens_from_vector(esm)
        esm_context, _ = self.esm_to_ccd(esm_tokens, ccd_tokens, ccd_tokens, need_weights=False)
        ccd_context, _ = self.ccd_to_esm(ccd_tokens, esm_tokens, esm_tokens, need_weights=False)
        esm_fused = self.esm_norm(esm_tokens + self.drop(esm_context))
        ccd_fused = self.ccd_norm(ccd_tokens + self.drop(ccd_context))
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
                self.cross_tower(cross_pooled),
                self.ccd_tower(ccd),
                self.esm_tower(esm),
            ],
            dim=1,
        )
        return self.head(fused).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural CCD/ESM fusion models with standard and hard-test evaluation.")
    parser.add_argument("--tasks", nargs="+", default=["amp", "tox"], choices=["amp", "tox"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["concat_mlp", "gated", "cross_attention"],
        choices=["ccd_mlp", "esm_mlp", "concat_mlp", "gated", "cross_attention", "cross_attention_residual"],
    )
    parser.add_argument("--ccd-root", default=str(REPO_ROOT / "data--final" / "feature_physchem"))
    parser.add_argument("--ccd-union-subdir", default="union_final_groups")
    parser.add_argument("--ccd-feature-json", default=str(REPO_ROOT / "data--final" / "feature_physchem" / "reports" / "final_lightgbm_compact_feature_union.json"))
    parser.add_argument("--esm-root", default=str(REPO_ROOT / "data--final" / "feature_esm2"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data--final" / "fusion_models"))
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use tiny epochs and batches for a fast integration check.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(device_arg)


def hard_split(task: str) -> str:
    return "test_hard_amp" if task == "amp" else "test_hard_tox"


def group_name(feature: str) -> str:
    return feature.split("_", 1)[0]


def grouped_feature_order(features: list[str]) -> tuple[list[str], dict[str, list[str]], list[int]]:
    groups: dict[str, list[str]] = {}
    group_order: list[str] = []
    for feature in features:
        group = group_name(feature)
        if group not in groups:
            groups[group] = []
            group_order.append(group)
        groups[group].append(feature)
    ordered_features = [feature for group in group_order for feature in groups[group]]
    group_dims = [len(groups[group]) for group in group_order]
    return ordered_features, groups, group_dims


def ccd_path(root: Path, task: str, union_subdir: str, split: str) -> Path:
    primary = root / task / union_subdir / f"{union_subdir}_{task}_{split}.parquet"
    if primary.exists():
        return primary
    fallback_names = [
        f"ccd_{union_subdir}_{task}_{split}.parquet",
        f"ccd_shared_union_{task}_{split}.parquet",
    ]
    for name in fallback_names:
        fallback = root / task / union_subdir / name
        if fallback.exists():
            return fallback
    return primary


def esm_path(root: Path, task: str, split: str) -> Path:
    return root / task / f"esm2_{task}_{split}.parquet"


def load_split_arrays(
    task: str,
    split: str,
    ccd_root: Path,
    ccd_union_subdir: str,
    ccd_features: list[str],
    esm_root: Path,
) -> SplitArrays:
    ccd_file = ccd_path(ccd_root, task, ccd_union_subdir, split)
    esm_file = esm_path(esm_root, task, split)
    if not ccd_file.exists():
        raise FileNotFoundError(ccd_file)
    if not esm_file.exists():
        raise FileNotFoundError(esm_file)
    ccd = pd.read_parquet(ccd_file)
    esm = pd.read_parquet(esm_file)
    esm_cols = [column for column in esm.columns if column.startswith("esm2_")]
    missing = sorted(set(ccd_features).difference(ccd.columns))
    if missing:
        raise RuntimeError(f"{ccd_file} missing CCD features: {missing[:10]}")

    merged = ccd[["fasta_id", "label_id", *ccd_features]].merge(
        esm[["fasta_id", *esm_cols]],
        on="fasta_id",
        how="left",
        sort=False,
    )
    if merged[esm_cols].isnull().any().any():
        raise RuntimeError(f"Missing ESM features after merge for {task}/{split}")
    return SplitArrays(
        ccd=merged.loc[:, ccd_features].to_numpy(dtype=np.float32),
        esm=merged.loc[:, esm_cols].to_numpy(dtype=np.float32),
        y=merged["label_id"].to_numpy(dtype=np.float32),
        fasta_id=merged["fasta_id"].astype(str).tolist(),
    )


def make_loader(arrays: SplitArrays, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays.ccd),
        torch.from_numpy(arrays.esm),
        torch.from_numpy(arrays.y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def build_model(model_name: str, ccd_dim: int, esm_dim: int, group_dims: list[int], args: argparse.Namespace) -> nn.Module:
    if model_name == "ccd_mlp":
        return SingleModalityMLP(ccd_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, modality="ccd")
    if model_name == "esm_mlp":
        return SingleModalityMLP(esm_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, modality="esm")
    if model_name == "concat_mlp":
        return ConcatMLP(ccd_dim, esm_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    if model_name == "gated":
        return GatedTwoTower(ccd_dim, esm_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    if model_name == "cross_attention":
        return CrossAttentionFusion(
            ccd_group_dims=group_dims,
            esm_dim=esm_dim,
            esm_tokens=args.esm_tokens,
            d_model=args.d_model,
            heads=args.heads,
            dropout=args.dropout,
        )
    if model_name == "cross_attention_residual":
        return ResidualCrossAttentionFusion(
            ccd_dim=ccd_dim,
            esm_dim=esm_dim,
            ccd_group_dims=group_dims,
            esm_tokens=args.esm_tokens,
            d_model=args.d_model,
            heads=args.heads,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    n_batches = 0
    with torch.no_grad():
        for ccd, esm, y in loader:
            ccd = ccd.to(device, non_blocking=True)
            esm = esm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(ccd, esm)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            n_batches += 1
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    return np.concatenate(targets), np.concatenate(probs), total_loss / max(n_batches, 1)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.02, 0.98, 97)))
    if thresholds.size == 0:
        return 0.5
    scores = [balanced_accuracy_score(y_true, (y_prob >= threshold).astype(np.int64)) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def train_one(
    task: str,
    model_name: str,
    seed: int,
    splits: dict[str, SplitArrays],
    group_dims: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
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
    history: list[dict[str, object]] = []
    max_epochs = 3 if args.smoke else args.epochs
    patience = 2 if args.smoke else args.patience

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
    eval_loaders = {
        split: make_loader(splits[split], batch_size, shuffle=False, num_workers=args.num_workers)
        for split in ("val", "test", hard_split(task))
    }
    y_val, p_val, _ = predict(model, eval_loaders["val"], device)
    threshold = best_threshold(y_val, p_val)
    rows: list[dict[str, object]] = []
    for split, loader in eval_loaders.items():
        y_true, y_prob, loss = predict(model, loader, device)
        metrics = calculate_metrics(y_true, y_prob, threshold)
        row = {
            "task": task,
            "model": model_name,
            "seed": seed,
            "split": split,
            "n": int(y_true.shape[0]),
            "n_ccd_features": int(ccd_dim),
            "n_esm_features": int(esm_dim),
            "threshold": threshold,
            "loss": loss,
            **metrics,
        }
        rows.append(row)
        print(
            f"[EVAL] {task}/{model_name}/seed{seed}/{split} "
            f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"BA={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f}"
        )
    state_summary = {
        "best_val_auc": best_val_auc,
        "n_epochs_ran": len(history),
        "threshold": threshold,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    model_state = {key: value.detach().cpu() for key, value in best_state.items()}
    return rows, {"history": history, "summary": state_summary, "model_state": model_state}


def fit_standardizers(splits: dict[str, SplitArrays]) -> tuple[Standardizer, Standardizer]:
    return Standardizer.fit(splits["train"].ccd), Standardizer.fit(splits["train"].esm)


def apply_standardizers(
    splits: dict[str, SplitArrays],
    ccd_scaler: Standardizer,
    esm_scaler: Standardizer,
) -> dict[str, SplitArrays]:
    transformed = {}
    for split, arrays in splits.items():
        transformed[split] = SplitArrays(
            ccd=ccd_scaler.transform(arrays.ccd),
            esm=esm_scaler.transform(arrays.esm),
            y=arrays.y.astype(np.float32),
            fasta_id=arrays.fasta_id,
        )
    return transformed


def standardize_splits(splits: dict[str, SplitArrays]) -> dict[str, SplitArrays]:
    ccd_scaler = Standardizer.fit(splits["train"].ccd)
    esm_scaler = Standardizer.fit(splits["train"].esm)
    return apply_standardizers(splits, ccd_scaler, esm_scaler)


def aggregate_results(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "balanced_accuracy",
        "f1",
        "mcc",
    ]
    grouped = df.groupby(["task", "model", "split"], as_index=False)
    agg_parts = []
    for keys, sub in grouped:
        task, model, split = keys
        row: dict[str, object] = {"task": task, "model": model, "split": split, "n_seeds": int(sub["seed"].nunique())}
        for metric in metric_cols:
            row[f"mean_{metric}"] = float(sub[metric].mean())
            row[f"std_{metric}"] = float(sub[metric].std(ddof=0))
        agg_parts.append(row)
    return pd.DataFrame(agg_parts)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output_root = Path(args.output_root).resolve()
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ccd_features_raw = json.loads(Path(args.ccd_feature_json).read_text(encoding="utf-8"))
    ccd_features, ccd_groups, group_dims = grouped_feature_order(ccd_features_raw)
    manifest = {
        "device": str(device),
        "tasks": args.tasks,
        "models": args.models,
        "seeds": args.seeds,
        "ccd_root": str(Path(args.ccd_root).resolve()),
        "ccd_union_subdir": args.ccd_union_subdir,
        "ccd_feature_json": str(Path(args.ccd_feature_json).resolve()),
        "esm_root": str(Path(args.esm_root).resolve()),
        "ccd_groups": {group: len(features) for group, features in ccd_groups.items()},
        "args": vars(args),
    }

    all_rows: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []

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
            for split in ("train", "val", "test", hard_split(task))
        }
        ccd_scaler, esm_scaler = fit_standardizers(raw_splits)
        splits = apply_standardizers(raw_splits, ccd_scaler, esm_scaler)
        for model_name in args.models:
            for seed in args.seeds:
                rows, payload = train_one(task, model_name, seed, splits, group_dims, args, device)
                all_rows.extend(rows)
                all_history.extend(payload["history"])
                run_summary = {"task": task, "model": model_name, "seed": seed, **payload["summary"]}
                checkpoint_path = checkpoint_dir / f"{task}__{model_name}__seed{seed}.pt"
                checkpoint_payload = {
                    "task": task,
                    "model_name": model_name,
                    "seed": seed,
                    "model_state_dict": payload["model_state"],
                    "threshold": payload["summary"]["threshold"],
                    "best_val_auc": payload["summary"]["best_val_auc"],
                    "n_epochs_ran": payload["summary"]["n_epochs_ran"],
                    "ccd_scaler": ccd_scaler.state_dict(),
                    "esm_scaler": esm_scaler.state_dict(),
                    "ccd_features": ccd_features,
                    "esm_features": [f"esm2_{index:04d}" for index in range(raw_splits["train"].esm.shape[1])],
                    "ccd_groups": ccd_groups,
                    "group_dims": group_dims,
                    "model_hyperparameters": {
                        "hidden_dim": args.hidden_dim,
                        "d_model": args.d_model,
                        "heads": args.heads,
                        "esm_tokens": args.esm_tokens,
                        "dropout": args.dropout,
                    },
                    "feature_sources": {
                        "ccd_root": str(Path(args.ccd_root).resolve()),
                        "ccd_union_subdir": args.ccd_union_subdir,
                        "ccd_feature_json": str(Path(args.ccd_feature_json).resolve()),
                        "esm_root": str(Path(args.esm_root).resolve()),
                    },
                }
                torch.save(checkpoint_payload, checkpoint_path)
                run_summary["checkpoint"] = str(checkpoint_path)
                run_summaries.append(run_summary)

    result_df = pd.DataFrame(all_rows)
    summary_df = aggregate_results(all_rows)
    history_df = pd.DataFrame(all_history)
    run_summary_df = pd.DataFrame(run_summaries)

    result_path = report_dir / "fusion_model_results.csv"
    summary_path = report_dir / "fusion_model_summary.csv"
    history_path = report_dir / "fusion_model_training_history.csv"
    run_summary_path = report_dir / "fusion_model_run_summary.csv"
    manifest_path = report_dir / "fusion_model_manifest.json"
    result_df.to_csv(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    run_summary_df.to_csv(run_summary_path, index=False)
    manifest["result_csv"] = str(result_path)
    manifest["summary_csv"] = str(summary_path)
    manifest["history_csv"] = str(history_path)
    manifest["run_summary_csv"] = str(run_summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] detailed results -> {result_path}")
    print(f"[DONE] summary -> {summary_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
