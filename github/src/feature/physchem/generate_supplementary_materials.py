from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from config import FEATURE_OUTPUT_ROOT, LGBM_PARAMS
from common import report_path
from feature_registry import FEATURE_SPECS


SUPP_ROOT = FEATURE_OUTPUT_ROOT / "supplementary"
SUPP_TABLE_DIR = SUPP_ROOT / "tables"
SUPP_FIGURE_DIR = SUPP_ROOT / "figures"
SUPP_SOURCE_DIR = SUPP_ROOT / "source_data"


def ensure_dirs() -> None:
    SUPP_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def build_table_s1() -> Path:
    quick_screen = pd.read_csv(report_path("quick_screen_summary.csv"))
    dim_map = (
        quick_screen.loc[:, ["feature_group", "n_features"]]
        .drop_duplicates(subset=["feature_group"])
        .set_index("feature_group")["n_features"]
        .to_dict()
    )
    dim_map["TPC"] = 8000

    rows: list[dict[str, object]] = []
    for spec in FEATURE_SPECS:
        rows.append({
            "feature_group": spec.name,
            "extractor_key": spec.extractor_key,
            "backend": spec.backend,
            "family": spec.family,
            "description": spec.description,
            "status": spec.status,
            "included_in_screening": bool(spec.enabled and spec.status == "ready"),
            "n_dimensions": dim_map.get(spec.name),
        })
    out_path = SUPP_TABLE_DIR / "table_s1_feature_groups.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def build_table_s2() -> Path:
    rows = [{"parameter": key, "value": value} for key, value in LGBM_PARAMS.items()]
    rows.append({"parameter": "random_state (quick screen)", "value": 13})
    rows.append({"parameter": "random_states (refined stage)", "value": "7, 13, 29, 47, 101"})
    out_path = SUPP_TABLE_DIR / "table_s2_lightgbm_params.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def build_table_s3() -> Path:
    quick_screen = pd.read_csv(report_path("quick_screen_summary.csv"))
    pivot = quick_screen.pivot_table(
        index="feature_group",
        columns="task",
        values=["mean_roc_auc", "mean_pr_auc", "n_features"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{task}" for metric, task in pivot.columns]
    pivot = pivot.reset_index()
    pivot["screening_seed"] = 13
    pivot = pivot.rename(columns={
        "n_features_amp": "n_dimensions",
        "mean_roc_auc_amp": "amp_val_roc_auc",
        "mean_pr_auc_amp": "amp_val_pr_auc",
        "mean_roc_auc_tox": "tox_val_roc_auc",
        "mean_pr_auc_tox": "tox_val_pr_auc",
    })
    keep_columns = [
        "feature_group",
        "n_dimensions",
        "screening_seed",
        "amp_val_roc_auc",
        "amp_val_pr_auc",
        "tox_val_roc_auc",
        "tox_val_pr_auc",
    ]
    out_path = SUPP_TABLE_DIR / "table_s3_single_feature_screening.csv"
    pivot.loc[:, keep_columns].sort_values("feature_group").to_csv(out_path, index=False)
    return out_path


def copy_figure_s1() -> list[Path]:
    copied: list[Path] = []
    src_base = FEATURE_OUTPUT_ROOT / "figures" / "figureB_topk_elbow"
    dst_base = SUPP_FIGURE_DIR / "figureS1_topk_elbow"
    for suffix in (".svg", ".pdf", ".png"):
        src = Path(f"{src_base}{suffix}")
        dst = Path(f"{dst_base}{suffix}")
        shutil.copy2(src, dst)
        copied.append(dst)

    src_source = FEATURE_OUTPUT_ROOT / "source_data" / "figureB_topk_elbow_source_data.csv"
    dst_source = SUPP_SOURCE_DIR / "figureS1_topk_elbow_source_data.csv"
    shutil.copy2(src_source, dst_source)
    copied.append(dst_source)
    return copied


def write_main_methods() -> Path:
    text = """## 3.2 理化特征提取与多阶段自适应筛选

本研究构建了一个三阶段的理化描述符（CCD）提取与精修流程，以获得同时适用于抗菌肽（AMP）与毒性（TOX）判别任务的紧凑共享特征集。首先，基于 iFeatureOmegaCLI 与自定义理化特征组 PHYS8 组建候选描述符库。该候选库共包含 47 个特征群；其中，TPC 因 8,000 维三肽组成带来的计算与存储开销过高而被预先剔除，EGAAC 因对变长短肽输入稳定性不足而被排除，最终保留 45 个特征群进入正式筛选。PHYS8 由 8 个显式理化量构成，包括归一化序列长度、pH 7.4 条件下的净电荷、归一化疏水性、归一化 Boman 指数、缩放后的脂肪族指数、归一化等电点、芳香族残基比例以及归一化疏水矩。其余标准描述符由 iFeatureOmegaCLI 提取，完整列表见补充材料 Table S1。

在第一阶段中，每个特征群均被独立用于 AMP 和 TOX 任务建模，并在固定训练集/验证集划分上采用单随机种子 LightGBM 基线分类器进行评估（random_state=13；超参数见补充材料 Table S2）。针对 45 个特征群分别计算验证集 ROC-AUC 和 PR-AUC，据此构建单特征群性能景观。随后，依据初筛结果选取 17 个表现较优的候选特征群进入精筛轮。第二阶段在相同的训练集/验证集划分上使用 5 个随机种子（7、13、29、47、101）重复训练 LightGBM，并统计各特征群在验证集上的平均 ROC-AUC 与平均 PR-AUC。与固定 Top-K 群数策略不同，本研究采用基于经验 75 分位数的自适应阈值机制：AMP 任务阈值设定为平均 ROC-AUC ≥ 0.979354 且平均 PR-AUC ≥ 0.980618；TOX 任务阈值设定为平均 ROC-AUC ≥ 0.990998 且平均 PR-AUC ≥ 0.991052。基于该准则，AMP 任务筛得 ASDC、CKSAAP、DDE、AAC 和 DistancePair，TOX 任务筛得 ASDC、CTDD、KSCTriad 和 APAAC。对两任务入选群取并集后，得到由 AAC、APAAC、ASDC、CKSAAP、CTDD、DDE、DistancePair 和 KSCTriad 组成的联合特征矩阵，总维度为 4,033。

第三阶段在 4,033 维联合矩阵上进行细粒度特征精修。具体地，基于 LightGBM 的 feature importance gain 对全部具体特征维度排序，并在验证集上执行 Top-K 扫描（64、128、256、512、768、1024、1536、2048、3072 和 4033 维）。紧凑模型选择准则定义为：在验证集最优性能点附近，选择同时满足 ROC-AUC 和 PR-AUC 均不低于最优值 0.0005 容忍带内的最小 K。在该准则下，AMP 与 TOX 两个任务的最优紧凑子集均为 512 维。最终，对两任务特异性 512 维特征集取并集，得到包含 832 个具体理化特征的共享黄金特征集，其中 192 维为双任务重叠特征。该 832 维矩阵最终以 Parquet 格式保存，并作为后续模型的理化输入。
"""
    out_path = FEATURE_OUTPUT_ROOT / "reports" / "physchem_methods_main_text_20260607.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_supplement_outline(paths: dict[str, str]) -> Path:
    text = f"""# Physchem Supplementary Outline

## Table S1
- File: `{paths['table_s1']}`
- Content: complete candidate physicochemical descriptor groups, including backend, extractor key, descriptor family, inclusion status, and feature dimensionality where available.

## Table S2
- File: `{paths['table_s2']}`
- Content: LightGBM hyperparameter settings used for quick screening, refined repeated evaluation, and fine-grained feature selection.

## Table S3
- File: `{paths['table_s3']}`
- Content: single-feature-group validation performance across all screened groups, including AMP and TOX ROC-AUC/PR-AUC.

## Figure S1
- Files:
  - `{paths['fig_s1_svg']}`
  - `{paths['fig_s1_pdf']}`
  - `{paths['fig_s1_png']}`
- Source data:
  - `{paths['fig_s1_source']}`
- Content: Top-K elbow curve and tolerance-band rationale for selecting the compact 512-dimensional feature subsets.
"""
    out_path = FEATURE_OUTPUT_ROOT / "reports" / "physchem_supplement_outline_20260607.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> None:
    ensure_dirs()
    table_s1 = build_table_s1()
    table_s2 = build_table_s2()
    table_s3 = build_table_s3()
    copied = copy_figure_s1()
    main_methods = write_main_methods()

    paths = {
        "table_s1": str(table_s1),
        "table_s2": str(table_s2),
        "table_s3": str(table_s3),
        "fig_s1_svg": str(SUPP_FIGURE_DIR / "figureS1_topk_elbow.svg"),
        "fig_s1_pdf": str(SUPP_FIGURE_DIR / "figureS1_topk_elbow.pdf"),
        "fig_s1_png": str(SUPP_FIGURE_DIR / "figureS1_topk_elbow.png"),
        "fig_s1_source": str(SUPP_SOURCE_DIR / "figureS1_topk_elbow_source_data.csv"),
        "main_methods": str(main_methods),
    }
    supp_outline = write_supplement_outline(paths)

    manifest = {
        "table_s1": str(table_s1),
        "table_s2": str(table_s2),
        "table_s3": str(table_s3),
        "copied_files": [str(path) for path in copied],
        "main_methods": str(main_methods),
        "supplement_outline": str(supp_outline),
    }
    out_path = FEATURE_OUTPUT_ROOT / "reports" / "supplement_manifest_20260607.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] supplementary manifest -> {out_path}")


if __name__ == "__main__":
    main()
