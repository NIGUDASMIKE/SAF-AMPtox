import os
import sys
import pandas as pd
import numpy as np
import itertools
import Levenshtein

# 动态寻址
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def calc_uniqueness(sequences):
    """计算唯一性：去重后的数量 / 总数量"""
    if not sequences: return 0.0
    return len(set(sequences)) / len(sequences)


def calc_diversity(sequences, max_samples=500):
    """
    计算序列多样性：1 - Mean Pairwise Identity
    为防止 1000条序列两两比对(近 50万次)过慢，采用随机采样加速计算
    """
    seqs = list(set(sequences))  # 先去重
    if len(seqs) < 2: return 0.0

    # 如果序列太多，随机抽取 max_samples 条进行两两比对（统计学上足够精确）
    if len(seqs) > max_samples:
        import random
        random.seed(42)
        seqs = random.sample(seqs, max_samples)

    pairs = list(itertools.combinations(seqs, 2))
    identities = []

    for s1, s2 in pairs:
        # 序列一致性 = 1 - (编辑距离 / 最长序列长度)
        dist = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        identity = 1.0 - (dist / max_len)
        identities.append(identity)

    mean_pairwise_identity = np.mean(identities)
    return 1.0 - mean_pairwise_identity


def run_evaluation():
    print("==================================================")
    print("🔬 [AMP_DeNovo] 序列质量量化评估引擎启动")
    print("==================================================")

    # 定义要评估的文件路径
    groups = {
        "Group 1 (Wild EvoDiff)": os.path.join(PROJECT_ROOT, "results", "group1_zeroshot", "group1_metrics.csv"),
        "Group 2 (SFT Tuned)": os.path.join(PROJECT_ROOT, "results", "group2_sft", "group2_metrics.csv"),
        "Group 3 (DPO Epiphany)": os.path.join(PROJECT_ROOT, "results", "group3_dpo", "group3_metrics.csv")
    }

    results = []

    for name, path in groups.items():
        if not os.path.exists(path):
            print(f"[!] 找不到文件，跳过: {name}")
            continue

        df = pd.read_csv(path)
        seqs = df['Sequence'].dropna().astype(str).tolist()
        scores = df['Oracle_Score'].tolist()

        avg_score = np.mean(scores)
        uniqueness = calc_uniqueness(seqs)
        diversity = calc_diversity(seqs)

        results.append({
            "Model": name,
            "Avg AMP": f"{avg_score:.4f}",
            "Uniqueness": f"{uniqueness:.4f}",
            "Diversity": f"{diversity:.4f}"
        })

    # 打印顶刊风格的数据表格
    df_results = pd.DataFrame(results)
    print("\n📊 核心指标大满贯 (可以直接写入论文 Table 中)：")
    print(df_results.to_markdown(index=False))
    print("\n==================================================")


if __name__ == "__main__":
    run_evaluation()