import os
import csv
import difflib


# ==========================================
# AMP_DeNovo_Project - 查重与新颖性卫士 (Homology Tool)
# 目标: 扫描新生成的肽，将其与已知 AMP 数据库进行比对，揪出抄袭者！
# ==========================================

def load_known_amps(database_path):
    """加载真实世界中已知的 AMP 作为查重比对库"""
    known_amps = []
    if not os.path.exists(database_path):
        print(f"❌ 找不到查重数据库: {database_path}")
        return []

    with open(database_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            # 确保这行有数据，并且至少有两列
            if row and len(row) > 1:
                # row[0] 是 ID (如 AP02484)，row[1] 才是序列！
                seq = row[1].strip().upper()
                if seq.isalpha():  # 确保提取出来的是纯字母序列
                    known_amps.append(seq)
    return known_amps


def check_novelty(new_seq, known_amps_db, threshold=0.8):
    """
    检查新序列的新颖性。
    返回: is_novel (布尔值), max_sim (最高相似度), most_similar_seq (最像的已知序列)
    """
    max_sim = 0.0
    most_similar_seq = ""
    new_seq = new_seq.upper()

    # 逐一和数据库里的真实 AMP 进行比对 (计算编辑距离相似度)
    for known_seq in known_amps_db:
        # difflib 计算两个字符串的相似度比率 (0.0 ~ 1.0)
        sim = difflib.SequenceMatcher(None, new_seq, known_seq).ratio()
        if sim > max_sim:
            max_sim = sim
            most_similar_seq = known_seq

            # 提前熔断机制：如果相似度已经超过 95%，基本就是抄袭，直接终止比对节省算力
            if max_sim >= 0.95:
                break

    is_novel = max_sim < threshold
    return is_novel, max_sim, most_similar_seq


if __name__ == "__main__":
    # ---------------- 查重雷达测试 ----------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    # 把我们之前用于训练 Oracle 的真实 AMP 数据作为比对库
    db_path = os.path.join(base_dir, '..', 'data', 'veltri_positive.csv')

    print("[*] 正在加载自然界已知 AMP 查重库...")
    known_database = load_known_amps(db_path)
    print(f"[*] 成功装载 {len(known_database)} 条真实抗菌肽特征！\n")

    # 我们拿三个嫌疑人来测试
    suspects = [
        "RRAAVVNIVIRR",  # 嫌疑人A：我们之前空投的那个真实王者 (必然判定为抄袭)
        "MARRLLVWKWAGHAA",  # 嫌疑人B：刚才 EvoDiff 瞎生成的废沙 (大概率新颖，但没用)
        "KWKLFKKIGIGKFLHSAK"  # 嫌疑人C：著名的真实抗菌肽 Magainin 2 (必然判定为抄袭)
    ]

    print("================== 🔍 查重报告 🔍 ==================")
    for seq in suspects:
        is_novel, max_sim, closest_seq = check_novelty(seq, known_database, threshold=0.8)

        if is_novel:
            print(f"[✅ 通过] 序列: {seq:<20} | 身份: 自然界新物种！")
            print(f"           最接近已知序列: {closest_seq} (相似度: {max_sim:.2f})")
        else:
            print(f"[❌ 抄袭] 序列: {seq:<20} | 身份: 涉嫌洗稿已有 AMP！")
            print(f"           原版已知序列: {closest_seq} (相似度: {max_sim:.2f})")
        print("-" * 50)