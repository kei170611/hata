import numpy as np

# =================================================================
# 1. 基本設定と距離マトリクス
# =================================================================
num_trucks = 10        # 所有する積載車の台数
truck_cap = 5          # 1台あたりの最大積載容量（サイズ合計）
num_jobs = 100         # 1日の総配送依頼数
locations = ["中央区", "北区", "南区", "西区", "東区"]

# 拠点間の距離データ (km)
dist_raw = {
    ("中央区", "北区"): 5, ("中央区", "南区"): 8, ("中央区", "西区"): 7, ("中央区", "東区"): 6,
    ("北区", "南区"): 7, ("北区", "西区"): 6, ("北区", "東区"): 5,
    ("南区", "西区"): 4, ("南区", "東区"): 6,
    ("西区", "東区"): 5,
}

# 距離マトリクスの完全化（双方向・自己参照対応）
dist = {}
for i in locations:
    for j in locations:
        if i == j:
            dist[(i,j)] = 0
        else:
            key = tuple(sorted((i, j)))
            dist[(i,j)] = dist_raw.get(key, 0)

# =================================================================
# 2. ジョブ生成 (ランダム・シミュレーション)
# =================================================================
# np.random.seed(42)  # 特定のパターンでテストしたい場合はコメントを外す
jobs = []
for _ in range(num_jobs):
    p = np.random.choice(locations)
    d = np.random.choice([l for l in locations if l != p])
    s = np.random.randint(1, 4)  # 車両サイズ 1:軽/普通, 2:大型, 3:特大
    jobs.append({"pickup": p, "drop": d, "size": s})

# =================================================================
# 3. 最適化エンジン (エネルギー計算ロジック)
# =================================================================
def compute_energy(assignment):
    """
    配車計画の『ダメさ加減』を数値化する。
    距離が長いほど、また過積載が発生するほど数値（スコア）が高くなる。
    """
    total_score = 0
    penalty = 0
    
    for t in range(num_trucks):
        # 積載車tに割り振られたジョブIDを取得
        my_indices = [i for i, truck_id in enumerate(assignment) if truck_id == t]
        if not my_indices: continue
        
        # 簡易ルート最適化：積み込み地点のエリア順に並べる
        my_indices = sorted(my_indices, key=lambda i: locations.index(jobs[i]["pickup"]))
        
        current_loc = "中央区"  # 全車、中央区からスタート
        current_load = 0
        
        for idx in my_indices:
            job = jobs[idx]
            
            # A. 積み込み地点への移動距離を加算
            total_score += dist[(current_loc, job["pickup"])]
            current_loc = job["pickup"]
            
            # B. 積み込み実行と過積載チェック
            current_load += job["size"]
            if current_load > truck_cap:
                # 過積載には非常に重いペナルティ(1000)を課す
                penalty += 1000 * (current_load - truck_cap)
            
            # C. 荷降ろし地点への移動距離を加算
            total_score += dist[(current_loc, job["drop"])]
            current_loc = job["drop"]
            
            # D. 荷降ろし完了（ここで荷台の空き枠が復活！）
            current_load -= job["size"]
            
        # E. 全ての仕事を終えて拠点（中央区）に戻る距離
        total_score += dist[(current_loc, "中央区")]
        
    return total_score + penalty

# =================================================================
# 4. アニーリング探索 (試行錯誤アルゴリズム)
# =================================================================
def anneal_search(iterations=10000):
    # 初期状態：100個の仕事を10台にランダムに割り振る
    current_assign = np.random.randint(0, num_trucks, num_jobs)
    current_E = compute_energy(current_assign)
    
    best_assign = current_assign.copy()
    best_E = current_E
    
    T = 100.0  # 初期温度（探索の勢い）
    
    for i in range(iterations):
        # 仕事を1つ選び、別のトラックへ積み替えてみる（近傍探索）
        target_job = np.random.randint(num_jobs)
        old_truck = current_assign[target_job]
        new_truck = np.random.randint(num_trucks)
        
        if old_truck == new_truck: continue
        
        current_assign[target_job] = new_truck
        new_E = compute_energy(current_assign)
        
        # 判定：改善すれば採用、悪化しても確率（温度に依存）で採用
        if new_E < current_E or np.random.rand() < np.exp(-(new_E - current_E) / T):
            current_E = new_E
            if current_E < best_E:
                best_E = current_E
                best_assign = current_assign.copy()
        else:
            # 却下して元のトラックに戻す
            current_assign[target_job] = old_truck
            
        # 温度を徐々に下げる（最後は良い解に落ち着かせる）
        T *= 0.9995
        
    return best_assign, best_E

# =================================================================
# 5. 実行と詳細レポート出力
# =================================================================
best_assign, best_E = anneal_search(iterations=10000)

print("\n" + "★" * 30)
print("   巡回配送計画 最適化レポート")
print("★" * 30)
print(f"🎯 最終評価スコア: {best_E:.1f} (低いほど高効率)")

# --- 運行指示書の出力 ---
for t in range(num_trucks):
    t_jobs = [i for i, truck_id in enumerate(best_assign) if truck_id == t]
    
    print(f"\n" + "="*70)
    print(f"【積載車 {t}番】 運行指示書 (担当ジョブ数: {len(t_jobs)}件)")
    print(f"{'移動':<4} | {'Job ID':<7} | {'積地':<6} → {'降地':<6} | {'サイズ':<4} | {'状態'}")
    print("-" * 70)
    
    if not t_jobs:
        print("   ※ 本日の稼働予定はありません。")
        continue
        
    t_jobs_sorted = sorted(t_jobs, key=lambda i: locations.index(jobs[i]["pickup"]))
    
    temp_load = 0
    for step, idx in enumerate(t_jobs_sorted):
        j = jobs[idx]
        temp_load += j["size"]
        status = "OK" if temp_load <= truck_cap else "!!過積載!!"
        print(f"{step+1:<4} | ID:{idx:<5} | {j['pickup']:<8} → {j['drop']:<8} | {j['size']:<5} | {status}")
        temp_load -= j["size"] # 降ろした後の処理
    
    total_size = sum(jobs[idx]["size"] for idx in t_jobs)
    print("-" * 70)
    print(f" >>> 延べ積載量: {total_size}台分 / 稼働効率平均: {total_size/len(t_jobs):.1f}")

# --- 元データの確認用リスト ---
print("\n" + "📋 【参考】本日の配送依頼（元データ）全100件")
print("-" * 45)
print(f"{'ID':<6} | {'積む場所':<6} → {'降ろす場所':<6} | {'サイズ'}")
for i, j in enumerate(jobs):
    if i % 20 == 0 and i != 0: print("-" * 45) # 20件ごとに区切り
    print(f"ID:{i:<3} | {j['pickup']:<8} → {j['drop']:<8} | {j['size']}")
print("-" * 45)
