import numpy as np

# --- 設定 ---
num_trucks = 10
truck_cap = 5
num_jobs = 100
locations = ["中央区", "北区", "南区", "西区", "東区"]
dist_raw = {
    ("中央区", "北区"): 5, ("中央区", "南区"): 8, ("中央区", "西区"): 7, ("中央区", "東区"): 6,
    ("北区", "南区"): 7, ("北区", "西区"): 6, ("北区", "東区"): 5,
    ("南区", "西区"): 4, ("南区", "東区"): 6, ("西区", "東区"): 5,
}
dist = {}
for i in locations:
    for j in locations:
        if i == j: dist[(i,j)] = 0
        else:
            key = tuple(sorted((i, j))); dist[(i,j)] = dist_raw.get(key, 0)

# ジョブ生成
#np.random.seed(42) ランダム生成
jobs = []
for i in range(num_jobs):
    p = np.random.choice(locations)
    d = np.random.choice([l for l in locations if l != p])
    s = np.random.randint(1, 4)
    jobs.append({"id": i, "pickup": p, "drop": d, "size": s})

# 混載ルート構築エンジン
def get_mixed_load_route(my_job_indices):
    if not my_job_indices: return [], 0
    unvisited_pickups = my_job_indices.copy()
    on_board = []
    current_loc = "中央区"
    current_load = 0
    total_dist = 0
    history = []
    
    while unvisited_pickups or on_board:
        best_target = None
        min_d = float('inf')
        target_type = ""
        
        # 候補1：積み（空き容量がある場合）
        for idx in unvisited_pickups:
            if current_load + jobs[idx]["size"] <= truck_cap:
                d = dist[(current_loc, jobs[idx]["pickup"])]
                if d < min_d: min_d, best_target, target_type = d, idx, "pickup"
        
        # 候補2：降ろし
        for idx in on_board:
            d = dist[(current_loc, jobs[idx]["drop"])]
            if d < min_d: min_d, best_target, target_type = d, idx, "drop"
        
        if best_target is None: break
        
        total_dist += min_d
        job = jobs[best_target]
        
        if target_type == "pickup":
            current_loc = job["pickup"]
            current_load += job["size"]
            unvisited_pickups.remove(best_target)
            on_board.append(best_target)
            history.append({"type": "積", "loc": current_loc, "id": job["id"], "size": job["size"], "load": current_load, "dist": min_d})
        else:
            current_loc = job["drop"]
            current_load -= job["size"]
            on_board.remove(best_target)
            history.append({"type": "降", "loc": current_loc, "id": job["id"], "size": job["size"], "load": current_load, "dist": min_d})
            
    total_dist += dist[(current_loc, "中央区")]
    return history, total_dist

def compute_energy(assignment):
    score = 0
    for t in range(num_trucks):
        indices = [i for i, tid in enumerate(assignment) if tid == t]
        _, d = get_mixed_load_route(indices)
        score += d
    return score

# アニーリング探索
def solve():
    curr_assign = np.random.randint(0, num_trucks, num_jobs)
    curr_E = compute_energy(curr_assign)
    best_assign, best_E = curr_assign.copy(), curr_E
    T = 100.0
    for _ in range(15000):
        idx = np.random.randint(num_jobs)
        old, new = curr_assign[idx], np.random.randint(num_trucks)
        if old == new: continue
        curr_assign[idx] = new
        new_E = compute_energy(curr_assign)
        if new_E < curr_E or np.random.rand() < np.exp(-(new_E - curr_E) / T):
            curr_E = new_E
            if curr_E < best_E: best_E, best_assign = curr_E, curr_assign.copy()
        else: curr_assign[idx] = old
        T *= 0.9995
    return best_assign, best_E

best_assign, best_E = solve()

# --- 出力レポート ---
print("\n" + "★" * 60)
print("   巡回配送計画 最適化レポート (合わせ積み・完全混載版)")
print("★" * 60)
print(f"🎯 最終評価スコア: {best_E:.1f}\n")

for t in range(num_trucks):
    t_indices = [i for i, tid in enumerate(best_assign) if tid == t]
    history, _ = get_mixed_load_route(t_indices)
    
    print(f"\n{'='*65}")
    print(f"【積載車 {t}番】 運行指示書 (担当: {len(t_indices)}件)")
    print(f"{'行動':<4} | {'地点':<6} | {'Job':<6} | {'サイズ':<4} | {'積載量':<5} | {'移動'}")
    print("-" * 65)
    
    if not history:
        print(" ※ 稼働なし")
        continue

    last_loc = "中央区"
    for h in history:
        act = f"[{h['type']}]"
        print(f"{act:<4} | {h['loc']:<8} | ID:{h['id']:<3} | {h['size']:^6} | {h['load']:^6} | {h['dist']}km")
        last_loc = h['loc']
    print("-" * 65)
    print(f" >>> 最終帰還: {dist[(last_loc, '中央区')]}km")

# --- 全ジョブ可視化 ---
print("\n" + "📋 【全100件】本日の配送依頼データ一覧")
print("-" * 45)
print(f"{'ID':<6} | {'積み地点':<6} → {'降ろし地点':<6} | {'サイズ'}")
print("-" * 45)
for i, j in enumerate(jobs):
    if i % 20 == 0 and i != 0: print("-" * 45)
    print(f"ID:{i:<3} | {j['pickup']:<8} → {j['drop']:<8} | {j['size']:^6}")
print("-" * 45)