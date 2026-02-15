import numpy as np

# --- 設定（変更なし） ---
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

# ジョブ生成（毎回ランダム）
jobs = []
for i in range(num_jobs):
    p = np.random.choice(locations)
    d = np.random.choice([l for l in locations if l != p])
    s = np.random.randint(1, 4)
    jobs.append({"id": i, "pickup": p, "drop": d, "size": s})

# 混載ルート構築エンジン（前回と同じ）
def get_mixed_load_route(my_job_indices):
    if not my_job_indices: return [], 0
    unvisited_pickups = my_job_indices.copy()
    on_board = []
    current_loc = "中央区"
    current_load = 0
    total_dist = 0
    history = []
    while unvisited_pickups or on_board:
        best_target, min_d, target_type = None, float('inf'), ""
        for idx in unvisited_pickups:
            if current_load + jobs[idx]["size"] <= truck_cap:
                d = dist[(current_loc, jobs[idx]["pickup"])]
                if d < min_d: min_d, best_target, target_type = d, idx, "pickup"
        for idx in on_board:
            d = dist[(current_loc, jobs[idx]["drop"])]
            if d < min_d: min_d, best_target, target_type = d, idx, "drop"
        if best_target is None: break
        total_dist += min_d
        job = jobs[best_target]
        if target_type == "pickup":
            current_loc = job["pickup"]; current_load += job["size"]; unvisited_pickups.remove(best_target); on_board.append(best_target)
            history.append({"type": "積", "loc": current_loc, "id": job["id"], "size": job["size"], "load": current_load, "dist": min_d})
        else:
            current_loc = job["drop"]; current_load -= job["size"]; on_board.remove(best_target)
            history.append({"type": "降", "loc": current_loc, "id": job["id"], "size": job["size"], "load": current_load, "dist": min_d})
    total_dist += dist[(current_loc, "中央区")]
    return history, total_dist

# =================================================================
# 改良：評価関数（距離 ＋ 件数のばらつきペナルティ）
# =================================================================
def compute_energy(assignment):
    total_distance = 0
    counts = []
    
    for t in range(num_trucks):
        indices = [i for i, tid in enumerate(assignment) if tid == t]
        _, d = get_mixed_load_route(indices)
        total_distance += d
        counts.append(len(indices))
    
    # 件数の標準偏差（ばらつき）を計算
    # ペナルティ係数 5.0 は「1件の格差を5km分と同等にみなす」という設定
    counts_penalty = np.std(counts) * 5.0
    
    return total_distance + counts_penalty

# --- アニーリング探索 ---
def solve():
    curr_assign = np.random.randint(0, num_trucks, num_jobs)
    curr_E = compute_energy(curr_assign)
    best_assign, best_E = curr_assign.copy(), curr_E
    T = 100.0
    for _ in range(15000):
        idx = np.random.randint(num_jobs); old, new = curr_assign[idx], np.random.randint(num_trucks)
        if old == new: continue
        curr_assign[idx] = new; new_E = compute_energy(curr_assign)
        if new_E < curr_E or np.random.rand() < np.exp(-(new_E - curr_E) / T):
            curr_E = new_E
            if curr_E < best_E: best_E, best_assign = curr_E, curr_assign.copy()
        else: curr_assign[idx] = old
        T *= 0.9995
    return best_assign, best_E

best_assign, best_E = solve()

# --- レポート表示（前回同様の指示書 ＋ 統計情報） ---
print(f"\n🎯 総合評価スコア: {best_E:.1f} (距離 + 平準化ペナルティ)")
total_actual_dist = 0
final_counts = []

for t in range(num_trucks):
    t_indices = [i for i, tid in enumerate(best_assign) if tid == t]
    history, d = get_mixed_load_route(t_indices)
    total_actual_dist += d
    final_counts.append(len(t_indices))
    print(f"車両 {t}番: {len(t_indices)}件 / 走行 {d}km")

print(f"\n総実走行距離: {total_actual_dist}km")
print(f"件数格差: 最小{min(final_counts)}件 〜 最大{max(final_counts)}件")

# (これまでの最適化エンジン・平準化ロジックを用いて最終レポートを生成します)

print("\n" + "★" * 70)
print("   巡回配送計画 最適化レポート (実務平準化・完全混載版)")
print("★" * 70)
print(f"🎯 総合評価スコア: {best_E:.1f} (距離 + 平準化ペナルティ)")
print(f"件数格差: 最小 {min(final_counts)}件 〜 最大 {max(final_counts)}件")
print(f"総実走行距離: {total_actual_dist}km")

# --- 各車両の運行指示書 ---
for t in range(num_trucks):
    t_indices = [i for i, tid in enumerate(best_assign) if tid == t]
    history, d = get_mixed_load_route(t_indices)
    
    print(f"\n{'='*75}")
    print(f"【積載車 {t}番】 運行指示書 (担当: {len(t_indices)}件 / 走行: {d}km)")
    print(f"{'行動':<4} | {'地点':<10} | {'Job':<6} | {'サイズ':<4} | {'積載量':<5} | {'移動'}")
    print("-" * 75)
    
    if not history:
        print(" ※ 稼働なし")
        continue

    last_loc = "中央区"
    for h in history:
        act = f"[{h['type']}]"
        print(f"{act:<4} | {h['loc']:<12} | ID:{h['id']:<3} | {h['size']:^6} | {h['load']:^6} | {h['dist']}km")
        last_loc = h['loc']
    
    final_return = dist[(last_loc, '中央区')]
    print("-" * 75)
    print(f" >>> 最終帰還: {final_return}km (拠点:中央区へ)")

# --- 全100件の依頼データ一覧 ---
print("\n" + "📋 【全100件】本日の配送依頼データ一覧")
print("-" * 55)
print(f"{'ID':<6} | {'積み地点':<10} → {'降ろし地点':<10} | {'サイズ'}")
print("-" * 55)
for i, j in enumerate(jobs):
    if i % 20 == 0 and i != 0: print("-" * 55)
    print(f"ID:{i:<3} | {j['pickup']:<12} → {j['drop']:<12} | {j['size']:^6}")
print("-" * 55)



def generate_html_report(best_assign, jobs, num_trucks, total_actual_dist, final_counts):
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: "Helvetica Neue", Arial, "Hiragino Kaku Gothic ProN", "Hiragino Sans", sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: auto; padding: 20px; }}
            h1 {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #2c3e50; border-left: 10px solid #2c3e50; padding-left: 15px; margin-top: 50px; background: #f4f7f6; }}
            .summary {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; display: flex; justify-content: space-around; }}
            .summary-item {{ text-align: center; }}
            .summary-item span {{ display: block; font-size: 1.2rem; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; table-layout: fixed; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; word-wrap: break-word; }}
            th {{ background-color: #34495e; color: white; font-size: 0.9rem; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .type-pick {{ color: #d35400; font-weight: bold; background: #fff5eb; }}
            .type-drop {{ color: #27ae60; font-weight: bold; background: #f0fff4; }}
            .footer {{ text-align: right; font-size: 0.8rem; color: #7f8c8d; margin-top: 50px; border-top: 1px solid #eee; padding-top: 10px; }}
            @media print {{
                h2 {{ page-break-before: always; }}
                .summary {{ background: #eee !important; color: black !important; border: 1px solid #333; }}
            }}
        </style>
    </head>
    <body>
        <h1>🚛 巡回配送計画 運行指示レポート</h1>
        <div class="summary">
            <div class="summary-item">総走行距離<span>{total_actual_dist}km</span></div>
            <div class="summary-item">車両台数<span>{num_trucks}台</span></div>
            <div class="summary-item">件数格差<span>{min(final_counts)} 〜 {max(final_counts)}件</span></div>
        </div>
    """

    for t in range(num_trucks):
        t_indices = [i for i, tid in enumerate(best_assign) if tid == t]
        history, d = get_mixed_load_route(t_indices)
        
        html += f"""
        <div class="truck-section">
            <h2>車両 {t}番 指示書（担当: {len(t_indices)}件 / 走行距離: {d}km）</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 15%;">行動</th>
                        <th style="width: 25%;">地点</th>
                        <th style="width: 15%;">Job ID</th>
                        <th style="width: 15%;">サイズ</th>
                        <th style="width: 15%;">積載量</th>
                        <th style="width: 15%;">区間距離</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for h in history:
            type_label = h['type']
            type_class = "type-pick" if type_label == "積" else "type-drop"
            html += f"""
                <tr>
                    <td class="{type_class}">[{type_label}]</td>
                    <td>{h['loc']}</td>
                    <td>ID:{h['id']}</td>
                    <td>{h['size']}</td>
                    <td>{h['load']}/5</td>
                    <td>{h['dist']}km</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """

    html += """
        <div class="footer">
            生成日時: 2026年2月15日 | 配送最適化システム Gemini Logistics Engine
        </div>
    </body>
    </html>
    """
    
    with open("logistics_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ 'logistics_report.html' を作成しました。")

# 実行（引数に計算結果を渡してください）
generate_html_report(best_assign, jobs, num_trucks, total_actual_dist, final_counts)