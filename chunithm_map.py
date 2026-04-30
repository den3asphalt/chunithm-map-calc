import pulp
import datetime
import calendar
from collections import Counter

class ChunithmHybridOptimizer:
    def __init__(self):
        self.base_tp = 170
        self.base_map = 13
        
        self.tickets = {
            '1x': (0, 1, 3, 1.0),
            '2x': (100, 2, 3, 2.4),
            '3x': (200, 3, 4, 4.0),
            '4x': (300, 4, 5, 6.0),
            '5x': (400, 5, 6, 8.0),
            'free6': (0, 1, 6, 1.5),
            'free9': (0, 1, 9, 2.0)
        }

    def calc_play(self, settings, is_first_play, day_info):
        ticket, is_designated, songs = settings
        add_cost, map_mult, exp_base, tp_mult = self.tickets[ticket]
        
        cost = 100 + add_cost

        char_map_bonus = 2 if is_designated else 0
        daily_map_bonus = 2 if day_info['has_map_bonus'] else 0
        map_per_song = (self.base_map + char_map_bonus + daily_map_bonus) * map_mult
        earned_map = map_per_song * songs

        exp_multiplier = 1.5 if day_info['has_exp_bonus'] else 1.0
        char_exp_multiplier = 0.7 if is_designated else 1.0
        exp_per_song = exp_base * exp_multiplier * char_exp_multiplier
        earned_exp = exp_per_song * songs

        tp_boost_mult = 2.0 if day_info['has_tp_boost'] else 1.0
        first_play_mult = 2.0 if is_first_play else 1.0
        earned_tp = self.base_tp * tp_mult * tp_boost_mult * first_play_mult * songs

        return {
            'cost': cost,
            'map': earned_map,
            'exp': earned_exp,
            'tp': earned_tp,
            'songs': songs
        }

    def _get_top_n_patterns(self, patterns, day_info, status, n=5):
        """指定されたパターンのリストから、5つの特化軸で上位N件を抽出し重複を排除して返す"""
        def get_eval_score(p):
            ideal_plays = p['cost'] / 100.0 if p['cost'] > 0 else 0
            exp_mult_day = 1.5 if day_info['has_exp_bonus'] else 1.0
            ideal_exp = ideal_plays * (3 * exp_mult_day * status['base_songs'])
            exp_loss = max(0, ideal_exp - p['exp'])
            ideal_songs = ideal_plays * status['base_songs']
            song_loss = max(0, ideal_songs - p['songs'])
            penalty = (exp_loss * 8.33) + (song_loss * 5.0)
            reward = (p['exp'] * 0.1) + (p['tp'] * 0.0001)
            return p['cost'] + penalty - reward

        for p in patterns:
            if 'eval_score' not in p:
                p['eval_score'] = get_eval_score(p)

        # 5つの特化軸でそれぞれ上位を取得
        top_map = sorted(patterns, key=lambda x: x['map'], reverse=True)[:n]
        top_exp = sorted(patterns, key=lambda x: x['exp'], reverse=True)[:n]
        top_tp = sorted(patterns, key=lambda x: x['tp'], reverse=True)[:n]
        # チケット節約特化: 消費チケット(無料)が少なく、かつ eval_score が良い
        top_save = sorted(patterns, key=lambda x: (x['free6'] + x['free9'], x['eval_score']))[:n]
        # バランス型: eval_scoreが最も良い（低い）
        top_bal = sorted(patterns, key=lambda x: x['eval_score'])[:n]

        # 重複排除のためのユニーク処理
        unique_patterns = {}
        for p in top_map + top_exp + top_tp + top_save + top_bal:
            sig = (p['cost'], p['map'], p['exp'], p['tp'], p['free6'], p['free9'], p['map_designated_used'])
            if sig not in unique_patterns:
                unique_patterns[sig] = p
                
        return list(unique_patterns.values())

    def solve(self, days_info, status):
        print("最適化モデルを構築中...")
        
        total_days = len(days_info)
        months_in_span = sorted(list(set([d['month'] for d in days_info])))

        # 日々の属性をグループ化
        groups = {}
        day_to_group = []
        for d_idx, day in enumerate(days_info):
            is_today = (d_idx == 0)
            key = (day['month'], day['has_exp_bonus'], day['has_map_bonus'], day['has_tp_boost'], is_today)
            if key not in groups:
                groups[key] = {
                    'num_days': 0,
                    'day_info': day,
                    'is_today': is_today,
                    'month': day['month'],
                    'dates': []
                }
            groups[key]['num_days'] += 1
            groups[key]['dates'].append(day['date'])
            day_to_group.append(key)

        prob = pulp.LpProblem("Chunithm_Hybrid_Optimization", pulp.LpMinimize)

        single_play_options = []
        for t in self.tickets.keys():
            for c in [True, False]:
                single_play_options.append((t, c, status['base_songs']))

        # グループごとにパターンを作成
        group_patterns = {}
        for key, g in groups.items():
            day = g['day_info']
            is_special = day['has_map_bonus'] or day['has_exp_bonus'] or day['has_tp_boost']
            
            if g['is_today']:
                max_k = status['max_play_count'] if status['max_play_count'] != float('inf') else 10
            elif is_special:
                max_k = 10
            else:
                max_k = 1
                
            if max_k > 10:
                max_k = 10

            all_patterns = []
            dp_state = {}

            # Step 1: 1回目のプレイ
            for opt in single_play_options:
                res = self.calc_play(opt, True, day)
                p = {
                    'cost': res['cost'], 'map': res['map'], 'exp': res['exp'], 'tp': res['tp'],
                    'songs': res['songs'],
                    'free6': 1 if opt[0] == 'free6' else 0,
                    'free9': 1 if opt[0] == 'free9' else 0,
                    'map_designated_used': res['map'] if opt[1] else 0,
                    'k': 1,
                    'opts': [opt]
                }
                if g['is_today'] and p['cost'] > status['max_budget']:
                    continue
                dp_state.setdefault(p['cost'], []).append(p)
                all_patterns.append(p)

            current_states = dp_state

            # Step 2: 2回目以降のプレイ（DPで柔軟な組み合わせを作成）
            for step in range(2, max_k + 1):
                next_states = {}
                for cost, patterns in current_states.items():
                    for p in patterns:
                        for opt in single_play_options:
                            # 重複した組み合わせ（A+B と B+A）を防ぐため、インデックス順を強制
                            if step > 2:
                                last_add_opt = p['opts'][-1]
                                if single_play_options.index(opt) < single_play_options.index(last_add_opt):
                                    continue
                            
                            res = self.calc_play(opt, False, day)
                            new_p = {
                                'cost': p['cost'] + res['cost'],
                                'map': p['map'] + res['map'],
                                'exp': p['exp'] + res['exp'],
                                'tp': p['tp'] + res['tp'],
                                'songs': p['songs'] + res['songs'],
                                'free6': p['free6'] + (1 if opt[0] == 'free6' else 0),
                                'free9': p['free9'] + (1 if opt[0] == 'free9' else 0),
                                'map_designated_used': p['map_designated_used'] + (res['map'] if opt[1] else 0),
                                'k': p['k'] + 1,
                                'opts': p['opts'] + [opt]
                            }
                            
                            if g['is_today'] and new_p['cost'] > status['max_budget']:
                                continue
                            
                            next_states.setdefault(new_p['cost'], []).append(new_p)
                            all_patterns.append(new_p)
                
                # 状態の爆発を防ぐため、各ステップの各コスト帯で特化型Top5のみを残す
                current_states = {}
                for c, pats in next_states.items():
                    current_states[c] = self._get_top_n_patterns(pats, day, status, n=5)

            # Step 3: 全プレイ回数が混ざった全パターンから、最終的なコスト帯別Top5を抽出
            final_patterns_dict = {}
            for p in all_patterns:
                final_patterns_dict.setdefault(p['cost'], []).append(p)
            
            group_final_patterns = []
            for cost, patterns in final_patterns_dict.items():
                best_for_cost = self._get_top_n_patterns(patterns, day, status, n=5)
                
                for p in best_for_cost:
                    # 選ばれたパターンの表示用テキストを整形
                    f_opt = p['opts'][0]
                    desc_f = f"{f_opt[0]}/{'指定' if f_opt[1] else '好き'}/{f_opt[2]}曲"
                    if p['k'] > 1:
                        opt_counts = Counter(p['opts'][1:])
                        add_strs = []
                        for a_opt, count in opt_counts.items():
                            add_strs.append(f"{count}回({a_opt[0]}/{'指定' if a_opt[1] else '好き'}/{a_opt[2]}曲)")
                        desc_a = " ＋ " + " / ".join(add_strs)
                    else:
                        desc_a = ""
                    p['desc'] = f"[{p['k']}回] {desc_f}{desc_a}"
                    group_final_patterns.append(p)

            group_patterns[key] = group_final_patterns

        x = {}
        for key, g in groups.items():
            x[key] = [pulp.LpVariable(f"x_{key}_{i}", lowBound=0, upBound=g['num_days'], cat='Integer') for i in range(len(group_patterns[key]))]

        overflow_designated = pulp.LpVariable("overflow_designated", lowBound=0)

        # 目的関数
        prob += pulp.lpSum([
            p['eval_score'] * x[key][i]
            for key in groups for i, p in enumerate(group_patterns[key])
        ]) + overflow_designated * 2.0

        # 制約：各グループで選ぶ日数の合計が、そのグループの実日数に等しい
        for key, g in groups.items():
            prob += pulp.lpSum(x[key]) == g['num_days']

        # 各月のTPノルマ制約
        for m in months_in_span:
            quota = status['tp_quotas'].get(m, 0)
            if quota > 0:
                prob += pulp.lpSum([
                    p['tp'] * x[key][i]
                    for key, g in groups.items() if g['month'] == m
                    for i, p in enumerate(group_patterns[key])
                ]) >= quota
        
        # マップ全体ノルマ制約
        total_map_target = status['remain_normal_map'] + status['remain_designated_map']
        prob += pulp.lpSum([
            p['map'] * x[key][i]
            for key in groups for i, p in enumerate(group_patterns[key])
        ]) >= total_map_target
        
        # 指定マップの消費制約
        prob += pulp.lpSum([
            p['map_designated_used'] * x[key][i]
            for key in groups for i, p in enumerate(group_patterns[key])
        ]) - overflow_designated <= status['remain_designated_map']

        # チケット残数制約
        prob += pulp.lpSum([
            p['free6'] * x[key][i]
            for key in groups for i, p in enumerate(group_patterns[key])
        ]) <= status['remain_free6']
        
        prob += pulp.lpSum([
            p['free9'] * x[key][i]
            for key in groups for i, p in enumerate(group_patterns[key])
        ]) <= status['remain_free9']

        print("計算を実行中...")
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != 'Optimal':
            print("\n【警告】条件が厳しすぎます。残りの予算や日数では目標を達成できません。")
            print("今日の予算上限を上げるか、ノルマの見直しが必要です。")
            return

        total_cost = 0; total_map = 0; total_exp = 0; map_designated = 0
        f6_used = 0; f9_used = 0; total_songs = 0
        monthly_tp_results = {m: 0 for m in months_in_span}

        allocations = {key: [] for key in groups}
        for key in groups:
            for i, p in enumerate(group_patterns[key]):
                days_assigned = int(round(pulp.value(x[key][i])))
                for _ in range(days_assigned):
                    allocations[key].append(p)

        schedule_output = []
        today_plan = None
        today_pattern_data = None

        for d_idx, day in enumerate(days_info):
            key = day_to_group[d_idx]
            p = allocations[key].pop(0)

            total_cost += p['cost']
            total_map += p['map']
            map_designated += p['map_designated_used']
            total_exp += p['exp']
            total_songs += p['songs']
            f6_used += p['free6']
            f9_used += p['free9']
            monthly_tp_results[day['month']] += p['tp']

            date_str = day['date'].strftime('%m/%d')
            is_special = (p['k'] > 1) or day['has_map_bonus'] or day['has_exp_bonus'] or day['has_tp_boost']
            
            if day['has_tp_boost']: b_str = "[TP2倍]"
            elif day['has_exp_bonus']: b_str = "[EXP1.5]"
            elif day['has_map_bonus']: b_str = "[Map+2]"
            else: b_str = "[通常日]"

            plan_str = f"{date_str} {b_str:<10} : {p['desc']} (消費: {p['cost']}円 / 獲得TP: {p['tp']:.0f})"
            
            if d_idx == 0:
                today_plan = plan_str
                today_pattern_data = p
            elif is_special:
                schedule_output.append(plan_str)

        print("\n========================================================")
        print(" 【今日の推奨アクション】")
        print("========================================================")
        
        if today_pattern_data and today_pattern_data['map_designated_used'] > 0:
            print("▶ 推奨マップ: 「キャラ指定あり」のマップ")
            print("   (理由: 指定キャラ編成によるマス進行ボーナスを活かすため)")
        else:
            print("▶ 推奨マップ: 「キャラ指定なし」のマップ (または任意のマップ)")
            print("   (理由: 今日は好きキャラでのEXP育成を優先するため、マップの縛りはありません)")
        print(f"▶ プレイ内容: {today_plan}")

        if "通常日" in today_plan and "1回" in today_plan:
            print("※今日は通常日のため、最低限のプレイに留めて未来のボーナス日にツケを回すのが最もお得です。")
        elif "TP2倍" in today_plan or "EXP1.5" in today_plan or "Map+2" in today_plan:
            print("※今日はボーナス日のため、上記のようにリソースを投入することが推奨されます。")

        print("\n========================================================")
        print(" 【明日以降のスケジュール見通し (ボーナス日/追加プレイ日のみ抜粋)】")
        print("========================================================")
        for line in schedule_output:
            print(line)
        print(f"※上記以外の日程は「1回プレイ (等倍/好きキャラ/{status['base_songs']}曲) 100円」を予定しています。")

        print("\n[最終到達見込みリザルト]")
        print(f"■ 今日からの総課金: {total_cost:,} 円")
        print(f"■ 獲得総マップ    : {total_map:,.1f} マス (目標: {total_map_target:,} マス)")
        
        overflow_val = pulp.value(overflow_designated) if pulp.value(overflow_designated) else 0.0
        print(f"  ├ 指定キャラ使用分: {map_designated:,.1f} マス (はみ出し分 {overflow_val:,.1f} マスを含む)")
        
        for m in months_in_span:
            if status['tp_quotas'].get(m, 0) > 0:
                print(f"■ {m}月獲得TP       : {monthly_tp_results[m]:,.0f} pt (目標: {status['tp_quotas'][m]:,} pt)")
            else:
                print(f"■ {m}月獲得TP       : {monthly_tp_results[m]:,.0f} pt (目標なし・余剰分)")
                
        print(f"■ 獲得EXP         : {total_exp:,.1f} EXP")
        print(f"■ 6倍チケット残り : {status['remain_free6'] - f6_used} 枚 / 9倍チケット残り: {status['remain_free9'] - f9_used} 枚")

def generate_default_tp_dates(start_date, end_date, year, month, exp_wds, map_wds):
    dates = []
    if exp_wds:
        for d in range(1, 8):
            try:
                dt = datetime.date(year, month, d)
                if dt.weekday() in exp_wds:
                    dates.append(dt)
                    break
            except ValueError: pass
    if map_wds:
        for d in range(8, 15):
            try:
                dt = datetime.date(year, month, d)
                if dt.weekday() in map_wds:
                    dates.append(dt)
                    break
            except ValueError: pass
    
    used_wds = set(exp_wds + map_wds)
    for d in range(15, 32):
        try:
            dt = datetime.date(year, month, d)
            if dt.weekday() not in used_wds:
                dates.append(dt)
                used_wds.add(dt.weekday())
            if len(dates) >= 4:
                break
        except ValueError:
            pass
            
    return [d for d in dates if start_date <= d <= end_date]

if __name__ == "__main__":
    print("チュウニズム 日々軌道修正ナビゲーター")
    print("現在の進捗を入力してください。（未入力でEnterを押すとカッコ内のデフォルト値が適用されます）\n")

    try:
        today_input = input("今日の日付をカンマ区切りで入力 (例: 2026,4,26) [未入力: PCの今日の日付]: ").strip()
        if today_input:
            t_year, t_month, t_day = map(int, today_input.split(','))
            start_date = datetime.date(t_year, t_month, t_day)
        else:
            start_date = datetime.date.today()
            
        end_date = datetime.date(2026, 7, 1)
        
        if start_date > end_date:
            print("エラー: 開始日が終了日(2026/7/1)を過ぎています。")
            exit()
            
        total_days = (end_date - start_date).days + 1
        
        months_in_span = []
        curr = start_date
        while curr <= end_date:
            if curr.month not in months_in_span:
                months_in_span.append(curr.month)
            curr += datetime.timedelta(days=1)

        songs_input = input("\n通っている店舗の曲数設定を入力してください (3または4) [未入力: 4]: ").strip()
        base_songs = int(songs_input) if songs_input in ['3', '4'] else 4

        tp_quotas = {}
        if 5 in months_in_span:
            tp5 = input("5月の目標TPまでの残り (未入力: 180000): ").strip()
            tp_quotas[5] = int(tp5) if tp5 else 180000
        else:
            tp_quotas[5] = 0
            
        if 6 in months_in_span:
            tp6 = input("6月の目標TPまでの残り (未入力: 180000): ").strip()
            tp_quotas[6] = int(tp6) if tp6 else 180000
        else:
            tp_quotas[6] = 0
            
        for m in months_in_span:
            if m not in [5, 6]:
                tp_quotas[m] = 0
        
        normal_map_input = input("\n「キャラ指定なし」マップの残りマス (未入力: 14000): ").strip()
        remain_normal_map = int(normal_map_input) if normal_map_input else 14000
        
        designated_map_input = input("「キャラ指定あり」マップの残りマス (未入力: 3000): ").strip()
        remain_designated_map = int(designated_map_input) if designated_map_input else 3000

        free6_input = input("EXP6倍チケットの残り枚数 (未入力: 40): ").strip()
        remain_free6 = int(free6_input) if free6_input else 40

        free9_input = input("EXP9倍チケットの残り枚数 (未入力: 20): ").strip()
        remain_free9 = int(free9_input) if free9_input else 20

        budget_input = input("\n今日の予算上限(円)を入力 (未入力: 制限なし): ").strip()
        max_budget = int(budget_input) if budget_input else float('inf')
        
        play_count_input = input("今日の最大プレイ可能クレジット数を入力 (未入力: 制限なし): ").strip()
        max_play_count = int(play_count_input) if play_count_input else float('inf')

        status = {
            'base_songs': base_songs,
            'tp_quotas': tp_quotas,
            'remain_normal_map': remain_normal_map,
            'remain_designated_map': remain_designated_map,
            'remain_free6': remain_free6,
            'remain_free9': remain_free9,
            'max_budget': max_budget,
            'max_play_count': max_play_count
        }

        print("\n曜日を数字で指定してください (0:月, 1:火, 2:水, 3:木, 4:金, 5:土, 6:日)")
        
        exp_wd_input = input("「EXP 1.5倍」の曜日をカンマ区切りで入力 (-1でなし) [未入力: 0,2]: ").strip()
        if not exp_wd_input:
            exp_weekdays = [0, 2]
        elif exp_wd_input == "-1":
            exp_weekdays = []
        else:
            exp_weekdays = [int(x.strip()) for x in exp_wd_input.split(',')]

        map_wd_input = input("「マップ +2」の曜日をカンマ区切りで入力 (-1でなし) [未入力: 1,4]: ").strip()
        if not map_wd_input:
            map_weekdays = [1, 4]
        elif map_wd_input == "-1":
            map_weekdays = []
        else:
            map_weekdays = [int(x.strip()) for x in map_wd_input.split(',')]

        tp_boost_dates = []
        
        print("\n5月と6月の「期間内に残っている」TPブースト日を入力してください。（4月や7月分は自動設定されます）")
        for m in months_in_span:
            if m in [5, 6]:
                tp_input = input(f"{m}月の日付(日のみ)をカンマ区切りで入力 [未入力: 自動設定]: ").strip()
                if tp_input:
                    for d in tp_input.split(','):
                        tp_boost_dates.append(datetime.date(2026, m, int(d.strip())))
                else:
                    tp_boost_dates.extend(generate_default_tp_dates(start_date, end_date, 2026, m, exp_weekdays, map_weekdays))
            else:
                tp_boost_dates.extend(generate_default_tp_dates(start_date, end_date, 2026, m, exp_weekdays, map_weekdays))

        days_info = []
        for i in range(total_days):
            curr_date = start_date + datetime.timedelta(days=i)
            days_info.append({
                'date': curr_date,
                'month': curr_date.month,
                'has_exp_bonus': (curr_date.weekday() in exp_weekdays),
                'has_map_bonus': (curr_date.weekday() in map_weekdays),
                'has_tp_boost': (curr_date in tp_boost_dates)
            })

        print("\n設定が完了しました。")
        optimizer = ChunithmHybridOptimizer()
        optimizer.solve(days_info, status)

    except Exception as e:
        print(f"\nエラーが発生しました。入力形式が正しいか確認してください。詳細: {e}")