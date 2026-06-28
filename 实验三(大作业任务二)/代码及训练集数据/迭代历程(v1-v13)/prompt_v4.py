
def generate_prompt(ctx):
    """
    Prompt v4：v2 保守增强版
    测试集得分 95.20（#792，2026-06-21）
    """
    movie = ctx.target_movie
    stats = ctx.get_user_stats()
    cold_start = stats.get('count', 0) == 0

    def movie_title(m):
        return m.get('name') or m.get('movie_name') or '未知'

    def short_text(text, limit):
        s = str(text or '').strip()
        if not s:
            return '未知'
        return s[:limit] + ('...' if len(s) > limit else '')

    def format_dist(distribution):
        if not distribution:
            return '无'
        return '，'.join(f"{k}分×{v}" for k, v in sorted(distribution.items()))

    def avg_rating(items):
        if not items:
            return None
        return sum(x.get('rating', 3) for x in items) / len(items)

    def pick_diverse_refs(pool, n=3):
        chosen = []
        used_ids = set()
        for rating in [2, 3, 4, 5, 1]:
            for item in pool:
                mid = item.get('movie_id') or item.get('movie_name')
                if item.get('rating') == rating and mid not in used_ids:
                    chosen.append(item)
                    used_ids.add(mid)
                    break
            if len(chosen) >= n:
                break
        for item in pool:
            mid = item.get('movie_id') or item.get('movie_name')
            if mid not in used_ids:
                chosen.append(item)
                used_ids.add(mid)
            if len(chosen) >= n:
                break
        return chosen[:n]

    def build_cold_ref_pool():
        pool = []
        seen = set()
        for _ in range(6):
            batch = ctx.get_random_user_history(2)
            for item in batch:
                key = (item.get('movie_id'), item.get('movie_name'), item.get('rating'))
                if key not in seen:
                    seen.add(key)
                    pool.append(item)
        return pool

    target_name = movie_title(movie)
    target_director = short_text(movie.get('director'), 25)
    target_tags = short_text(movie.get('tags'), 65)
    target_summary = short_text(movie.get('summary'), 195)

    system_prompt = """你是电影评分预测系统。根据给定信息，预测该用户对目标电影的评分。

评分含义（1-5整数）：
1=非常差  2=较差  3=一般  4=较好  5=非常好

预测原则：
- 分数是相对于该用户个人标准的，不是绝对电影质量
- 综合参考：用户均分、最近均分、同类型历史均分、高低分锚点
- 同类型历史均分是同类电影的重要参考，但目标片明显更好/更差时可偏离
- 无历史时参考跨用户示例的分数区间，勿机械默认3分

输出要求（必须严格遵守）：
- 不要输出分析、解释、理由或任何其他文字
- 回复有且仅有一行，格式为：[Result:X]（X为1-5的整数）"""

    if cold_start:
        ref_pool = build_cold_ref_pool()
        ref_items = pick_diverse_refs(ref_pool, 3)
        ref_text = ctx.format_history_list(ref_items, 'compact')
        ref_avg = avg_rating(ref_items)
        ref_hint = f"（参考均分约{ref_avg:.1f}）" if ref_avg is not None else ''
        system_prompt += "\n\n冷启动：结合参考示例分数区间与目标电影特征判断，避免系统性偏高。"

        user_prompt = f"""## 冷启动用户（无历史）

## 跨用户评分参考{ref_hint}
{ref_text}

## 目标电影
名称：{target_name} | 导演：{target_director}
类型：{target_tags}
简介：{target_summary}

[Result:"""
    else:
        avg = stats['avg']
        if avg <= 3.2:
            cal = "该用户偏严格：3分=中性，2分=明显不满，4分=很喜欢，5分=极少给出。"
        elif avg >= 4.0:
            cal = "该用户偏宽松：4分较常见，3分=一般或略失望，2分=强烈不喜欢。"
        else:
            cal = "该用户打分居中：请严格对照其历史区间与锚点，3/4分边界需细辨。"
        system_prompt += f"\n\n{cal}"

        recent = ctx.get_history_sample(3, 'recent')
        similar = ctx.get_similar_movies(3)
        highest = ctx.get_history_sample(1, 'highest')
        lowest = ctx.get_history_sample(1, 'lowest')

        recent_avg = avg_rating(recent)
        recent_text = ctx.format_history_list(recent, 'compact')

        if similar:
            sim_avg = avg_rating(similar)
            top = similar[0]
            similar_text = (
                f"最似{movie_title(top)}({top.get('rating')}分) 均分{sim_avg:.1f} → "
                + ctx.format_history_list(similar, 'compact')
            )
        else:
            similar_text = '无'

        taste_lines = []
        if highest:
            hi = highest[0]
            taste_lines.append(
                f"上限：{movie_title(hi)}({hi.get('rating')}分) {short_text(hi.get('comment'), 30)}"
            )
        if lowest:
            lo = lowest[0]
            if not highest or lo.get('movie_id') != highest[0].get('movie_id'):
                taste_lines.append(
                    f"下限：{movie_title(lo)}({lo.get('rating')}分) {short_text(lo.get('comment'), 30)}"
                )
        taste_text = ' | '.join(taste_lines) if taste_lines else '无'

        recent_hint = f"最近均分{recent_avg:.1f} | " if recent_avg is not None else ''
        user_prompt = f"""## 用户画像
{recent_hint}均分{stats['avg']:.1f} 区间{stats['min']}-{stats['max']} | {format_dist(stats.get('distribution', {}))}

## 最近评分
{recent_text}

## 同类型历史
{similar_text}

## 口味锚点
{taste_text}

## 目标电影
名称：{target_name} | 导演：{target_director}
类型：{target_tags}
简介：{target_summary}

[Result:"""

    return system_prompt, user_prompt
