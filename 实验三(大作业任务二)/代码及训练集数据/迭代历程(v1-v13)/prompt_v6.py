def generate_prompt(ctx):
    """
    Prompt v6：自定义轻量标签匹配 + 目标标签偏好摘要 + 常态锚点
    目标：在不显著增加 token 的前提下，提升精准命中与 ±1.0。
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

    def mode_rating(distribution):
        if not distribution:
            return None
        return max(distribution, key=distribution.get)

    def parse_tags(raw):
        text = str(raw or '').strip()
        if not text:
            return []
        for ch in "[]'\"":
            text = text.replace(ch, '')
        parts = []
        seen = set()
        for item in text.split(','):
            tag = item.strip()
            if not tag or tag.isdigit() or tag in seen:
                continue
            seen.add(tag)
            parts.append(tag)
        return parts

    def overlap_tags(a, b):
        if not a or not b:
            return []
        bset = set(b)
        return [x for x in a if x in bset]

    def build_tag_preferences(history):
        scores = {}
        counts = {}
        for item in history:
            rating = item.get('rating', 3)
            for tag in parse_tags(item.get('tags')):
                scores[tag] = scores.get(tag, 0) + (rating - 3)
                counts[tag] = counts.get(tag, 0) + 1
        return scores, counts

    def top_target_tag_signals(target_tags, scores, counts):
        liked = []
        disliked = []
        for tag in target_tags:
            if tag not in scores:
                continue
            score = scores[tag]
            count = counts.get(tag, 0)
            if score > 0:
                liked.append((score, count, tag))
            elif score < 0:
                disliked.append((score, count, tag))
        liked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        disliked.sort(key=lambda x: (x[0], -x[1]))
        liked_text = '、'.join(x[2] for x in liked[:2]) if liked else '无明显偏好'
        disliked_text = '、'.join(x[2] for x in disliked[:2]) if disliked else '无明显回避'
        return liked_text, disliked_text

    def custom_similar(history, target_tags, target_director, n=2):
        scored = []
        for idx, item in enumerate(history):
            item_tags = parse_tags(item.get('tags'))
            common = overlap_tags(item_tags, target_tags)
            score = len(common) * 10
            if target_director and target_director != '未知':
                if short_text(item.get('director'), 25) == target_director:
                    score += 3
            score += idx * 0.01
            scored.append((score, common, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = []
        for score, common, item in scored:
            if score <= 0:
                continue
            picked.append((item, common))
            if len(picked) >= n:
                break
        return picked

    def representative_item(history, target_rating, target_tags):
        picked = None
        best_score = -1
        for idx, item in enumerate(history):
            if item.get('rating') != target_rating:
                continue
            common = overlap_tags(parse_tags(item.get('tags')), target_tags)
            score = len(common) * 10 + idx * 0.01
            if score > best_score:
                best_score = score
                picked = item
        return picked

    def pick_diverse_refs(pool, target_tags, n=3):
        scored = []
        for idx, item in enumerate(pool):
            common = overlap_tags(parse_tags(item.get('tags')), target_tags)
            score = len(common) * 10 + idx * 0.01
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen = []
        used_ids = set()
        ordered = [item for _, item in scored]
        for rating in [2, 3, 4, 5, 1]:
            for item in ordered:
                mid = item.get('movie_id') or item.get('movie_name')
                if item.get('rating') == rating and mid not in used_ids:
                    chosen.append(item)
                    used_ids.add(mid)
                    break
            if len(chosen) >= n:
                break
        for item in ordered:
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
    target_tags_list = parse_tags(movie.get('tags'))
    target_tags = '、'.join(target_tags_list[:5]) if target_tags_list else short_text(movie.get('tags'), 65)
    target_summary = short_text(movie.get('summary'), 185)

    system_prompt = """你是电影评分预测系统。根据给定信息，预测该用户对目标电影的评分。

评分含义（1-5整数）：
1=非常差  2=较差  3=一般  4=较好  5=非常好

预测原则：
- 分数是相对于该用户个人标准的，不是绝对电影质量
- 优先参考与目标片标签/导演更接近的历史，其次参考最近评分和高低分锚点
- 证据强时可偏离用户均分；证据弱时优先落在该用户常见分数档
- 无历史时参考跨用户示例的分数区间，勿机械默认3分

输出要求（必须严格遵守）：
- 不要输出分析、解释、理由或任何其他文字
- 回复有且仅有一行，格式为：[Result:X]（X为1-5的整数）"""

    if cold_start:
        ref_pool = build_cold_ref_pool()
        ref_items = pick_diverse_refs(ref_pool, target_tags_list, 3)
        ref_text = ctx.format_history_list(ref_items, 'compact')
        ref_avg = avg_rating(ref_items)
        ref_hint = f"（参考均分约{ref_avg:.1f}）" if ref_avg is not None else ''
        system_prompt += "\n\n冷启动：优先参考与目标片标签更接近的跨用户示例，避免系统性偏高。"

        user_prompt = f"""## 冷启动（无历史）

## 跨用户参考{ref_hint}
{ref_text}

## 目标电影
名称：{target_name} | 导演：{target_director}
类型：{target_tags}
简介：{target_summary}

[Result:"""
    else:
        avg = stats['avg']
        dist = stats.get('distribution', {})
        mode = mode_rating(dist)

        if avg <= 3.2:
            cal = "偏严格：3分=中性，2分=不满，4分=很喜欢。"
        elif avg >= 4.0:
            cal = "偏宽松：4分常见，3分=一般，2分=强烈不喜欢。"
        else:
            cal = "居中：对照常见分数档与锚点，3/4分边界需细辨。"
        system_prompt += f"\n\n{cal}"

        recent = ctx.get_history_sample(3, 'recent')
        highest = ctx.get_history_sample(1, 'highest')
        lowest = ctx.get_history_sample(1, 'lowest')
        similar_pairs = custom_similar(ctx.user_history, target_tags_list, target_director, 2)

        recent_avg = avg_rating(recent)
        recent_text = ctx.format_history_list(recent, 'compact')

        if similar_pairs:
            similar_lines = []
            similar_items = []
            for item, common in similar_pairs:
                similar_items.append(item)
                common_text = '、'.join(common[:2]) if common else '弱相关'
                similar_lines.append(f"{movie_title(item)}({item.get('rating')}分, 命中:{common_text})")
            similar_text = f"均分{avg_rating(similar_items):.1f} → " + '；'.join(similar_lines)
        else:
            similar_text = '无'

        tag_scores, tag_counts = build_tag_preferences(ctx.user_history)
        liked_tags, disliked_tags = top_target_tag_signals(target_tags_list, tag_scores, tag_counts)

        taste_lines = []
        if highest:
            hi = highest[0]
            taste_lines.append(
                f"上限{movie_title(hi)}({hi.get('rating')}分) "
                f"[{short_text(hi.get('tags'), 18)}] {short_text(hi.get('comment'), 24)}"
            )
        if lowest:
            lo = lowest[0]
            if not highest or lo.get('movie_id') != highest[0].get('movie_id'):
                taste_lines.append(
                    f"下限{movie_title(lo)}({lo.get('rating')}分) "
                    f"[{short_text(lo.get('tags'), 18)}] {short_text(lo.get('comment'), 24)}"
                )
        taste_text = ' | '.join(taste_lines) if taste_lines else '无'

        mode_item = representative_item(ctx.user_history, mode, target_tags_list) if mode is not None else None
        mode_text = (
            f"{movie_title(mode_item)}({mode_item.get('rating')}分) [{short_text(mode_item.get('tags'), 18)}]"
            if mode_item else '无'
        )

        profile = f"均分{avg:.1f} 区间{stats['min']}-{stats['max']}"
        if recent_avg is not None:
            profile += f" 最近均分{recent_avg:.1f}"
        if mode is not None:
            profile += f" 常见{mode}分"
        profile += f" | {format_dist(dist)}"

        user_prompt = f"""## 用户画像
{profile}

## 最近评分
{recent_text}

## 目标标签偏好
偏好：{liked_tags} | 回避：{disliked_tags}

## 最相关历史
{similar_text}

## 常态锚点
{mode_text}

## 极值锚点
{taste_text}

## 目标电影
名称：{target_name} | 导演：{target_director}
类型：{target_tags}
简介：{target_summary}

[Result:"""

    return system_prompt, user_prompt
