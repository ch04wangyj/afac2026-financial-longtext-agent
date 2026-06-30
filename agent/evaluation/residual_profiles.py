"""解析可继承的残差复核配置。"""

from __future__ import annotations


def resolve_profile_reviews(
    profiles: dict[str, dict],
    profile_name: str,
    trail: tuple[str, ...] = (),
) -> dict[str, dict]:
    """递归合并 profile，子配置可覆盖父配置但不能形成循环。"""
    if profile_name in trail:
        raise ValueError(f"残差 profile 继承形成循环: {' -> '.join((*trail, profile_name))}")
    if profile_name not in profiles:
        raise KeyError(f"未知残差 profile: {profile_name}")
    profile = profiles[profile_name]
    parent_name = str(profile.get("extends", "")).strip()
    reviews = (
        resolve_profile_reviews(profiles, parent_name, (*trail, profile_name))
        if parent_name
        else {}
    )
    reviews.update(dict(profile.get("reviews", {})))
    return reviews
