"""计算辅助工具。

V1 只提供安全数值解析；V2 可在这里扩展增长率、利率、保险公式等 Calculator。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


def safe_decimal(value: str) -> Decimal | None:
    """把中文金额/比例字符串安全转换为 Decimal。"""
    cleaned = value.replace(",", "").replace("，", "").strip()
    multiplier = Decimal("1")
    if cleaned.endswith("亿"):
        multiplier = Decimal("100000000")
        cleaned = cleaned[:-1]
    elif cleaned.endswith("万"):
        multiplier = Decimal("10000")
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace("元", "").replace("%", "").replace("％", "")
    try:
        return Decimal(cleaned) * multiplier
    except (InvalidOperation, ValueError):
        return None
