"""Shared probe cases for retrieval system diagnostics."""

from __future__ import annotations

BYD_2025_NET_PROFIT = {
    "name": "byd_2025_net_profit",
    "target_doc_id": "annual_byd_2025_report",
    "target_answer_terms": (
        "归属于母公司所有者的净利润",
        "32,619,022",
    ),
    "keyword_bundles": [
        ("比亚迪集团", "2025", "归母净利润"),
        ("比亚迪集团", "2025", "归属于母公司的净利润"),
        ("比亚迪集团", "2025", "归属于母公司所有者的净利润"),
    ],
}

MIDEA_2025_REVENUE_GROWTH = {
    "name": "midea_2025_revenue_growth",
    "target_doc_id": "annual_midea_2025_report",
    "answer_match_mode": "any_term_in_target_doc_chunk",
    "target_answer_terms": (
        "2025年，公司营业总收入4,585亿元，同比增长12%",
        "12%",
        "一、营业总收入\n458,502,407\n409,084,266",
    ),
    "keyword_bundles": [
        ("美的", "2025", "营业总收入增长率"),
        ("美的集团", "2025", "营业总收入"),
    ],
}

MIDEA_2024_SHARE_REPURCHASE = {
    "name": "midea_2024_share_repurchase",
    "target_doc_id": "annual_midea_2024_report",
    "target_answer_terms": (
        "在稳定分红派现的同时，公司持续推出实施了一系列股份回购的方案，自2019年起公司连续四年推出回购计划，持续用于实施公司股权激励计划及员工持股计划，维护公司市值稳定与全体股东利益。",
    ),
    "keyword_bundles": [
        ("美的", "2019", "股份回购方案"),
        ("美的集团", "2019", "连续实施", "股份回购方案"),
        ("美的集团", "2019", "实施", "股份回购方案"),
    ],
}

PBC_2025_BENEFICIAL_OWNER_DIFF = {
    "name": "pbc_2025_beneficial_owner_diff",
    "target_doc_id": "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）",
    "target_answer_terms": (
        "第二十七条 金融机构查询核对后发现差异的，应当与客户进行必要的沟通、核实。金融机构有合理理由认为由于自身识别不准确而导致差异的，应当更正其识别的受益所有人信息。金融机构有合理理由认为由于备案信息不准确而导致差异且差异重大的，应当在发现差异之日起30个工作日内通过受益所有人信息查询管理系统提交差异报告，记录并留存报告的理由、差异确认的过程，并提供相关佐证材料。",
    ),
    "keyword_bundles": [
        ("《金融机构客户受益所有人识别管理办法》", "受益所有人信息", "重大差异", "30 个工作日", "差异报告"),
        ("《金融机构客户尽职调查和客户身份资料及交易记录保存管理办法》", "受益所有人信息", "重大差异", "30 个工作日", "差异报告"),
    ],
}

PROBE_CASES = [
    BYD_2025_NET_PROFIT,
    MIDEA_2025_REVENUE_GROWTH,
    MIDEA_2024_SHARE_REPURCHASE,
    PBC_2025_BENEFICIAL_OWNER_DIFF,
]

PROBE_CASES_BY_NAME = {case["name"]: case for case in PROBE_CASES}
