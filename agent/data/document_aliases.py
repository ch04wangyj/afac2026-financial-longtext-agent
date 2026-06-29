"""公开 A 组文档标识与业务名称映射。

部分原始文件只以数字或 pack 编号命名，检索证据若只展示 doc_id，模型无法判断
条款属于哪个产品。这里保存由公开题目和原始材料可直接确定的别名，不包含答案。
"""

from __future__ import annotations

import re

from agent.schemas import Question


DOCUMENT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "insurance": {
        "1": ("平安智盈金生",),
        "2": ("国寿增益宝",),
        "3": ("众安白血病医疗险", "白血病医疗险"),
        "4": ("平安安佑福", "安佑福重疾险"),
        "5": ("平安e生保", "e生保"),
        "6": ("太保团体百万医疗", "团体百万医疗", "太保"),
        "7": ("平安预防接种意外险", "预防接种意外险"),
        "8": ("众安营运交通意外险", "营运交通意外险"),
        "9": ("平安特种车险",),
        "10": ("众安特种车险",),
        "11": ("平安家财险",),
        "12": ("众安家财险",),
        "13": ("众安食责险", "众安食品安全责任险"),
        "14": ("平安食品安全责任险",),
        "15": ("国寿鑫享添盈",),
        "16": ("平安富鸿金生",),
    },
    "financial_reports": {
        "annual_byd_2024_report": ("比亚迪", "比亚迪2024"),
        "annual_byd_2025_report": ("比亚迪", "比亚迪2025"),
        "annual_catl_2024_report": ("宁德时代", "宁德时代2024"),
        "annual_catl_2025_report": ("宁德时代", "宁德时代2025"),
        "annual_chinamobile_2025_report": ("中国移动", "中国移动2025"),
        "annual_cscec_2024_report": ("中国建筑", "中国建筑2024"),
        "annual_cscec_2025_report": ("中国建筑", "中国建筑2025"),
        "annual_midea_2024_report": ("美的集团", "美的2024"),
        "annual_midea_2025_report": ("美的集团", "美的2025"),
    },
    "research": {
        "pack2_text01": ("银保渠道", "韩国寿险", "保险行业"),
        "pack2_text02": ("网络安全运营", "安全运营"),
        "pack2_text03": ("服务消费", "消费趋势"),
        "pack2_text04": ("电动车行业策略", "锂电估值"),
        "pack2_text06": ("化工周报", "大宗商品"),
        "pack2_text07": ("车展前瞻", "新能源渗透率"),
        "pack2_text08": ("基金份额", "主动型新发"),
        "pack2_text09": (
            "芯原股份",
            "芯片设计服务",
            "芯片定制服务",
            "IP 授权",
            "IP授权",
            "数据中心半导体",
            "集成电路",
        ),
        "pack2_text10": ("上市券商", "客户资金杠杆"),
        "pack2_text11": ("光通信",),
        "pack2_text13": ("宠物医疗",),
        "pack2_text14": ("金融机构配置",),
        "pack2_text17": ("银行IT", "银行 IT"),
        "pack2_text19": ("远望谷", "RFID"),
        "pack2_text20": ("经济与金融趋势",),
    },
    "regulatory": {
        "csrc_0038_att1": (
            "上市公司年度报告",
            "年度报告",
            "定期报告",
            "董事会审议",
        ),
        "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）": (
            "客户尽职调查",
            "反洗钱调查",
            "空壳银行",
            "解除保险合同",
            "交易记录",
            "核实申请人身份",
        ),
    },
}


def document_label(domain: str, doc_id: str, fallback_title: str = "") -> str:
    """返回可读文档名；多个别名只展示主名称。"""
    aliases = DOCUMENT_ALIASES.get(domain, {}).get(str(doc_id), ())
    if aliases:
        return aliases[0]
    title = str(fallback_title or "").strip()
    return title if title and title != str(doc_id) else str(doc_id)


def option_doc_scope(question: Question, option_text: str) -> list[str]:
    """选项明确命中文档业务名时缩小范围，否则保持题目给定的完整 doc_ids。"""
    ordinal_patterns = (
        (0, ("第一份文档", "第一个文档", "文档一", "前一份文档", "前者")),
        (1, ("第二份文档", "第二个文档", "文档二", "后一份文档", "后者")),
    )
    ordinal_matches = [
        question.doc_ids[index]
        for index, markers in ordinal_patterns
        if index < len(question.doc_ids) and any(marker in option_text for marker in markers)
    ]
    if ordinal_matches:
        return list(dict.fromkeys(str(doc_id) for doc_id in ordinal_matches))

    aliases_by_doc = DOCUMENT_ALIASES.get(question.domain, {})
    mentioned_years = set(re.findall(r"(?:19|20)\d{2}", option_text))
    matched: list[str] = []
    for doc_id in question.doc_ids:
        doc_id = str(doc_id)
        if question.domain == "financial_reports" and mentioned_years:
            doc_years = set(re.findall(r"(?:19|20)\d{2}", doc_id))
            if doc_years and not (doc_years & mentioned_years):
                continue
        aliases = list(aliases_by_doc.get(doc_id, ()))
        if question.domain == "financial_contracts":
            # 题目使用 fc_text_003，原文件与索引使用 text03。
            number_match = re.fullmatch(r"text0*(\d+)", doc_id, flags=re.IGNORECASE)
            if number_match:
                number = int(number_match.group(1))
                aliases.extend((f"fc_text_{number:03d}", f"fc_text_{number:02d}"))
        if any(alias and alias in option_text for alias in aliases):
            matched.append(doc_id)
    return matched or list(question.doc_ids)
