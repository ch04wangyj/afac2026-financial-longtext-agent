"""V15 离线 Qwen-VL 图像化表格提取。

在文档预处理阶段，对 B 类图像页（整页贴图、文字层极少、上下文含财务术语）
调用 Qwen-VL 把图像化表格转成结构化 Markdown 文本。该阶段产生的 Token
不计入最终评测 Token，且仅使用 Qwen 系列模型，满足赛题约束。

输出文本作为 V14 语料的增量补充（layout_vl_table_row），不替换已有语料。
"""

from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz

from agent.config import get_api_key
from agent.schemas import TokenUsage


NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?:%|％|元|万元|亿元|万|亿|倍|pp)?")
YEAR_RE = re.compile(r"20\d{2}\s*年?")
UNIT_RE = re.compile(r"(?:百万元|千元|万元|亿元|港元|美元|元|%)")

# 章节封面/声明/签名页特征词，命中则跳过
SKIP_KEYWORDS = (
    "授权委托书",
    "声明",
    "签名",
    "签章",
    "盖章",
    "第.*节",
    "第.*章",
    "目\\s*录",
    "声明页",
)

# 财务术语，B 类页上下文必须命中至少一个才视为可能含数据表
FINANCIAL_CONTEXT_TERMS = (
    "营业收入",
    "净利润",
    "资产",
    "负债",
    "现金流",
    "分红",
    "股息",
    "同比",
    "环比",
    "增长率",
    "比率",
    "金额",
    "万元",
    "亿元",
    "表",
)


@dataclass(frozen=True)
class VLExtractConfig:
    """控制 Qwen-VL 提取行为。"""

    dpi: int = 150
    min_text_chars: int = 4
    max_text_chars: int = 150
    min_image_ratio: float = 0.5
    skip_first_pages: int = 3
    skip_last_pages: int = 2
    vl_model: str = "qwen-vl-max"
    vl_max_tokens: int = 2048
    vl_temperature: float = 0.1
    request_timeout_seconds: int = 120
    max_retries: int = 2


@dataclass
class BClassPage:
    """一个被判定为 B 类（疑似图像化数据表）的页面。"""

    domain: str
    doc_id: str
    file_name: str
    page_index: int  # 0-based
    text_preview: str
    image_ratio: float


@dataclass
class VLTableResult:
    """Qwen-VL 提取单页表格的结果。"""

    domain: str
    doc_id: str
    file_name: str
    page_index: int
    text: str
    valid: bool
    invalid_reason: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)


def scan_b_class_pages(raw_root: Path, config: VLExtractConfig | None = None) -> list[BClassPage]:
    """扫描全量 PDF，返回 B 类图像页清单。

    B 类判定：图像占比 > min_image_ratio、文字层字符数 < max_text_chars、
    不是封面/末页、上下文含财务术语。
    """
    cfg = config or VLExtractConfig()
    results: list[BClassPage] = []
    seen_paths: set[str] = set()
    for domain in ("financial_reports", "financial_contracts", "research"):
        domain_dir = raw_root / domain
        if not domain_dir.exists():
            continue
        for pdf_path in sorted(list(domain_dir.glob("*.pdf")) + list(domain_dir.glob("*.PDF"))):
            resolved = str(pdf_path.resolve()).lower()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            results.extend(_scan_pdf_b_class(pdf_path, domain, cfg))
    return results


def _scan_pdf_b_class(pdf_path: Path, domain: str, cfg: VLExtractConfig) -> list[BClassPage]:
    """扫描单个 PDF 的所有页，返回 B 类页。"""
    doc_id = pdf_path.stem
    pages: list[BClassPage] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return pages
    try:
        total_pages = len(doc)
        page_texts: list[str] = []
        page_ratios: list[float] = []
        page_has_img: list[bool] = []
        for page in doc:
            txt = page.get_text("text")
            img_area = 0.0
            page_area = page.rect.width * page.rect.height
            for img_info in page.get_image_info():
                bbox = img_info.get("bbox", (0, 0, 0, 0))
                img_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            ratio = img_area / page_area if page_area else 0.0
            page_texts.append(txt)
            page_ratios.append(ratio)
            page_has_img.append(bool(page.get_images()))
        for i in range(total_pages):
            if not page_has_img[i] or page_ratios[i] < cfg.min_image_ratio:
                continue
            txt = page_texts[i]
            if len(txt) >= cfg.max_text_chars or len(txt) < cfg.min_text_chars:
                continue
            if i < cfg.skip_first_pages or i >= total_pages - cfg.skip_last_pages:
                continue
            if _is_skip_page(txt):
                continue
            ctx = ""
            for j in range(max(0, i - 1), min(total_pages, i + 2)):
                ctx += page_texts[j]
            if not any(term in ctx for term in FINANCIAL_CONTEXT_TERMS):
                continue
            pages.append(
                BClassPage(
                    domain=domain,
                    doc_id=doc_id,
                    file_name=pdf_path.name,
                    page_index=i,
                    text_preview=txt[:80],
                    image_ratio=round(page_ratios[i], 2),
                )
            )
    finally:
        doc.close()
    return pages


def _is_skip_page(text: str) -> bool:
    """判断是否为章节封面/声明/签名页。"""
    for pattern in SKIP_KEYWORDS:
        if re.search(pattern, text):
            return True
    return False


def render_page_to_png_bytes(pdf_path: Path, page_index: int, dpi: int = 300) -> bytes:
    """用 PyMuPDF 把指定页渲染为 PNG 字节流。"""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
        return png_bytes
    finally:
        doc.close()


def call_qwen_vl(
    image_png_bytes: bytes,
    prompt: str,
    config: VLExtractConfig | None = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> tuple[str, TokenUsage]:
    """调用 Qwen-VL 提取图像中的表格文本。

    返回 (模型输出文本, TokenUsage)。TokenUsage 用于离线审计，
    不计入最终评测 Token。
    """
    cfg = config or VLExtractConfig()
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing Qwen API key for VL extraction. "
            "Set DASHSCOPE_API_KEY/BAILIAN_API_KEY/QWEN_API_KEY."
        )
    image_b64 = base64.b64encode(image_png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{image_b64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    url = base_url.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            import requests

            payload = {
                "model": cfg.vl_model,
                "messages": messages,
                "temperature": cfg.vl_temperature,
                "max_tokens": cfg.vl_max_tokens,
            }
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=cfg.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = message.get("content", "") or ""
            raw_usage = data.get("usage") or {}
            usage = TokenUsage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
                total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
            )
            if usage.total_tokens == 0:
                usage = TokenUsage(
                    prompt_tokens=max(1, len(image_b64) // 6),
                    completion_tokens=max(1, len(text) // 2),
                )
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            return text, usage
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Qwen-VL API call failed after retries: {last_error}") from last_error


TABLE_EXTRACTION_PROMPT = (
    "请仔细观察这张金融文档页面图像。\n"
    "如果页面包含任何数值数据（表格、图表、价格走势中的数值），请提取为结构化文本。\n"
    "要求：\n"
    "1. 如果是表格，转为Markdown表格\n"
    "2. 如果是图表（折线图/柱状图等），提取图表标题、坐标轴标签和可辨识的数据点数值\n"
    "3. 保留所有数值、年份、单位和指标名称\n"
    "4. 如果页面不含任何数值数据，只输出：NOT_A_TABLE\n"
    "5. 只输出数据内容，不要添加解释说明\n"
)


def extract_table_from_page(
    pdf_path: Path,
    page_index: int,
    domain: str,
    doc_id: str,
    config: VLExtractConfig | None = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> VLTableResult:
    """提取单个页面的表格文本，包含渲染、VL 调用和确定性校验。"""
    cfg = config or VLExtractConfig()
    png_bytes = render_page_to_png_bytes(pdf_path, page_index, cfg.dpi)
    text, usage = call_qwen_vl(png_bytes, TABLE_EXTRACTION_PROMPT, cfg, base_url)
    text = text.strip()
    valid, reason = validate_table_output(text)
    return VLTableResult(
        domain=domain,
        doc_id=doc_id,
        file_name=pdf_path.name,
        page_index=page_index,
        text=text,
        valid=valid,
        invalid_reason=reason,
        usage=usage,
        metadata={"vl_model": cfg.vl_model, "image_dpi": cfg.dpi},
    )


def validate_table_output(text: str) -> tuple[bool, str]:
    """对 Qwen-VL 输出做确定性校验。

    返回 (是否通过, 不通过原因)。校验规则：
    - NOT_A_TABLE 标记视为无效（但记录为正常跳过）
    - 必须包含至少 2 个数值
    - 必须包含至少 1 个年份或至少 1 个单位
    - 文本长度不能过短（<20 字）或过长（>8000 字）
    """
    if not text:
        return False, "empty_output"
    if text == "NOT_A_TABLE":
        return False, "not_a_table"
    if len(text) < 20:
        return False, "too_short"
    if len(text) > 8000:
        return False, "too_long"
    numbers = NUMBER_RE.findall(text)
    if len(numbers) < 2:
        return False, "insufficient_numbers"
    has_year = bool(YEAR_RE.search(text))
    has_unit = bool(UNIT_RE.search(text))
    if not has_year and not has_unit:
        return False, "no_year_no_unit"
    return True, ""
