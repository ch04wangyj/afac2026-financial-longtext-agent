"""Agent 主流程共享的数据结构定义。

这些 schema 是脚本、检索、压缩、推理和输出之间的稳定接口，后续扩展 V2/V3
时应优先保持字段兼容。
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


AnswerFormat = Literal["mcq", "multi", "tf"]


@dataclass
class TokenUsage:
    """单次或多次 LLM 调用的 Token 统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        """如果上游只给了 prompt/completion，则自动补 total。"""
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """原地累加另一份 TokenUsage，便于汇总整批题目。"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        return self

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Question:
    """比赛题目结构，兼容单选、多选和判断题。"""

    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: AnswerFormat
    type: str = ""
    doc_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        """从原始 JSON 题目转成强类型对象。"""
        return cls(
            qid=str(data["qid"]),
            domain=str(data["domain"]),
            split=str(data.get("split", "")),
            question=str(data["question"]).strip(),
            options={str(k): str(v).strip() for k, v in dict(data.get("options", {})).items()},
            answer_format=data["answer_format"],
            type=str(data.get("type", "")),
            doc_ids=[str(x) for x in data.get("doc_ids", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    """原始文档及解析后的全文信息。"""

    doc_id: str
    domain: str
    title: str
    path: str
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path_obj(self) -> Path:
        """把序列化保存的 path 字符串恢复为 Path。"""
        return Path(self.path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass
class Chunk:
    """检索索引的最小证据块，保留页码、条款、数字和日期等元数据。"""

    chunk_id: str
    doc_id: str
    domain: str
    page: int | None
    section: str
    clause_id: str
    text: str
    tables: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class RetrievalResult:
    """一次检索命中的证据块及其分数和来源。"""

    chunk_id: str
    doc_id: str
    domain: str
    score: float
    source: str
    query: str
    evidence_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        return cls(**data)


@dataclass
class EvidencePack:
    """一组结构化展开后的证据块，保留 anchor 与成员列表。"""

    pack_id: str
    doc_id: str
    anchor_chunk_id: str
    member_chunk_ids: list[str]
    score: float
    source: str
    query: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidencePack":
        return cls(**data)


@dataclass
class AnswerResult:
    """单题最终答案、证据、Token 和原始模型输出。"""

    qid: str
    answer: str
    confidence: float
    evidence: list[RetrievalResult]
    token_usage: TokenUsage
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """递归序列化，保证 evidence/token_usage 也能写入 JSONL。"""
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        data["token_usage"] = self.token_usage.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerResult":
        return cls(
            qid=data["qid"],
            answer=data["answer"],
            confidence=float(data.get("confidence", 0.0)),
            evidence=[RetrievalResult.from_dict(x) for x in data.get("evidence", [])],
            token_usage=TokenUsage(**data.get("token_usage", {})),
            raw_response=data.get("raw_response", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LogicNode:
    """LogicRAG 规划中的单个子问题节点。"""

    node_id: str
    text: str
    depends_on: list[str] = field(default_factory=list)
    rank: int = 0
    pruned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicNode":
        return cls(
            node_id=str(data.get("node_id") or data.get("id") or "").strip(),
            text=str(data.get("text") or data.get("question") or "").strip(),
            depends_on=[str(item).strip() for item in data.get("depends_on", []) if str(item).strip()],
            rank=int(data.get("rank", 0) or 0),
            pruned=bool(data.get("pruned", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def normalized_text(self) -> str:
        """用于去重的轻量文本归一化。"""
        return " ".join(self.text.split()).casefold()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogicPlan:
    """LogicRAG 规划结果，保存 DAG 节点与整体说明。"""

    nodes: list[LogicNode] = field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicPlan":
        node_rows = data.get("nodes") or data.get("subproblems") or []
        return cls(
            nodes=[LogicNode.from_dict(row) for row in node_rows],
            rationale=str(data.get("rationale", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    def node_map(self) -> dict[str, LogicNode]:
        return {node.node_id: node for node in self.nodes}

    def topological_levels(self) -> list[list[str]]:
        """按依赖层级返回节点 ID，并同步写回 rank。"""
        node_map = self.node_map()
        node_ids = [node.node_id for node in self.nodes]
        node_set = set(node_ids)
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing = {node_id: [] for node_id in node_ids}
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in node_set:
                    continue
                indegree[node.node_id] += 1
                outgoing[dep].append(node.node_id)

        queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
        levels: list[list[str]] = []
        visited = 0
        current_rank = 0
        while queue:
            level_size = len(queue)
            level: list[str] = []
            for _ in range(level_size):
                node_id = queue.popleft()
                level.append(node_id)
                visited += 1
                node_map[node_id].rank = current_rank
                for child in outgoing[node_id]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        queue.append(child)
            levels.append(level)
            current_rank += 1

        if visited != len(self.nodes):
            raise ValueError("logic plan contains cycle")
        return levels
