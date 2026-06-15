"""Token 用量累加器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.schemas import TokenUsage


@dataclass
class TokenTracker:
    """记录多次 LLM 调用的 usage，便于调试批处理成本。"""

    calls: list[TokenUsage] = field(default_factory=list)

    def record(self, usage: TokenUsage) -> None:
        """追加一次调用的 Token 用量。"""
        self.calls.append(usage)

    def total(self) -> TokenUsage:
        """汇总所有已记录调用。"""
        total = TokenUsage()
        for usage in self.calls:
            total.add(usage)
        return total
