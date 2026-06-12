"""
SecurityAgent — toggleable static security analysis on generated/modified files.

Phase 5 will implement:
  - Human-in-the-loop gate: user must enable security scanning per session
  - Calls LLM (TaskType.SECURITY_SCAN → llama-3.3-70b-versatile) to analyse
    file content for common vulnerabilities (injection, secrets exposure, etc.)
  - Populates AgentResult.security_findings list[dict]
  - Non-blocking by default: findings are reported, not auto-fixed

TODO (Phase 5): implement SecurityAgent.scan()
"""

from __future__ import annotations

# TODO (Phase 5): from .models import TaskType, AgentResult
# TODO (Phase 5): from .router import ModelRouter


class SecurityAgent:
    """Stub — to be implemented in Phase 5."""

    def scan(self, file_path: str, content: str) -> list[dict]:
        """Perform a static security scan on *content*.

        Args:
            file_path: Path of the file being scanned (for reporting).
            content:   File content to analyse.

        Returns:
            List of finding dicts: [{severity, rule, line, description}, …]

        TODO (Phase 5): implement LLM-driven security analysis with
                        human-in-the-loop permission gate.
        """
        raise NotImplementedError("SecurityAgent.scan() — implement in Phase 5")
