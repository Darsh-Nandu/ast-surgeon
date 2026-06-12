"""
SystemInfoAgent — gathers lightweight system/environment information.

Phase 1 will implement:
  - OS, Python version, installed packages, available CLI tools
  - Project structure summary (files, entry points, test runner detection)
  - Result cached per session to avoid redundant LLM calls

TODO (Phase 1): implement SystemInfoAgent.gather()
"""

from __future__ import annotations

# TODO (Phase 1): import ModelRouter, TaskType, AgentResult


class SystemInfoAgent:
    """Stub — to be implemented in Phase 1."""

    def gather(self) -> dict:
        """Collect and summarise system/environment info.

        Returns:
            dict with keys: os, python_version, packages, entry_command, test_runner, …

        TODO (Phase 1): implement via subprocess + LLM summarisation call
                        (TaskType.SYSTEM_INFO → llama-3.1-8b-instant).
        """
        raise NotImplementedError("SystemInfoAgent.gather() — implement in Phase 1")
