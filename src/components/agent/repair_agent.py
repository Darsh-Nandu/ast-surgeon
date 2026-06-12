"""
RepairAgent — applies targeted LLM-driven fixes to files that failed checks.

Phase 2 will implement:
  - Receives a RepairRequest wrapping a StaticCheckResult or PipelineCheckResult
  - Calls LLM (TaskType.REPAIR → llama-3.3-70b-versatile) with the error context
    and original file content to produce a corrected file
  - Writes the corrected file via file_tools
  - Supports retry loop (up to N attempts) with escalating context
  - Sets RepairRequest.give_up = True after max attempts
  - Integrates with the permission_callback gate (Phase 2)

TODO (Phase 2): implement RepairAgent.repair()
"""

from __future__ import annotations

# TODO (Phase 2): from .models import RepairRequest, AgentResult
# TODO (Phase 2): from .router import ModelRouter, TaskType


class RepairAgent:
    """Stub — to be implemented in Phase 2."""

    def repair(self, request: "RepairRequest") -> str:  # type: ignore[name-defined]
        """Attempt to fix the file described in *request*.

        Args:
            request: RepairRequest describing the check failure.

        Returns:
            The repaired file content as a string.

        TODO (Phase 2): implement multi-attempt LLM repair loop.
        """
        raise NotImplementedError("RepairAgent.repair() — implement in Phase 2")
