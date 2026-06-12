"""
GitManagerAgent — sole holder of GitHub/git tools; generates commit messages
and manages repository operations.

Phase 6 will implement:
  - Commit message generation via LLM (TaskType.GIT_OPS → llama-3.1-8b-instant)
  - git add / commit / push orchestration using file_tools + subprocess
  - Branch management helpers (create, checkout, merge)
  - Populates AgentResult.git_summary dict on completion
  - Enforces the "sole holder" invariant: no other agent calls git directly

TODO (Phase 6): implement GitManagerAgent.commit(), .push(), .branch_ops()
"""

from __future__ import annotations

# TODO (Phase 6): from .models import TaskType, AgentResult
# TODO (Phase 6): from .router import ModelRouter


class GitManagerAgent:
    """Stub — to be implemented in Phase 6."""

    def commit(self, files: list[str], message: str = "") -> dict:
        """Stage *files* and create a commit, generating a message if not given.

        Args:
            files:   List of file paths to stage.
            message: Commit message; if empty, an LLM call generates one.

        Returns:
            dict with keys: commit_hash, message, files_committed.

        TODO (Phase 6): implement git staging + LLM commit-message generation
                        + subprocess git commit.
        """
        raise NotImplementedError("GitManagerAgent.commit() — implement in Phase 6")

    def push(self, remote: str = "origin", branch: str = "main") -> dict:
        """Push committed changes to the remote.

        TODO (Phase 6): implement subprocess git push.
        """
        raise NotImplementedError("GitManagerAgent.push() — implement in Phase 6")
