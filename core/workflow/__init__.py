from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "WorkflowService",
    "build_default_workflow_service",
]

if TYPE_CHECKING:
    from .factory import build_default_workflow_service
    from .service import WorkflowService


def __getattr__(name: str):
    if name == "WorkflowService":
        from .service import WorkflowService

        return WorkflowService
    if name == "build_default_workflow_service":
        from .factory import build_default_workflow_service

        return build_default_workflow_service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
