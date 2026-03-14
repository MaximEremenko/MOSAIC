from __future__ import annotations

import json
from pathlib import Path

from core.application.configuration.run_files import (
    load_run_settings,
    resolve_config_root,
    resolve_input_parameters_path,
)
from core.application.configuration.runtime import (
    apply_runtime_settings,
    resolve_form_factor_settings,
    resolve_runtime_settings,
)
from core.application.configuration.schema import (
    normalize_input_schema,
    normalize_parameter_paths,
)
from core.domain.models import (
    FormFactorSelection,
    RunSettings,
    RuntimeSettings,
    WorkflowParameters,
)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class ParameterLoadingService:
    def load(
        self, run_file: str = "run_parameters.json"
    ) -> tuple[RunSettings, WorkflowParameters]:
        repo_root = resolve_repo_root()
        run_path, run_settings_payload = load_run_settings(run_file, repo_root=repo_root)
        input_parameters_path, root_path = resolve_input_parameters_path(
            run_path, run_settings_payload
        )
        with input_parameters_path.open("r", encoding="utf-8") as handle:
            raw_parameters = json.load(handle)

        runtime = resolve_runtime_settings(raw_parameters)
        config_root = resolve_config_root(
            raw_parameters,
            input_parameters_path,
            root_path=root_path,
        )
        normalized = normalize_input_schema(raw_parameters)
        normalized = normalize_parameter_paths(
            normalized, config_root, input_parameters_path
        )
        run_settings = RunSettings(
            run_parameters_path=run_path,
            input_parameters_path=input_parameters_path.resolve(),
            config_root=config_root.resolve(),
            working_path=config_root.resolve(),
            runtime=runtime,
            root_path=root_path.resolve() if root_path is not None else None,
        )
        return run_settings, WorkflowParameters.from_payload(normalized)

    def apply_runtime_settings(self, runtime_settings: RuntimeSettings) -> None:
        apply_runtime_settings(runtime_settings)

    def resolve_form_factor_settings(
        self, workflow_parameters: WorkflowParameters
    ) -> FormFactorSelection:
        return resolve_form_factor_settings(workflow_parameters)
