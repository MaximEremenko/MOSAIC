from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config.values import (
    first_present,
    resolve_path_from,
    unique_paths,
)


def load_run_settings(run_file: str, *, repo_root: Path) -> tuple[Path, dict[str, Any]]:
    run_path = Path(run_file)
    if run_path.is_absolute():
        candidates = [run_path]
    else:
        candidates = [
            Path.cwd() / run_path,
            repo_root / run_path,
            repo_root / "examples" / run_path,
        ]

    resolved_candidates = unique_paths(candidates)
    for candidate in resolved_candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return candidate, json.load(handle) or {}
    return resolved_candidates[0], {}


def resolve_input_parameters_path(
    run_path: Path,
    run_settings_payload: dict[str, Any],
) -> tuple[Path, Path | None]:
    raw_input = first_present(
        run_settings_payload,
        (
            "input_parameters_path",
            "input_parameters",
            "input_parameters_json",
            "input_json",
        ),
    )
    if raw_input is not None:
        candidate = resolve_path_from(run_path.parent, raw_input)
        if candidate.is_dir():
            candidate = candidate / "input_parameters.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Configured input parameters file '{candidate}' does not exist."
            )
        return candidate, None

    root_path = None
    raw_root = first_present(
        run_settings_payload, ("working_path", "config_dir", "config_path")
    )
    if raw_root is not None:
        candidate = resolve_path_from(run_path.parent, raw_root)
        root_path = candidate if candidate.is_dir() else candidate.parent
        if candidate.is_dir():
            candidate = candidate / "input_parameters.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Configured working path '{candidate}' does not contain input_parameters.json."
            )
        return candidate, root_path

    example_roots = unique_paths(
        [
            Path.cwd() / "examples",
            run_path.parent / "examples",
            run_path.parent.parent / "examples",
        ]
    )
    test_roots = unique_paths(
        [
            Path.cwd() / "tests",
            run_path.parent / "tests",
            run_path.parent.parent / "tests",
        ]
    )

    def _find_unique(roots: list[Path]) -> list[Path]:
        found: list[Path] = []
        for root in roots:
            if root.exists():
                found.extend(root.rglob("input_parameters.json"))
        return unique_paths(found)

    found = _find_unique(example_roots)
    if len(found) == 1:
        return found[0], found[0].parent
    if len(found) > 1:
        found_msg = "\n".join(f" - {path}" for path in found)
        raise FileNotFoundError(
            "Multiple input_parameters.json files found under examples. Set 'input_parameters_path' in run_parameters.json.\n"
            f"{found_msg}"
        )

    found = _find_unique(test_roots)
    if len(found) == 1:
        return found[0], found[0].parent
    if not found:
        raise FileNotFoundError(
            "No input_parameters.json found. Set 'input_parameters_path' in run_parameters.json."
        )

    found_msg = "\n".join(f" - {path}" for path in found)
    raise FileNotFoundError(
        "Multiple input_parameters.json files found under tests. Set 'input_parameters_path' in run_parameters.json.\n"
        f"{found_msg}"
    )


def resolve_config_root(
    parameters: dict[str, Any],
    input_parameters_path: Path,
    root_path: Path | None = None,
) -> Path:
    json_dir = input_parameters_path.parent.resolve()
    paths = parameters.get("paths", {}) if isinstance(parameters, dict) else {}
    struct = parameters.get("structInfo", {}) if isinstance(parameters, dict) else {}

    explicit_root = first_present(
        paths,
        (
            "config_root",
            "configRoot",
            "base_directory",
            "baseDirectory",
            "data_root",
            "dataRoot",
        ),
    ) or first_present(
        struct,
        (
            "config_root",
            "configRoot",
            "base_directory",
            "baseDirectory",
            "data_root",
            "dataRoot",
        ),
    ) or first_present(
        parameters,
        (
            "config_root",
            "configRoot",
            "base_directory",
            "baseDirectory",
            "data_root",
            "dataRoot",
        ),
    )

    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(resolve_path_from(json_dir, explicit_root))
    if root_path is not None:
        candidates.append(root_path.resolve())
    candidates.extend((json_dir, json_dir.parent, Path.cwd().resolve()))
    candidates = unique_paths(candidates)

    cfg_name = first_present(
        paths,
        (
            "structure_file",
            "structureFile",
            "average_structure_file",
            "averageStructureFile",
        ),
    ) or first_present(struct, ("filename", "filename_av"))
    if isinstance(cfg_name, str) and cfg_name.strip():
        cfg_path = Path(cfg_name)
        if cfg_path.is_absolute():
            return cfg_path.parent.resolve()
        for base in candidates:
            if (base / cfg_path).exists():
                return base

    if explicit_root is not None:
        return candidates[0]
    if root_path is not None:
        return root_path.resolve()
    return json_dir
