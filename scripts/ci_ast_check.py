from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_APPLY_MIRROR_NAME = "apply_half_space_conjugate_mirror"
_OLD_ARTIFACT_NAMES = {
    "HALF_SPACE_ROLE_LEGACY",
    "build_legacy_residual_field_output_artifacts",
}
_CURRENT_RUN_FALLBACK_NAMES = {
    "reduce_scattering_shards_for_chunk",
}
_WEAK_HASH_NAMES = {"md5", "sha1"}
_HASHLIB_MODULE = "hashlib"
_LEGACY_FORMAT_CHECKPOINT_TOKENS = {
    "scattering_shards",
    "residual_shards",
}
_PUBLIC_COMPATIBILITY_EXACT_TOKENS = {
    "processed_point_data",
    "public_manifest.json",
}
_PUBLIC_COMPATIBILITY_CHUNK_PREFIXES = {
    "point_data_chunk_",
    "residual_chunk_",
}
_WORKER_STAGE_FILENAMES = {
    "tasks.py",
    "execution.py",
    "chunk_processing.py",
}
_WORKER_STAGE_PARTS = {
    "decoding",
    "residual_field",
    "scattering",
}


def _is_worker_stage_path(path: Path) -> bool:
    parts = set(path.parts)
    return path.name in _WORKER_STAGE_FILENAMES and bool(parts & _WORKER_STAGE_PARTS)


def _is_core_path(path: Path) -> bool:
    return "core" in path.parts


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _contains_q_grid_third_coordinate(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        value_name = _name_of(child.value)
        if value_name != "q_grid":
            continue
        slice_node = child.slice
        if isinstance(slice_node, ast.Tuple):
            elements = slice_node.elts
            if len(elements) >= 2 and isinstance(elements[1], ast.Constant):
                if elements[1].value == 2:
                    return True
    return False


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_target_names(element))
        return names
    name = _name_of(node)
    return [name] if name is not None else []


def _is_half_space_metadata_target(name: str) -> bool:
    normalized = name.lower()
    return "half_space_role" in normalized or normalized in {"role", "metadata_role"}


def _is_public_compatibility_literal(value: str) -> bool:
    if any(token in value for token in _PUBLIC_COMPATIBILITY_EXACT_TOKENS):
        return True
    return any(
        value == prefix
        or value.startswith(prefix)
        or f"/{prefix}" in value
        or f"\\{prefix}" in value
        for prefix in _PUBLIC_COMPATIBILITY_CHUNK_PREFIXES
    )


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    column: int
    code: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code}: {self.message}"


def _iter_python_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                child
                for child in path.rglob("*.py")
                if "__pycache__" not in child.parts
            )
        elif path.suffix == ".py":
            files.append(path)
    return sorted(set(files))


def _is_zero_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in {0, 0.0, 0j}


def _check_ast(path: Path, tree: ast.AST) -> list[Violation]:
    violations: list[Violation] = []
    mirror_aliases = {_APPLY_MIRROR_NAME}
    hashlib_aliases = {_HASHLIB_MODULE}
    weak_hash_aliases: set[str] = set()
    old_artifact_aliases = set(_OLD_ARTIFACT_NAMES)
    current_run_fallback_aliases = set(_CURRENT_RUN_FALLBACK_NAMES)
    check_old_codebase_artifacts = _is_core_path(path)
    check_current_run_fallbacks = _is_worker_stage_path(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _HASHLIB_MODULE:
                    hashlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == _APPLY_MIRROR_NAME:
                    mirror_aliases.add(alias.asname or alias.name)
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            node.col_offset,
                            "forbidden-conjugate-mirror",
                            "Do not import apply_half_space_conjugate_mirror.",
                        )
                    )
                if node.module == _HASHLIB_MODULE and alias.name in _WEAK_HASH_NAMES:
                    weak_hash_aliases.add(alias.asname or alias.name)
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            node.col_offset,
                            "weak-hash",
                            f"Use SHA-256 for durable digests, not hashlib.{alias.name}.",
                        )
                    )
                if alias.name in _OLD_ARTIFACT_NAMES:
                    old_artifact_aliases.add(alias.asname or alias.name)
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            node.col_offset,
                            "old-intermediate-fallback",
                            "Do not import old intermediate-artifact compatibility helpers.",
                        )
                    )
                if alias.name in _CURRENT_RUN_FALLBACK_NAMES:
                    current_run_fallback_aliases.add(alias.asname or alias.name)
                    if check_old_codebase_artifacts:
                        violations.append(
                            Violation(
                                path,
                                node.lineno,
                                node.col_offset,
                                "old-codebase-artifact-fallback",
                                "Do not use old-codebase artifact/checkpoint fallback helpers in current-run code.",
                            )
                        )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if check_old_codebase_artifacts and node.name in _CURRENT_RUN_FALLBACK_NAMES:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "old-codebase-artifact-fallback",
                        "Do not define old-codebase artifact/checkpoint fallback entry points in current-run code.",
                    )
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in mirror_aliases:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "forbidden-conjugate-mirror",
                        "Do not call apply_half_space_conjugate_mirror or an alias.",
                    )
                )
            if isinstance(func, ast.Name) and func.id in weak_hash_aliases:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "weak-hash",
                        "Use SHA-256 for durable digests, not weak hashlib aliases.",
                    )
                )
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in hashlib_aliases
                and func.attr in _WEAK_HASH_NAMES
            ):
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "weak-hash",
                        f"Use SHA-256 for durable digests, not hashlib.{func.attr}.",
                    )
                )
            if isinstance(func, ast.Name) and func.id in old_artifact_aliases:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "old-intermediate-fallback",
                        "Do not use old intermediate-artifact compatibility helpers.",
                    )
                )
            if (
                check_old_codebase_artifacts
                and isinstance(func, ast.Name)
                and func.id in current_run_fallback_aliases
            ):
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "old-codebase-artifact-fallback",
                        "Do not use old-codebase artifact/checkpoint fallback helpers in current-run code.",
                    )
                )
            if (
                check_old_codebase_artifacts
                and isinstance(func, ast.Attribute)
                and func.attr in _CURRENT_RUN_FALLBACK_NAMES
            ):
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "old-codebase-artifact-fallback",
                        "Do not use old-codebase artifact/checkpoint fallback helpers in current-run code.",
                    )
                )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "imag" and _is_zero_constant(node.value):
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            node.col_offset,
                            "blanket-imag-zeroing",
                            "Do not zero imaginary components to hide RIFFT errors.",
                        )
                    )
                for target_name in _target_names(target):
                    if (
                        _is_half_space_metadata_target(target_name)
                        and _contains_q_grid_third_coordinate(node.value)
                    ):
                        violations.append(
                            Violation(
                                path,
                                node.lineno,
                                node.col_offset,
                                "missing-half-space-metadata",
                                "Do not infer current-run half-space role from q_grid[:, 2].",
                            )
                        )
        elif isinstance(node, ast.AnnAssign):
            for target_name in _target_names(node.target):
                if (
                    node.value is not None
                    and _is_half_space_metadata_target(target_name)
                    and _contains_q_grid_third_coordinate(node.value)
                ):
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            node.col_offset,
                            "missing-half-space-metadata",
                            "Do not infer current-run half-space role from q_grid[:, 2].",
                        )
                    )
        elif isinstance(node, ast.keyword):
            if node.arg == "allow_legacy" and isinstance(node.value, ast.Constant) and node.value.value is True:
                violations.append(
                    Violation(
                        path,
                        node.value.lineno,
                        node.value.col_offset,
                        "current-run-legacy-fallback",
                        "Do not enable allow_legacy=True in current-run code.",
                    )
                )
            if node.arg == "legacy_layout" and isinstance(node.value, ast.Constant) and node.value.value is True:
                violations.append(
                    Violation(
                        path,
                        node.value.lineno,
                        node.value.col_offset,
                        "old-intermediate-fallback",
                        "Do not select old intermediate artifact layouts in current-run code.",
                    )
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if check_old_codebase_artifacts and node.value in _CURRENT_RUN_FALLBACK_NAMES:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "old-codebase-artifact-fallback",
                        "Do not use old-codebase artifact/checkpoint fallback names in current-run code.",
                    )
                )
            if check_old_codebase_artifacts and any(
                token in node.value for token in _LEGACY_FORMAT_CHECKPOINT_TOKENS
            ):
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "old-codebase-artifact-format",
                        "Remove or replace old-codebase artifact/checkpoint formats; rerun under current manifests.",
                    )
                )
            if check_current_run_fallbacks and _is_public_compatibility_literal(node.value):
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        "worker-public-path",
                        "Workers must not write public or canonical compatibility paths.",
                    )
                )
        elif isinstance(node, ast.Name) and node.id in old_artifact_aliases:
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    node.col_offset,
                    "old-intermediate-fallback",
                    "Do not use old intermediate-artifact compatibility helpers.",
                )
            )
        elif check_old_codebase_artifacts and isinstance(node, ast.Name) and node.id in current_run_fallback_aliases:
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    node.col_offset,
                    "old-codebase-artifact-fallback",
                    "Do not use old-codebase artifact/checkpoint fallback helpers in current-run code.",
                )
            )
        elif check_old_codebase_artifacts and isinstance(node, ast.Attribute) and node.attr in _CURRENT_RUN_FALLBACK_NAMES:
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    node.col_offset,
                    "old-codebase-artifact-fallback",
                    "Do not use old-codebase artifact/checkpoint fallback helpers in current-run code.",
                )
            )
    return violations


def _check_regex(path: Path, source: str) -> list[Violation]:
    violations: list[Violation] = []
    patterns = [
        (
            "blanket-imag-zeroing",
            re.compile(r"np\.imag\s*\([^)]*\)\s*\*\s*0"),
            "Do not multiply imaginary components by zero.",
        ),
        (
            "old-intermediate-npz-fallback",
            re.compile(r"with_suffix\s*\(\s*['\"]\.npz['\"]\s*\)\s*\.exists\s*\("),
            "Do not repair missing current-run HDF5 artifacts from sibling .npz files.",
        ),
    ]
    for code, pattern, message in patterns:
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            last_newline = source.rfind("\n", 0, match.start())
            column = match.start() - last_newline - 1
            violations.append(Violation(path, line, column, code, message))
    return violations


def _dedupe_violations(violations: Iterable[Violation]) -> list[Violation]:
    seen: set[tuple[Path, int, int, str]] = set()
    deduped: list[Violation] = []
    for violation in violations:
        key = (violation.path, violation.line, violation.column, violation.code)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(violation)
    return deduped


def check_paths(paths: Iterable[str | Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_python_files(paths):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(
                Violation(path, exc.lineno or 1, exc.offset or 0, "syntax-error", str(exc))
            )
            continue
        violations.extend(_check_ast(path, tree))
        violations.extend(_check_regex(path, source))
    return _dedupe_violations(violations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MOSAIC forbidden-pattern guard rails.")
    parser.add_argument("paths", nargs="*", default=["core"], help="Files or directories to check.")
    args = parser.parse_args(argv)
    violations = check_paths(args.paths)
    for violation in violations:
        print(violation.format(), file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
