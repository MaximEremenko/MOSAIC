from pathlib import Path


OLD_IMPORT_PREFIXES = (
    "interfaces.",
    "processors.",
    "readers.",
    "factories.",
    "utilities.",
    "data_storage.",
    "data_structures.",
    "functions.",
    "managers.",
    "strategies.",
    "form_factors.",
    "parsers.",
    "calculators.",
)
FORBIDDEN_ARCHITECTURE_PREFIXES = (
    "core.application",
    "core.domain",
    "core.infrastructure",
)


def test_core_modules_use_only_package_qualified_imports():
    offenders = []
    for path in Path("core").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue
            if any(stripped.startswith(f"from {prefix}") or stripped.startswith(f"import {prefix}") for prefix in OLD_IMPORT_PREFIXES):
                offenders.append(f"{path}:{line_no}: {stripped}")
    assert offenders == []


def test_repo_python_modules_do_not_import_architecture_first_packages():
    offenders = []
    for root in (Path("core"), Path("tests")):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if not stripped.startswith(("from ", "import ")):
                    continue
                if any(
                    stripped.startswith(f"from {prefix}") or stripped.startswith(f"import {prefix}")
                    for prefix in FORBIDDEN_ARCHITECTURE_PREFIXES
                ):
                    offenders.append(f"{path}:{line_no}: {stripped}")
    assert offenders == []
