import ast
from pathlib import Path


def test_core_has_no_runtime_print_calls():
    offenders = []
    for path in Path("core").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
