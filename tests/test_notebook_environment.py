"""The %pip lesson must work in both locked notebook installation paths."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_pip_is_a_notebook_dependency_with_no_added_runtime_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    workshop = packages[project["name"]]
    assert any(re.match(r"pip[=<>]", dependency) for dependency in project["optional-dependencies"]["notebook"])
    assert {"name": "pip"} in workshop["optional-dependencies"]["notebook"]
    assert {"name": "pip"} not in workshop["dependencies"]
    assert {"name": "pip"} not in workshop["optional-dependencies"]["langfuse"]
    assert not packages["pip"].get("dependencies")


@pytest.mark.parametrize("extra", ["notebook", "langfuse", "runtime", "runtime-langfuse"])
def test_pip_is_hash_pinned_in_only_the_notebook_exports(extra):
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    pip = next(package for package in packages if package["name"] == "pip")
    exported = (ROOT / f"requirements-{extra}.lock").read_text()
    block = re.search(r"^pip==[^\n]*(?:\n[ \t]+[^\n]*)*", exported, re.MULTILINE)
    if extra.startswith("runtime"):
        assert block is None
    else:
        assert block is not None
        assert block[0].split()[0] == f"pip=={pip['version']}"
        assert set(re.findall(r"--hash=(\S+)", block[0])) == {
            artifact["hash"] for artifact in [pip["sdist"], *pip["wheels"]]
        }
