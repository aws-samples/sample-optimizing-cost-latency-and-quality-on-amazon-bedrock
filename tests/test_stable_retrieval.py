from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["managed", "vector"])
@pytest.mark.parametrize("implementation", ["local", "gateway"])
def test_managed_ranking_not_filtered_by_vector_score(kind, implementation):
    if implementation == "local":
        source = (ROOT / "03-developer-journey/utils/tools.py").read_text()
    else:
        notebook = json.loads((ROOT / "03-developer-journey/06-skills-and-gateway.ipynb").read_text())
        cell = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code" and "lambda_code =" in "".join(c["source"])
        )
        source = ast.literal_eval(
            next(
                n.value
                for n in ast.parse(cell).body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "lambda_code" for t in n.targets)
            )
        )
    fn = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "get_technical_support"
    )
    fn.decorator_list = []
    requests = []

    def retrieve(**kwargs):
        requests.append(kwargs)
        return {
            "retrievalResults": [
                {"score": 0.3, "content": {"text": "Relevant lower-score passage"}},
                {"score": 0.7, "content": {"text": "Higher-score passage"}},
            ]
        }

    clients = {
        "ssm": SimpleNamespace(
            get_parameter=lambda **kwargs: {
                "Parameter": {"Value": kind if kwargs["Name"].endswith("/type") else "kb-test"}
            }
        ),
        "sts": SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"}),
        "bedrock-agent-runtime": SimpleNamespace(retrieve=retrieve),
    }
    fake = SimpleNamespace(
        client=lambda name, **kwargs: clients[name], Session=lambda: SimpleNamespace(region_name="us-east-1")
    )
    ns = {"boto3": fake}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "retrieval", "exec"), ns)
    result = ns["get_technical_support"]("original customer question")
    assert ("Relevant lower-score passage" in result) == (kind == "managed")
    assert "Higher-score passage" in result
    assert requests[0]["retrievalQuery"] == {"text": "original customer question"}
    assert requests[0]["retrievalConfiguration"] == {
        ("managedSearchConfiguration" if kind == "managed" else "vectorSearchConfiguration"): {"numberOfResults": 3}
    }
