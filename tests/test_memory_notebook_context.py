"""Memory recall supplies retrieved text, not SDK record metadata, to the model."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_memory_context_accepts_sdk_datetime_fields_without_leaking_metadata():
    notebook = json.loads((Path(__file__).parents[1] / '02-optimization-playbook/02-medium-effort.ipynb').read_text())
    cell = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code'
                and 'retrieve_memory_records' in ''.join(cell['source']))
    facts = [{'memoryRecordId': 'service-id', 'createdAt': datetime(2026, 9, 25, tzinfo=UTC),
              'content': {'text': 'Order A-12345; preferred contact is email.'},
              'metadata': {'updatedAt': {'dateTimeValue': datetime(2026, 9, 25, tzinfo=UTC)}}}]
    run_case = Mock(return_value={'text': 'A-12345, email'})
    namespace = {'MEMORY_ID': 'owned-memory', 'MEMORY_NAMESPACE': '/users/fixture/facts',
                 'AGENTCORE': SimpleNamespace(retrieve_memory_records=Mock(return_value={'memoryRecordSummaries': facts})),
                 'os': SimpleNamespace(environ={}), 'time': SimpleNamespace(monotonic=lambda: 0, sleep=Mock()),
                 'json': json, 'SMALL': 'model-fixture', 'run_case': run_case, 'show': Mock()}
    exec(compile(''.join(cell['source']), '<memory-recall-cell>', 'exec'), namespace)
    context = run_case.call_args.kwargs['system'][0]['text']
    assert 'Order A-12345; preferred contact is email.' in context
    assert 'service-id' not in context
    assert 'createdAt' not in context and 'updatedAt' not in context
    assert 'transcript' not in context
