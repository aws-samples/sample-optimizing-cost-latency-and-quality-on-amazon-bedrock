"""Text-only SDK4/server3 response adapter, for the provisioned Langfuse server."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import httpx
from langfuse.api.commons.types.dataset_item import DatasetItem
from langfuse.api.core.pydantic_utilities import parse_obj_as
from langfuse.api.dataset_items.types.paginated_dataset_items import PaginatedDatasetItems

_MEDIA_KEYS = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "audio_url",
    "input_audio",
    "video",
    "video_url",
    "file",
    "file_url",
    "file_id",
    "file_data",
    "media",
    "media_id",
    "mediaid",
    "mime_type",
    "mimetype",
    "media_type",
    "attachments",
    "document",
    "mediareferences",
    "media_references",
}

_MEDIA_TYPES = _MEDIA_KEYS | {"imageurl", "audioinput", "document"}


class TextDatasetCompatibilityError(ValueError):
    """The response cannot safely be treated as a text-only v3 dataset."""


def _validate_text_json(value: Any) -> None:
    if isinstance(value, str):
        if "@@@langfuseMedia:" in value or value.lstrip().lower().startswith("data:"):
            raise TextDatasetCompatibilityError("Media references/data URIs are unsupported by the text-only adapter")
    elif isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in _MEDIA_KEYS or (
                key.lower() == "type" and isinstance(child, str) and child.lower() in _MEDIA_TYPES
            ):
                raise TextDatasetCompatibilityError("Media content blocks are unsupported by the text-only adapter")
            _validate_text_json(child)
    elif isinstance(value, list):
        for child in value:
            _validate_text_json(child)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise TextDatasetCompatibilityError("Dataset content must be text-only JSON")
    elif isinstance(value, float) and not math.isfinite(value):
        raise TextDatasetCompatibilityError("Non-finite dataset values are unsupported")


def _validate_item_content(item: dict[str, Any]) -> None:
    for field in ("mediaReferences", "media_references"):
        if field in item and item[field] != []:
            raise TextDatasetCompatibilityError("Only an empty mediaReferences list is supported")
    for field in ("input", "expectedOutput", "expected_output", "metadata"):
        if field in item:
            _validate_text_json(item[field])


def adapt_text_dataset_response(payload: Any, *, collection: bool) -> dict[str, Any]:
    """Add only the missing v3 field; preserve and validate every other field."""
    if not isinstance(payload, dict):
        raise TextDatasetCompatibilityError("Expected a dataset-item JSON object")
    adapted = copy.deepcopy(payload)
    items = adapted.get("data") if collection else [adapted]
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise TextDatasetCompatibilityError("Expected a dataset-item list")
    for item in items:
        _validate_item_content(item)
        item.setdefault("mediaReferences", [])
    schema = PaginatedDatasetItems if collection else DatasetItem
    # Use the SDK's REST decoder: its aliases live in Fern annotation metadata,
    # not Pydantic Field aliases. All normal SDK validation still runs.
    parse_obj_as(type_=schema, object_=adapted)
    return adapted


class TextDatasetTransport(httpx.BaseTransport):
    """Adapt only successful dataset-item responses from one configured host."""

    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("LANGFUSE_BASE_URL must be an HTTP(S) base URL without credentials/query/fragment")
        self._origin = (url.scheme, url.host, url.port)
        self._items_path = url.path.rstrip("/") + "/api/public/dataset-items"
        self._transport = transport if transport is not None else httpx.HTTPTransport(retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        suffix = path.removeprefix(self._items_path + "/")
        is_item = path.startswith(self._items_path + "/") and bool(suffix) and "/" not in suffix
        scoped = (
            (request.url.scheme, request.url.host, request.url.port) == self._origin
            and (path == self._items_path or is_item)
            and request.method in {"GET", "POST"}
        )
        if scoped and request.method == "POST":
            body = json.loads(request.read())
            if not isinstance(body, dict):
                raise TextDatasetCompatibilityError("Expected a dataset-item request object")
            _validate_item_content(body)  # Reject media before sending a write.
        response = self._transport.handle_request(request)
        if not scoped or not response.is_success:
            return response
        try:
            response.read()
            payload = adapt_text_dataset_response(
                response.json(),
                collection=request.method == "GET" and path == self._items_path,
            )
            headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower() not in {"content-length", "content-encoding"}
            ]
            return httpx.Response(response.status_code, headers=headers, json=payload, request=request)
        finally:
            response.close()

    def close(self) -> None:
        self._transport.close()
