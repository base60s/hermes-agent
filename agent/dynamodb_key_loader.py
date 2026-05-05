"""Startup-time loader that pulls LLM provider API keys from DynamoDB.

On first call, ``apply_dynamodb_overrides`` reads each provider in
``hermes_cli.auth.PROVIDER_REGISTRY`` from the ``chroma-llm-keys``
DynamoDB table and writes the result into the provider's primary env var
(e.g. ``OPENROUTER_API_KEY``). DynamoDB wins over any pre-existing env
value; on miss or any AWS error, the env var is left as-is.

Behaviour is opt-out: set ``HERMES_DYNAMODB_KEY_DISABLED=1`` or omit
``AWS_ACCESS_KEY_ID`` to skip the loader entirely. The function never
raises; failures are logged and ignored so agent startup is not coupled
to DynamoDB availability.

See ``docs/superpowers/specs/2026-05-05-dynamodb-llm-key-loader-design.md``.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "chroma-llm-keys"
DEFAULT_REGION = "us-east-1"
_TRUTHY = {"1", "true", "yes", "on"}

# Providers that Hermes treats specially and so are not present in
# hermes_cli.auth.PROVIDER_REGISTRY but still need to be sourced from
# DynamoDB. OpenRouter is the default routing layer (see
# agent/auxiliary_client.py:_try_openrouter) and is the primary key the
# Hermeregildo Telegram bot needs.
_EXTRA_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("openrouter", "OPENROUTER_API_KEY"),
)

_applied = False


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _should_run() -> bool:
    """Return True iff the loader should perform its DynamoDB work."""
    if _applied:
        return False
    if _is_truthy(os.environ.get("HERMES_DYNAMODB_KEY_DISABLED")):
        return False
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return False
    return True


def _build_client():
    """Construct a boto3 DynamoDB client. Returns None on failure."""
    try:
        import boto3  # type: ignore
    except ImportError:
        logger.warning(
            "dynamodb_key_loader: boto3 not installed; skipping DynamoDB key load"
        )
        return None

    region = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION"
    ) or DEFAULT_REGION

    try:
        return boto3.client("dynamodb", region_name=region)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "dynamodb_key_loader: failed to construct boto3 client: %s",
            type(exc).__name__,
        )
        return None


def _fetch_provider_key(
    client, table: str, provider_id: str
) -> Optional[str]:
    """Fetch a single provider's API key from DynamoDB.

    Returns the string value, or None if the row is absent, the value
    attribute is missing/empty, or any AWS error occurs. Never raises.
    """
    try:
        response = client.get_item(
            TableName=table,
            Key={"key": {"S": provider_id}},
        )
    except Exception as exc:
        logger.warning(
            "dynamodb_key_loader: GetItem failed for provider=%s: %s",
            provider_id,
            type(exc).__name__,
        )
        return None

    item = response.get("Item")
    if not item:
        logger.debug(
            "dynamodb_key_loader: no row for provider=%s", provider_id
        )
        return None

    value_attr = item.get("value") or {}
    value = value_attr.get("S") or ""
    if not value:
        logger.debug(
            "dynamodb_key_loader: empty value for provider=%s", provider_id
        )
        return None

    return value


def _provider_targets() -> Iterable[Tuple[str, str]]:
    """Yield (provider_id, primary_env_var) for every provider whose key
    we want to source from DynamoDB.

    Sources, in order: ``_EXTRA_TARGETS`` (special providers Hermes does
    not register, e.g. OpenRouter) followed by every provider in
    ``hermes_cli.auth.PROVIDER_REGISTRY`` that has at least one
    ``api_key_env_var`` configured. Provider ids are lowercased to match
    the DynamoDB key convention. Duplicates are filtered so each
    provider id is yielded at most once.
    """
    seen: set[str] = set()

    for provider_id, primary in _EXTRA_TARGETS:
        pid = provider_id.lower()
        if pid in seen or not primary:
            continue
        seen.add(pid)
        yield pid, primary

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception as exc:
        logger.warning(
            "dynamodb_key_loader: cannot import PROVIDER_REGISTRY: %s",
            type(exc).__name__,
        )
        return

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        env_vars = getattr(pconfig, "api_key_env_vars", ()) or ()
        if not env_vars:
            continue
        primary = env_vars[0]
        if not primary:
            continue
        pid = provider_id.lower()
        if pid in seen:
            continue
        seen.add(pid)
        yield pid, primary


def apply_dynamodb_overrides() -> None:
    """Read provider API keys from DynamoDB and write them into os.environ.

    Idempotent: subsequent calls in the same process are no-ops.
    Never raises.
    """
    global _applied
    if not _should_run():
        return

    client = _build_client()
    if client is None:
        _applied = True
        return

    table = os.environ.get("HERMES_DYNAMODB_KEY_TABLE") or DEFAULT_TABLE

    applied_count = 0
    total_count = 0
    for provider_id, primary_env_var in _provider_targets():
        total_count += 1
        value = _fetch_provider_key(client, table, provider_id)
        if value is None:
            continue
        os.environ[primary_env_var] = value
        applied_count += 1

    logger.info(
        "dynamodb_key_loader: applied %d/%d provider keys from %s",
        applied_count,
        total_count,
        table,
    )
    _applied = True
