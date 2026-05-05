"""Startup-time loader that pulls LLM provider API keys from DynamoDB.

The ``chroma-llm-keys`` table is a harvested-key pool keyed by the API
key value itself, with attributes ``provider`` (capitalized, e.g.
``Anthropic`` / ``Gemini`` / ``Groq``), ``validation_status`` (``valid``
when usable), ``last_validated_at``, and ``accessible_models``.

On first call, ``apply_dynamodb_overrides`` performs a single ``Scan`` of
that table, groups valid keys by provider, and writes the first valid
key per provider into the matching env var (e.g.
``ANTHROPIC_API_KEY``). DynamoDB wins over any pre-existing env value;
when no valid key exists for a provider, the env var is left as-is so
operator-supplied env vars (or the OpenRouter fallback) keep working.

Behaviour is opt-out: set ``HERMES_DYNAMODB_KEY_DISABLED=1`` or omit
``AWS_ACCESS_KEY_ID`` to skip the loader entirely. The function never
raises; failures are logged and ignored so agent startup is not coupled
to DynamoDB availability.

See ``docs/superpowers/specs/2026-05-05-dynamodb-llm-key-loader-design.md``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "chroma-llm-keys"
DEFAULT_REGION = "us-east-1"
_TRUTHY = {"1", "true", "yes", "on"}

# Mapping from lowercase Hermes provider id (as used in
# hermes_cli.auth.PROVIDER_REGISTRY) to the exact ``provider`` attribute
# value used in the chroma-llm-keys table. Add entries here as new
# providers start populating the pool.
_TABLE_PROVIDER_NAMES: Dict[str, str] = {
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "minimax": "Minimax",
    "xai": "xAI",
}

# Providers that Hermes either treats specially (OpenRouter — the
# default routing layer; see agent/auxiliary_client.py:_try_openrouter)
# or doesn't register at all but which still exist as table providers.
# Each tuple is (hermes_id, primary_env_var, table_provider_name).
_EXTRA_TARGETS: Tuple[Tuple[str, str, str], ...] = (
    ("openrouter", "OPENROUTER_API_KEY", "OpenRouter"),
    ("groq", "GROQ_API_KEY", "Groq"),
    ("openai", "OPENAI_API_KEY", "OpenAI"),
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


def _scan_valid_keys_by_provider(
    client, table: str
) -> Dict[str, List[str]]:
    """Scan the table once and return {provider_name: [valid_key, ...]}.

    Only items with ``validation_status == 'valid'`` are included.
    Pagination is handled via ``LastEvaluatedKey``. On any AWS error the
    function returns whatever it has accumulated so far (possibly empty)
    after logging a warning.
    """
    by_provider: Dict[str, List[str]] = {}
    last_key: Optional[Dict] = None

    while True:
        kwargs = {
            "TableName": table,
            "ProjectionExpression": "#k, #p, validation_status",
            "ExpressionAttributeNames": {"#k": "key", "#p": "provider"},
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key

        try:
            response = client.scan(**kwargs)
        except Exception as exc:
            logger.warning(
                "dynamodb_key_loader: Scan failed (partial=%d providers): %s",
                len(by_provider),
                type(exc).__name__,
            )
            return by_provider

        for item in response.get("Items", []):
            provider = (item.get("provider") or {}).get("S")
            status = (item.get("validation_status") or {}).get("S")
            key = (item.get("key") or {}).get("S")
            if not provider or not key or status != "valid":
                continue
            by_provider.setdefault(provider, []).append(key)

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break

    return by_provider


def _provider_targets() -> Iterable[Tuple[str, str, str]]:
    """Yield (hermes_id, primary_env_var, table_provider_name) for every
    provider whose key we want to source from DynamoDB.

    Order is: ``_EXTRA_TARGETS`` first (OpenRouter, Groq, OpenAI — not
    in PROVIDER_REGISTRY), then every entry in
    ``hermes_cli.auth.PROVIDER_REGISTRY`` whose lowercase id is a key
    in ``_TABLE_PROVIDER_NAMES``. Duplicates are filtered so each
    Hermes id is yielded at most once.
    """
    seen: set[str] = set()

    for hermes_id, env_var, table_name in _EXTRA_TARGETS:
        hid = hermes_id.lower()
        if hid in seen or not env_var or not table_name:
            continue
        seen.add(hid)
        yield hid, env_var, table_name

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception as exc:
        logger.warning(
            "dynamodb_key_loader: cannot import PROVIDER_REGISTRY: %s",
            type(exc).__name__,
        )
        return

    for hermes_id, pconfig in PROVIDER_REGISTRY.items():
        hid = hermes_id.lower()
        if hid in seen:
            continue
        table_name = _TABLE_PROVIDER_NAMES.get(hid)
        if not table_name:
            continue
        env_vars = getattr(pconfig, "api_key_env_vars", ()) or ()
        if not env_vars:
            continue
        primary = env_vars[0]
        if not primary:
            continue
        seen.add(hid)
        yield hid, primary, table_name


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

    keys_by_provider = _scan_valid_keys_by_provider(client, table)

    applied_count = 0
    total_count = 0
    for hermes_id, primary_env_var, table_provider_name in _provider_targets():
        total_count += 1
        candidates = keys_by_provider.get(table_provider_name) or []
        if not candidates:
            logger.debug(
                "dynamodb_key_loader: no valid key for provider=%s "
                "(table_name=%s)",
                hermes_id,
                table_provider_name,
            )
            continue
        os.environ[primary_env_var] = candidates[0]
        applied_count += 1

    summary = (
        f"dynamodb_key_loader: applied {applied_count}/{total_count} "
        f"provider keys from {table} (scanned {sum(len(v) for v in keys_by_provider.values())} valid rows across "
        f"{len(keys_by_provider)} providers)"
    )
    logger.info(summary)
    # Also print to stderr so the summary is visible during early startup
    # before any logging handler is configured.
    print(summary, file=sys.stderr, flush=True)
    _applied = True
