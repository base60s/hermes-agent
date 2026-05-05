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

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
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


# Provider probe priority. The loader walks this list in order, calls
# each provider's `/models` endpoint with the env-var-loaded API key,
# and picks the first preferred model that the response advertises. The
# first provider whose probe returns 200 with at least one viable model
# wins, and the choice is written into ``$HERMES_HOME/config.yaml``.
@dataclass(frozen=True)
class _ProviderProbe:
    id: str                      # hermes-internal short name
    env_var: str                 # env var the loader writes the key into
    models_url: str              # GET endpoint that lists models
    auth: str                    # 'bearer' or 'x-api-key'
    hermes_provider: str         # value to put in config.yaml model.provider
    base_url: Optional[str]      # value to put in config.yaml model.base_url
    preferred_models: Tuple[str, ...] = field(default_factory=tuple)


_PROBES: Tuple[_ProviderProbe, ...] = (
    _ProviderProbe(
        id="openai",
        env_var="OPENAI_API_KEY",
        models_url="https://api.openai.com/v1/models",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://api.openai.com/v1",
        preferred_models=("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"),
    ),
    _ProviderProbe(
        id="gemini",
        env_var="GOOGLE_API_KEY",
        models_url="https://generativelanguage.googleapis.com/v1beta/openai/models",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        preferred_models=(
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-preview",
            "gemini-3-flash-preview",
        ),
    ),
    _ProviderProbe(
        id="xai",
        env_var="XAI_API_KEY",
        models_url="https://api.x.ai/v1/models",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://api.x.ai/v1",
        preferred_models=("grok-4-latest", "grok-4", "grok-3-latest", "grok-3"),
    ),
    _ProviderProbe(
        id="anthropic",
        env_var="ANTHROPIC_API_KEY",
        models_url="https://api.anthropic.com/v1/models",
        auth="x-api-key",
        hermes_provider="anthropic",
        base_url=None,
        preferred_models=(
            "claude-opus-4.6",
            "claude-sonnet-4.6",
            "claude-haiku-4.5",
        ),
    ),
    _ProviderProbe(
        id="groq",
        env_var="GROQ_API_KEY",
        models_url="https://api.groq.com/openai/v1/models",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://api.groq.com/openai/v1",
        preferred_models=(
            "llama-4.1-70b-versatile",
            "llama-4-70b",
            "mixtral-8x7b-32768",
        ),
    ),
    _ProviderProbe(
        id="minimax",
        env_var="MINIMAX_API_KEY",
        models_url="https://api.minimax.chat/v1/models",
        auth="bearer",
        hermes_provider="minimax",
        base_url=None,
        preferred_models=("minimax-3.5", "abab6.5s-chat"),
    ),
)


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


def _probe_models_endpoint(probe: _ProviderProbe) -> Optional[List[str]]:
    """GET the probe's models endpoint. Returns the list of advertised
    model ids on success, or None if the env var is unset, the call
    fails, or the response is not parseable. Never raises.
    """
    key = os.environ.get(probe.env_var)
    if not key:
        return None

    req = urllib.request.Request(probe.models_url)
    if probe.auth == "bearer":
        req.add_header("Authorization", f"Bearer {key}")
    elif probe.auth == "x-api-key":
        req.add_header("x-api-key", key)
        req.add_header("anthropic-version", "2023-06-01")
    else:
        logger.warning(
            "dynamodb_key_loader: unknown auth scheme '%s' for %s",
            probe.auth,
            probe.id,
        )
        return None

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return None
            body = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        logger.debug(
            "dynamodb_key_loader: probe %s failed: %s",
            probe.id,
            type(exc).__name__,
        )
        return None

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None

    items = data.get("data") or data.get("models") or []
    ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("id") or item.get("name") or ""
        if isinstance(name, str) and name.startswith("models/"):
            name = name[len("models/"):]
        if name and isinstance(name, str):
            ids.append(name)
    return ids


def _select_provider() -> Optional[Tuple[_ProviderProbe, str]]:
    """Walk the probe list in order; return the first (probe, model) for
    which the /models endpoint responds 200 and at least one preferred
    model is in the response (or, failing that, the first preferred
    model on record — many small providers don't expose /models)."""
    for probe in _PROBES:
        advertised = _probe_models_endpoint(probe)
        if advertised is None:
            continue
        available = set(advertised)
        chosen: Optional[str] = None
        for pref in probe.preferred_models:
            if pref in available:
                chosen = pref
                break
        if not chosen and probe.preferred_models:
            # The endpoint responded but none of our preferred models
            # appeared. Use the first preferred name anyway — it may
            # still resolve at request time even when /models hides it.
            chosen = probe.preferred_models[0]
        if not chosen:
            continue
        return probe, chosen
    return None


def _write_config_overrides(probe: _ProviderProbe, model: str) -> None:
    """Patch ``model.default`` / ``model.provider`` / ``model.base_url``
    in ``$HERMES_HOME/config.yaml`` so the gateway's
    ``_resolve_gateway_model`` picks up the probed selection. Skips
    silently when the file is missing or PyYAML is unavailable.
    """
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    path = os.path.join(home, "config.yaml")
    if not os.path.exists(path):
        logger.debug(
            "dynamodb_key_loader: config.yaml not at %s; skipping override",
            path,
        )
        return

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning(
            "dynamodb_key_loader: PyYAML missing; cannot write config overrides"
        )
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "dynamodb_key_loader: failed to read %s: %s",
            path,
            type(exc).__name__,
        )
        return

    model_section = cfg.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
        cfg["model"] = model_section

    model_section["default"] = model
    model_section["provider"] = probe.hermes_provider
    if probe.base_url:
        model_section["base_url"] = probe.base_url
    else:
        model_section.pop("base_url", None)

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except OSError as exc:
        logger.warning(
            "dynamodb_key_loader: failed to write %s: %s",
            path,
            type(exc).__name__,
        )
        return

    msg = (
        f"dynamodb_key_loader: probed and selected "
        f"{probe.hermes_provider}/{model}"
    )
    if probe.base_url:
        msg += f" base_url={probe.base_url}"
    logger.info(msg)
    print(msg, file=sys.stderr, flush=True)


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

    # Dynamic provider selection. Walk _PROBES in priority order and
    # write the first working provider's choice into config.yaml so the
    # gateway always has a model that can answer. Skipped when the
    # operator has pinned a model via entrypoint's HERMES_MODEL_OVERRIDE
    # block, or via the explicit HERMES_PROBE_DISABLED escape hatch.
    if (
        not _is_truthy(os.environ.get("HERMES_PROBE_DISABLED"))
        and not os.environ.get("HERMES_MODEL_OVERRIDE")
    ):
        selection = _select_provider()
        if selection is not None:
            probe, model = selection
            _write_config_overrides(probe, model)
        else:
            no_provider_msg = (
                "dynamodb_key_loader: no provider responded successfully "
                "to /models probes; leaving config.yaml unchanged"
            )
            logger.warning(no_provider_msg)
            print(no_provider_msg, file=sys.stderr, flush=True)

    _applied = True
