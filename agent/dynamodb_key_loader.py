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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

# Probe budget per provider. The harvested-key pool has high failure
# rates (flagged-as-leaked, expired, daily quota burned) so we want to
# find multiple working keys per provider to populate
# ``fallback_providers`` for runtime rotation, but we also want to cap
# startup time. _MAX_KEYS_PROBED_PER_PROVIDER bounds HTTP calls per
# provider; _MAX_WORKING_KEYS_PER_PROVIDER bounds how many verified
# entries we accept before moving on.
_MAX_KEYS_PROBED_PER_PROVIDER = 20
_MAX_WORKING_KEYS_PER_PROVIDER = 5


# Provider probe priority. The loader walks this list in order, calls
# each provider's `/models` endpoint with the env-var-loaded API key,
# and picks the first preferred model that the response advertises. The
# first provider whose probe returns 200 with at least one viable model
# wins, and the choice is written into ``$HERMES_HOME/config.yaml``.
@dataclass(frozen=True)
class _ProviderProbe:
    id: str                      # hermes-internal short name
    env_var: str                 # env var the loader writes the key into
    completions_url: str         # POST endpoint for a 1-token sanity check
    body_style: str              # 'openai' or 'anthropic'
    auth: str                    # 'bearer' or 'x-api-key'
    hermes_provider: str         # value to put in config.yaml model.provider
    base_url: Optional[str]      # value to put in config.yaml model.base_url
    preferred_models: Tuple[str, ...] = field(default_factory=tuple)


_PROBES: Tuple[_ProviderProbe, ...] = (
    _ProviderProbe(
        id="openai",
        env_var="OPENAI_API_KEY",
        completions_url="https://api.openai.com/v1/chat/completions",
        body_style="openai",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://api.openai.com/v1",
        preferred_models=("gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4"),
    ),
    _ProviderProbe(
        id="gemini",
        env_var="GOOGLE_API_KEY",
        completions_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        body_style="openai",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        preferred_models=(
            "gemini-3.1-flash-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
        ),
    ),
    _ProviderProbe(
        id="xai",
        env_var="XAI_API_KEY",
        completions_url="https://api.x.ai/v1/chat/completions",
        body_style="openai",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://api.x.ai/v1",
        preferred_models=("grok-4-latest", "grok-4", "grok-3-latest", "grok-3"),
    ),
    _ProviderProbe(
        id="anthropic",
        env_var="ANTHROPIC_API_KEY",
        completions_url="https://api.anthropic.com/v1/messages",
        body_style="anthropic",
        auth="x-api-key",
        hermes_provider="anthropic",
        base_url=None,
        preferred_models=(
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ),
    ),
    _ProviderProbe(
        id="groq",
        env_var="GROQ_API_KEY",
        completions_url="https://api.groq.com/openai/v1/chat/completions",
        body_style="openai",
        auth="bearer",
        hermes_provider="custom",
        base_url="https://api.groq.com/openai/v1",
        preferred_models=(
            "llama-4.1-8b-instant",
            "llama-4.1-70b-versatile",
            "llama-4-70b",
        ),
    ),
    _ProviderProbe(
        id="minimax",
        env_var="MINIMAX_API_KEY",
        completions_url="https://api.minimax.chat/v1/text/chatcompletion_v2",
        body_style="openai",
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


def _probe_completion_with_key(
    probe: _ProviderProbe, model: str, key: str
) -> bool:
    """POST a 1-token chat completion using a specific key. Returns True
    on 2xx, False on any auth/quota/network failure. Never raises.

    Auth-only checks (e.g. GET /models) miss the common pool failure
    modes — flagged-as-leaked, expired, free-tier daily quota burned —
    so the probe forces a real completion against the model the bot
    would actually use.
    """
    if not key:
        return False

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    req = urllib.request.Request(
        probe.completions_url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    if probe.auth == "bearer":
        req.add_header("Authorization", f"Bearer {key}")
    elif probe.auth == "x-api-key":
        req.add_header("x-api-key", key)
        req.add_header("anthropic-version", "2023-06-01")
    else:
        return False

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        logger.debug(
            "dynamodb_key_loader: probe %s/%s rejected (key=%s…): HTTP %s",
            probe.id,
            model,
            key[:6],
            exc.code,
        )
        return False
    except (urllib.error.URLError, OSError) as exc:
        logger.debug(
            "dynamodb_key_loader: probe %s/%s network error (key=%s…): %s",
            probe.id,
            model,
            key[:6],
            type(exc).__name__,
        )
        return False


def _provider_table_name_for(hermes_id: str) -> str:
    """Return the table's capitalized provider value for a Hermes id."""
    for hid, _, table_name in _EXTRA_TARGETS:
        if hid == hermes_id:
            return table_name
    return _TABLE_PROVIDER_NAMES.get(hermes_id, hermes_id)


def _build_fallback_chain(
    keys_by_provider: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Walk the probe list in priority order, find up to
    ``_MAX_WORKING_KEYS_PER_PROVIDER`` working (provider, key, model)
    combinations per provider (capped at
    ``_MAX_KEYS_PROBED_PER_PROVIDER`` HTTP probes), and return them as
    fallback-chain entries ready to be dropped into config.yaml under
    ``fallback_providers``.

    Each entry is a dict:
        {provider, model, base_url?, api_key}

    Hermes's AIAgent.__init__ normalizes this list into ``_fallback_chain``
    and ``_try_activate_fallback`` advances through it whenever the active
    credential returns a non-retryable 4xx, giving us free runtime key
    rotation across both keys-within-a-provider and across providers.

    The returned list preserves priority: best provider's working keys
    first (for primary + early fallbacks), then next provider's, etc.
    """
    chain: List[Dict[str, Any]] = []
    for probe in _PROBES:
        candidates = keys_by_provider.get(_provider_table_name_for(probe.id))
        if not candidates:
            continue
        probed = 0
        found = 0
        for key in candidates:
            if probed >= _MAX_KEYS_PROBED_PER_PROVIDER:
                break
            if found >= _MAX_WORKING_KEYS_PER_PROVIDER:
                break
            chosen_model: Optional[str] = None
            for model in probe.preferred_models:
                probed += 1
                if _probe_completion_with_key(probe, model, key):
                    chosen_model = model
                    break
                if probed >= _MAX_KEYS_PROBED_PER_PROVIDER:
                    break
            if chosen_model is None:
                continue
            entry: Dict[str, Any] = {
                "provider": probe.hermes_provider,
                "model": chosen_model,
                "api_key": key,
            }
            if probe.base_url:
                entry["base_url"] = probe.base_url
            chain.append(entry)
            found += 1
    return chain


def _write_config_overrides(chain: List[Dict[str, Any]]) -> None:
    """Patch ``$HERMES_HOME/config.yaml`` so the gateway picks up:

      * ``model.default`` / ``model.provider`` / ``model.base_url`` /
        ``model.api_key`` — the first verified entry from the chain
        becomes the primary that the gateway boots on.
      * ``fallback_providers`` — every other verified entry, in
        priority order, so AIAgent.__init__ can rotate to it on
        non-retryable errors.

    Skips silently when the file is missing or PyYAML is unavailable
    (the loader is best-effort; agent must still boot).
    """
    if not chain:
        return

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

    primary = chain[0]
    fallbacks = chain[1:]

    model_section = cfg.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
        cfg["model"] = model_section

    model_section["default"] = primary["model"]
    model_section["provider"] = primary["provider"]
    if primary.get("base_url"):
        model_section["base_url"] = primary["base_url"]
    else:
        model_section.pop("base_url", None)
    if primary.get("api_key"):
        model_section["api_key"] = primary["api_key"]
    else:
        model_section.pop("api_key", None)

    if fallbacks:
        cfg["fallback_providers"] = fallbacks
    else:
        cfg.pop("fallback_providers", None)

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
        f"dynamodb_key_loader: primary={primary['provider']}/{primary['model']} "
        f"+ {len(fallbacks)} fallback entries (probed pool keys)"
    )
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

    # Dynamic provider selection + fallback chain. For each provider in
    # priority order, probe up to _MAX_KEYS_PROBED_PER_PROVIDER keys
    # × each preferred model with a 1-token completion, accumulating
    # up to _MAX_WORKING_KEYS_PER_PROVIDER verified entries. The first
    # entry becomes the primary (model.default/provider/base_url/api_key),
    # the rest become config.yaml's `fallback_providers`. Hermes's
    # existing _try_activate_fallback rotates through the chain on every
    # non-retryable 4xx, giving us runtime key rotation across both keys
    # within a provider and across providers when one is exhausted.
    # Skipped when HERMES_MODEL_OVERRIDE pins a choice, or
    # HERMES_PROBE_DISABLED is truthy.
    if (
        not _is_truthy(os.environ.get("HERMES_PROBE_DISABLED"))
        and not os.environ.get("HERMES_MODEL_OVERRIDE")
    ):
        chain = _build_fallback_chain(keys_by_provider)
        if chain:
            primary = chain[0]
            # Mirror the primary's key into the env var so any code path
            # that still reads OPENAI_API_KEY / ANTHROPIC_API_KEY etc.
            # (rather than the per-fallback explicit api_key) sees the
            # verified-working credential.
            for probe in _PROBES:
                if (
                    probe.hermes_provider == primary["provider"]
                    and probe.base_url == primary.get("base_url")
                ):
                    os.environ[probe.env_var] = primary["api_key"]
                    break
            _write_config_overrides(chain)
        else:
            no_provider_msg = (
                "dynamodb_key_loader: no (provider, key, model) combination "
                "returned 2xx on a 1-token completion; leaving config.yaml "
                "unchanged — the bot will fall back to whatever the existing "
                "config and env vars allow"
            )
            logger.warning(no_provider_msg)
            print(no_provider_msg, file=sys.stderr, flush=True)

    _applied = True
