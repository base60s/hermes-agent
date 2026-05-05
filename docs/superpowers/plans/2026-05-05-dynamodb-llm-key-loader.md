# DynamoDB LLM Key Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a startup-time loader that pulls LLM provider API keys from the `chroma-llm-keys` DynamoDB table and writes them into env vars, so the Hermeregildo Telegram bot on Railway sources its OpenRouter key from DynamoDB instead of `OPENROUTER_API_KEY` directly.

**Architecture:** One new module `agent/dynamodb_key_loader.py` exposes `apply_dynamodb_overrides()`. It iterates `hermes_cli.auth.PROVIDER_REGISTRY`, calls `GetItem` for each provider id against `chroma-llm-keys`, and writes the `value` attribute into the provider's primary `api_key_env_vars` entry. Idempotent, never raises, skipped when AWS creds absent or `HERMES_DYNAMODB_KEY_DISABLED` is set. Three entry points (`run_agent.py`, `cli.py`, `gateway/run.py`) gain a single import-and-call pair.

**Tech Stack:** Python 3.13, `boto3` (already an optional dep under `[bedrock]`), `botocore.stub.Stubber` for tests, pytest.

**Spec:** [`docs/superpowers/specs/2026-05-05-dynamodb-llm-key-loader-design.md`](../specs/2026-05-05-dynamodb-llm-key-loader-design.md)

---

## File Structure

**Create:**
- `agent/dynamodb_key_loader.py` — public `apply_dynamodb_overrides()`, private helpers for client construction, provider iteration, and per-provider lookup.
- `tests/agent/test_dynamodb_key_loader.py` — unit tests using `botocore.stub.Stubber`.

**Modify:**
- `run_agent.py` — add import + call inside `main()` (line ~13849).
- `cli.py` — add import + call inside `main()` (line ~11755).
- `gateway/run.py` — add import + call inside `main()` (line ~13665).
- `pyproject.toml` — promote `boto3` from the `[bedrock]` extra into core dependencies (the loader runs by default; soft-fail on ImportError handles edge cases but Railway image must have boto3).

---

## Task 1: Module skeleton with idempotent guard

**Files:**
- Create: `agent/dynamodb_key_loader.py`
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing test for idempotency**

Create `tests/agent/test_dynamodb_key_loader.py`:

```python
"""Tests for the DynamoDB-backed LLM key loader."""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_loader_state():
    """Reset the module-level _applied flag between tests."""
    from agent import dynamodb_key_loader
    dynamodb_key_loader._applied = False
    yield
    dynamodb_key_loader._applied = False


@pytest.fixture
def _no_aws_env(monkeypatch):
    """Strip AWS env vars so tests don't accidentally hit real AWS."""
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "HERMES_DYNAMODB_KEY_DISABLED",
        "HERMES_DYNAMODB_KEY_TABLE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_idempotent_skip_when_already_applied(_no_aws_env, monkeypatch):
    """Second call is a no-op even if env changes between calls."""
    from agent import dynamodb_key_loader

    dynamodb_key_loader._applied = True
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    with patch.object(dynamodb_key_loader, "_build_client") as mock_build:
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_idempotent_skip_when_already_applied -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.dynamodb_key_loader'`

- [ ] **Step 3: Write the minimal module**

Create `agent/dynamodb_key_loader.py`:

```python
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
    except Exception as exc:  # pragma: no cover — boto3 client construction is forgiving
        logger.warning(
            "dynamodb_key_loader: failed to construct boto3 client: %s",
            type(exc).__name__,
        )
        return None


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

    # Provider iteration and per-provider GetItem are added in later tasks.
    _applied = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_idempotent_skip_when_already_applied -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/dynamodb_key_loader.py tests/agent/test_dynamodb_key_loader.py
git commit -m "feat(dynamodb_key_loader): module skeleton with idempotency guard"
```

---

## Task 2: Skip when AWS creds absent or loader disabled

**Files:**
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/agent/test_dynamodb_key_loader.py`:

```python
def test_skip_when_aws_access_key_id_unset(_no_aws_env):
    """Without AWS_ACCESS_KEY_ID, no boto3 client is built."""
    from agent import dynamodb_key_loader

    with patch.object(dynamodb_key_loader, "_build_client") as mock_build:
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_not_called()


def test_skip_when_disabled_env_truthy(_no_aws_env, monkeypatch):
    """HERMES_DYNAMODB_KEY_DISABLED=1 short-circuits even with AWS creds."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HERMES_DYNAMODB_KEY_DISABLED", "1")

    with patch.object(dynamodb_key_loader, "_build_client") as mock_build:
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_not_called()


@pytest.mark.parametrize("flag", ["true", "TRUE", " yes ", "On"])
def test_disabled_env_is_truthy_with_common_values(_no_aws_env, monkeypatch, flag):
    """Truthy parsing accepts case-insensitive variants and trims whitespace."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HERMES_DYNAMODB_KEY_DISABLED", flag)

    with patch.object(dynamodb_key_loader, "_build_client") as mock_build:
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_not_called()


@pytest.mark.parametrize("flag", ["", "0", "false", "no", "off", "anything-else"])
def test_disabled_env_is_falsy_when_not_truthy(_no_aws_env, monkeypatch, flag):
    """Non-truthy values for HERMES_DYNAMODB_KEY_DISABLED do not skip."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HERMES_DYNAMODB_KEY_DISABLED", flag)

    with patch.object(dynamodb_key_loader, "_build_client") as mock_build:
        # _build_client is mocked to return None so the loader exits early
        # but it should still be *called* — that's what we're asserting.
        mock_build.return_value = None
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/agent/test_dynamodb_key_loader.py -v`
Expected: All five new tests PASS (the gating logic was already implemented in Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/agent/test_dynamodb_key_loader.py
git commit -m "test(dynamodb_key_loader): cover skip conditions and truthy parsing"
```

---

## Task 3: boto3 ImportError soft-fail

**Files:**
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_dynamodb_key_loader.py`:

```python
def test_boto3_import_error_returns_without_raising(_no_aws_env, monkeypatch, caplog):
    """If boto3 isn't installed, log a warning and return cleanly."""
    import sys
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    with caplog.at_level("WARNING", logger="agent.dynamodb_key_loader"):
        with patch("builtins.__import__", side_effect=_fake_import):
            # Must not raise.
            dynamodb_key_loader.apply_dynamodb_overrides()

    assert any("boto3 not installed" in r.message for r in caplog.records)
    assert dynamodb_key_loader._applied is True
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_boto3_import_error_returns_without_raising -v`
Expected: PASS (logic was already in `_build_client`).

- [ ] **Step 3: Commit**

```bash
git add tests/agent/test_dynamodb_key_loader.py
git commit -m "test(dynamodb_key_loader): cover boto3 ImportError soft-fail"
```

---

## Task 4: Provider iteration helper

**Files:**
- Modify: `agent/dynamodb_key_loader.py`
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_dynamodb_key_loader.py`:

```python
def test_provider_targets_yields_id_and_primary_env_var():
    """_provider_targets yields (provider_id, primary_env_var) for providers
    that have at least one api_key_env_var, lowercasing the provider id."""
    from agent import dynamodb_key_loader

    targets = dict(dynamodb_key_loader._provider_targets())

    # OpenRouter must be present and map to OPENROUTER_API_KEY.
    assert "openrouter" in targets
    assert targets["openrouter"] == "OPENROUTER_API_KEY"

    # Anthropic must be present and pick the *first* env var alias.
    assert "anthropic" in targets
    assert targets["anthropic"] == "ANTHROPIC_API_KEY"

    # All keys are lowercase, all values are non-empty strings.
    for provider_id, env_var in targets.items():
        assert provider_id == provider_id.lower(), provider_id
        assert env_var and isinstance(env_var, str), (provider_id, env_var)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_provider_targets_yields_id_and_primary_env_var -v`
Expected: FAIL with `AttributeError: module 'agent.dynamodb_key_loader' has no attribute '_provider_targets'`

- [ ] **Step 3: Add the helper**

Edit `agent/dynamodb_key_loader.py` — add this function above `apply_dynamodb_overrides`:

```python
def _provider_targets() -> Iterable[Tuple[str, str]]:
    """Yield (provider_id, primary_env_var) for every provider in the
    Hermes registry that has at least one api_key_env_var configured."""
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
        yield provider_id.lower(), primary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_provider_targets_yields_id_and_primary_env_var -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/dynamodb_key_loader.py tests/agent/test_dynamodb_key_loader.py
git commit -m "feat(dynamodb_key_loader): provider iteration helper"
```

---

## Task 5: Per-provider GetItem helper (happy path)

**Files:**
- Modify: `agent/dynamodb_key_loader.py`
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_dynamodb_key_loader.py`:

```python
def test_fetch_provider_key_happy_path():
    """_fetch_provider_key returns the value when GetItem returns an Item
    with a non-empty `value` string attribute."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.get_item.return_value = {
        "Item": {
            "key": {"S": "openrouter"},
            "value": {"S": "sk-or-v1-test"},
        }
    }

    result = dynamodb_key_loader._fetch_provider_key(
        fake_client, "chroma-llm-keys", "openrouter"
    )

    assert result == "sk-or-v1-test"
    fake_client.get_item.assert_called_once_with(
        TableName="chroma-llm-keys",
        Key={"key": {"S": "openrouter"}},
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_fetch_provider_key_happy_path -v`
Expected: FAIL with `AttributeError: ... has no attribute '_fetch_provider_key'`

- [ ] **Step 3: Add the helper**

Edit `agent/dynamodb_key_loader.py` — add this function above `_provider_targets`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_dynamodb_key_loader.py::test_fetch_provider_key_happy_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/dynamodb_key_loader.py tests/agent/test_dynamodb_key_loader.py
git commit -m "feat(dynamodb_key_loader): per-provider GetItem helper (happy path)"
```

---

## Task 6: GetItem failure modes

**Files:**
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/agent/test_dynamodb_key_loader.py`:

```python
def test_fetch_provider_key_returns_none_when_no_item():
    """No Item in response → return None."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.get_item.return_value = {}

    result = dynamodb_key_loader._fetch_provider_key(
        fake_client, "chroma-llm-keys", "openrouter"
    )

    assert result is None


def test_fetch_provider_key_returns_none_when_value_attribute_missing():
    """Item present but no `value` attribute → return None."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.get_item.return_value = {
        "Item": {"key": {"S": "openrouter"}}
    }

    result = dynamodb_key_loader._fetch_provider_key(
        fake_client, "chroma-llm-keys", "openrouter"
    )

    assert result is None


def test_fetch_provider_key_returns_none_when_value_string_empty():
    """`value` present but empty string → return None."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.get_item.return_value = {
        "Item": {"key": {"S": "openrouter"}, "value": {"S": ""}}
    }

    result = dynamodb_key_loader._fetch_provider_key(
        fake_client, "chroma-llm-keys", "openrouter"
    )

    assert result is None


def test_fetch_provider_key_swallows_client_error(caplog):
    """Boto ClientError must not propagate; warning is logged."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.get_item.side_effect = RuntimeError("AccessDeniedException")

    with caplog.at_level("WARNING", logger="agent.dynamodb_key_loader"):
        result = dynamodb_key_loader._fetch_provider_key(
            fake_client, "chroma-llm-keys", "openrouter"
        )

    assert result is None
    assert any(
        "GetItem failed for provider=openrouter" in r.message
        for r in caplog.records
    )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/agent/test_dynamodb_key_loader.py -v`
Expected: All four new tests PASS (logic added in Task 5 already covers these).

- [ ] **Step 3: Commit**

```bash
git add tests/agent/test_dynamodb_key_loader.py
git commit -m "test(dynamodb_key_loader): cover GetItem absence and error paths"
```

---

## Task 7: Wire iteration + write into os.environ

**Files:**
- Modify: `agent/dynamodb_key_loader.py`
- Test: `tests/agent/test_dynamodb_key_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/agent/test_dynamodb_key_loader.py`:

```python
def test_apply_overrides_writes_env_var_for_present_row(
    _no_aws_env, monkeypatch
):
    """A present DynamoDB row overwrites the corresponding env var."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-env-value")

    fake_client = MagicMock()

    def _get_item(TableName, Key):
        if Key == {"key": {"S": "openrouter"}}:
            return {
                "Item": {
                    "key": {"S": "openrouter"},
                    "value": {"S": "sk-or-fresh"},
                }
            }
        return {}

    fake_client.get_item.side_effect = _get_item

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-fresh"


def test_apply_overrides_preserves_env_var_when_row_absent(
    _no_aws_env, monkeypatch
):
    """Absent rows leave existing env vars untouched."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "preserved-env-value")

    fake_client = MagicMock()
    fake_client.get_item.return_value = {}  # nothing in the table

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    assert os.environ["OPENROUTER_API_KEY"] == "preserved-env-value"


def test_apply_overrides_partial_failure_does_not_block_other_providers(
    _no_aws_env, monkeypatch
):
    """One provider's GetItem error must not prevent others from being applied."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    fake_client = MagicMock()

    def _get_item(TableName, Key):
        provider = Key["key"]["S"]
        if provider == "openrouter":
            raise RuntimeError("AccessDeniedException")
        if provider == "anthropic":
            return {
                "Item": {
                    "key": {"S": "anthropic"},
                    "value": {"S": "sk-ant-fresh"},
                }
            }
        return {}

    fake_client.get_item.side_effect = _get_item

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    assert "OPENROUTER_API_KEY" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-fresh"


def test_apply_overrides_uses_custom_table_name(_no_aws_env, monkeypatch):
    """HERMES_DYNAMODB_KEY_TABLE overrides the default table name."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HERMES_DYNAMODB_KEY_TABLE", "staging-llm-keys")

    fake_client = MagicMock()
    fake_client.get_item.return_value = {}

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    # At least one call must have used the override table name.
    seen_tables = {
        call.kwargs.get("TableName") for call in fake_client.get_item.call_args_list
    }
    assert seen_tables == {"staging-llm-keys"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_dynamodb_key_loader.py -v -k "apply_overrides"`
Expected: All four FAIL — currently `apply_dynamodb_overrides` does no iteration.

- [ ] **Step 3: Wire the iteration**

Edit `agent/dynamodb_key_loader.py` — replace the body of `apply_dynamodb_overrides` after `client = _build_client()` with the full implementation:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_dynamodb_key_loader.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/dynamodb_key_loader.py tests/agent/test_dynamodb_key_loader.py
git commit -m "feat(dynamodb_key_loader): wire iteration into apply_dynamodb_overrides"
```

---

## Task 8: Promote boto3 to a core dependency

**Files:**
- Modify: `pyproject.toml`

The Telegram bot deployment must always have boto3 available — it's no longer optional now that key resolution depends on it. The soft-fail path remains for unusual environments.

- [ ] **Step 1: Locate the boto3 declaration**

Open `pyproject.toml` and find the line:

```
bedrock = ["boto3>=1.35.0,<2"]
```

It lives under `[project.optional-dependencies]`.

- [ ] **Step 2: Add boto3 to core dependencies**

Find the `dependencies = [...]` array under `[project]` and add (in alphabetical order with the other entries):

```
"boto3>=1.35.0,<2",
```

Leave the `bedrock = ["boto3>=1.35.0,<2"]` extra in place — it remains a no-op extra so existing install commands using `[bedrock]` keep working.

- [ ] **Step 3: Verify the dependency declaration**

Run: `grep -n "boto3" /Users/mauriciovelez/Proyectos/hermes/pyproject.toml`
Expected: at least two lines — one in `dependencies` and one in `bedrock`.

- [ ] **Step 4: Verify boto3 is importable in the dev env**

Run: `python -c "import boto3; print(boto3.__version__)"`
Expected: a version string `>=1.35.0`.
If it fails: `uv pip install -e .` (or your project's standard install command) to refresh deps.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: promote boto3 to a core dependency for DynamoDB key loader"
```

---

## Task 9: Wire loader into `gateway/run.py` (Telegram bot entry point)

**Files:**
- Modify: `gateway/run.py:13665-13686`

This is the Railway deployment entry point — the highest priority of the three.

- [ ] **Step 1: Read the current main()**

Run: `sed -n '13665,13687p' /Users/mauriciovelez/Proyectos/hermes/gateway/run.py`
Expected: the body shown below (verify before editing).

The current body is:

```python
def main():
    """CLI entry point for the gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)
```

- [ ] **Step 2: Add the import-and-call pair**

Insert the two new lines immediately before `success = asyncio.run(start_gateway(config))`. The patched body becomes:

```python
def main():
    """CLI entry point for the gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)

    from agent.dynamodb_key_loader import apply_dynamodb_overrides
    apply_dynamodb_overrides()

    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)
```

- [ ] **Step 3: Smoke-test the import path**

Run: `python -c "from gateway.run import main; print('ok')"`
Expected: `ok` (no ImportError).

Run: `python -c "
import os
os.environ.pop('AWS_ACCESS_KEY_ID', None)
from agent.dynamodb_key_loader import apply_dynamodb_overrides
apply_dynamodb_overrides()
print('ok')
"`
Expected: `ok` (no AWS creds → loader skips, no errors).

- [ ] **Step 4: Commit**

```bash
git add gateway/run.py
git commit -m "feat(gateway): apply DynamoDB key overrides at gateway startup"
```

---

## Task 10: Wire loader into `run_agent.py`

**Files:**
- Modify: `run_agent.py:13849` (top of `main()`)

- [ ] **Step 1: Read the current main() signature and first lines**

Run: `sed -n '13849,13900p' /Users/mauriciovelez/Proyectos/hermes/run_agent.py`
Expected: the function signature followed by its docstring (verify before editing).

- [ ] **Step 2: Find the end of the docstring and insert the call**

The first executable statement in `main()` (after the closing `"""` of the docstring) gets two new lines inserted *before* it. The two lines:

```python
    from agent.dynamodb_key_loader import apply_dynamodb_overrides
    apply_dynamodb_overrides()
```

To find the insertion point precisely:

Run: `awk 'NR>=13849 && /^    """$/ && ++hit==2 {print NR; exit}' /Users/mauriciovelez/Proyectos/hermes/run_agent.py`
This prints the line number of the closing `"""` of `main()`'s docstring. Call it `L`. Insert the two new lines on line `L+1`.

Use Edit to splice in the two-line block immediately after the docstring's closing `"""` and before the first non-docstring statement of `main()`. Indentation: four spaces (the function body indent).

- [ ] **Step 3: Smoke-test the import path**

Run: `python -c "from run_agent import main; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add run_agent.py
git commit -m "feat(run_agent): apply DynamoDB key overrides at agent startup"
```

---

## Task 11: Wire loader into `cli.py`

**Files:**
- Modify: `cli.py:11755` (top of `main()`)

- [ ] **Step 1: Read the current main() signature and first lines**

Run: `sed -n '11755,11820p' /Users/mauriciovelez/Proyectos/hermes/cli.py`
Expected: the function signature, parameters, and the first lines of its body.

- [ ] **Step 2: Insert the two-line call at the start of the function body**

After the function signature closes (`):`), and *before* the first executable statement (i.e. before any docstring if present, or right at the top of the body if not — apply *after* a docstring if one exists, same as Task 10), insert:

```python
    from agent.dynamodb_key_loader import apply_dynamodb_overrides
    apply_dynamodb_overrides()
```

Indentation: four spaces (function-body indent).

- [ ] **Step 3: Smoke-test the import path**

Run: `python -c "from cli import main; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add cli.py
git commit -m "feat(cli): apply DynamoDB key overrides at CLI startup"
```

---

## Task 12: Full-suite verification

**Files:** none

- [ ] **Step 1: Run the loader's unit tests**

Run: `pytest tests/agent/test_dynamodb_key_loader.py -v`
Expected: every test passes.

- [ ] **Step 2: Run the broader auxiliary client tests to confirm no regressions**

Run: `pytest tests/agent/test_auxiliary_client.py tests/agent/test_credential_pool.py -v`
Expected: all green. (These cover provider resolution and the credential pool — the surfaces nearest to ours.)

- [ ] **Step 3: Run a fast-import smoke for all three entry points**

Run:
```bash
python -c "from gateway.run import main; print('gateway: ok')"
python -c "from run_agent import main; print('run_agent: ok')"
python -c "from cli import main; print('cli: ok')"
```
Expected: three `ok` lines, no ImportError.

- [ ] **Step 4: Manual integration smoke (optional, only if AWS creds available locally)**

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
unset OPENROUTER_API_KEY
python -c "
from agent.dynamodb_key_loader import apply_dynamodb_overrides
apply_dynamodb_overrides()
import os
print('OPENROUTER_API_KEY set:', bool(os.environ.get('OPENROUTER_API_KEY')))
"
```
Expected: `OPENROUTER_API_KEY set: True` and a log line `applied N/<total> provider keys from chroma-llm-keys`.
Skip this step in environments without AWS access — CI must not depend on it.

---

## Task 13: Railway deployment checklist

**Files:** none (operational)

- [ ] **Step 1: Set the AWS env vars on the Railway service**

In the Railway dashboard for the Hermeregildo service, add two service variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

These are the IAM credentials that have `dynamodb:GetItem` permission on `arn:aws:dynamodb:us-east-1:539247475729:table/chroma-llm-keys`.

- [ ] **Step 2: Confirm `AWS_REGION` is correct**

If `AWS_REGION` is not already set on the service, leave it unset — the loader defaults to `us-east-1`. If it *is* set to a different region, change it to `us-east-1`.

- [ ] **Step 3: (Optional) remove `OPENROUTER_API_KEY` from Railway**

Once the next deploy logs `dynamodb_key_loader: applied N/<total> provider keys from chroma-llm-keys` with a non-zero `N`, remove `OPENROUTER_API_KEY` from Railway service variables. The bot will continue using the DynamoDB-sourced key.

If you'd rather keep it as a belt-and-suspenders fallback, leave it — DynamoDB still wins on every startup.

- [ ] **Step 4: Trigger a Railway redeploy**

Either push an empty commit (`git commit --allow-empty -m "chore: trigger redeploy"`) and `git push`, or use the Railway dashboard's "Redeploy" button.

- [ ] **Step 5: Verify the bot answers**

Send a message to the Telegram bot and confirm it responds with output from `deepseek-v4-flash` via OpenRouter. Check the Railway logs for the `dynamodb_key_loader: applied …` line.

- [ ] **Step 6: Document in the PR description**

When you open the PR, include in the description:
- Link to the spec.
- The Railway service variables that were added.
- Whether `OPENROUTER_API_KEY` was removed or left in place.
