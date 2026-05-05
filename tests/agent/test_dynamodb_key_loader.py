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


def _scan_response(items, last_key=None):
    """Helper: build a Scan response with the given items and pagination cursor."""
    out = {"Items": items}
    if last_key is not None:
        out["LastEvaluatedKey"] = last_key
    return out


# ---------------------------------------------------------------------------
# Skip / gating
# ---------------------------------------------------------------------------

def test_idempotent_skip_when_already_applied(_no_aws_env, monkeypatch):
    """Second call is a no-op even if env changes between calls."""
    from agent import dynamodb_key_loader

    dynamodb_key_loader._applied = True
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    with patch.object(dynamodb_key_loader, "_build_client") as mock_build:
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_not_called()


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
        mock_build.return_value = None
        dynamodb_key_loader.apply_dynamodb_overrides()
        mock_build.assert_called_once()


def test_boto3_import_error_returns_without_raising(_no_aws_env, monkeypatch, caplog):
    """If boto3 isn't installed, log a warning and return cleanly."""
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
            dynamodb_key_loader.apply_dynamodb_overrides()

    assert any("boto3 not installed" in r.message for r in caplog.records)
    assert dynamodb_key_loader._applied is True


# ---------------------------------------------------------------------------
# _provider_targets
# ---------------------------------------------------------------------------

def test_provider_targets_yields_id_env_var_and_table_name():
    """_provider_targets yields (hermes_id, primary_env_var, table_provider_name)
    for every provider whose key we want to source from DynamoDB."""
    from agent import dynamodb_key_loader

    targets = list(dynamodb_key_loader._provider_targets())
    by_id = {hid: (env, tname) for (hid, env, tname) in targets}

    # _EXTRA_TARGETS entries
    assert by_id["openrouter"] == ("OPENROUTER_API_KEY", "OpenRouter")
    assert by_id["groq"] == ("GROQ_API_KEY", "Groq")
    assert by_id["openai"] == ("OPENAI_API_KEY", "OpenAI")

    # _TABLE_PROVIDER_NAMES entries that exist in the registry
    assert by_id["anthropic"] == ("ANTHROPIC_API_KEY", "Anthropic")
    assert by_id["xai"] == ("XAI_API_KEY", "xAI")
    assert by_id["minimax"] == ("MINIMAX_API_KEY", "Minimax")

    # gemini's primary env var is GOOGLE_API_KEY (Hermes's first alias)
    assert by_id["gemini"] == ("GOOGLE_API_KEY", "Gemini")


def test_provider_targets_does_not_duplicate():
    """If a hermes id appears in both _EXTRA_TARGETS and the registry, it
    is yielded only once (the _EXTRA_TARGETS entry wins)."""
    from agent import dynamodb_key_loader

    ids = [hid for (hid, _, _) in dynamodb_key_loader._provider_targets()]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# _scan_valid_keys_by_provider
# ---------------------------------------------------------------------------

def test_scan_groups_valid_keys_by_provider():
    """Single-page Scan groups keys by provider, dropping non-valid rows."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.scan.return_value = _scan_response([
        {"key": {"S": "k1"}, "provider": {"S": "Anthropic"},
         "validation_status": {"S": "valid"}},
        {"key": {"S": "k2"}, "provider": {"S": "Anthropic"},
         "validation_status": {"S": "valid"}},
        {"key": {"S": "k3"}, "provider": {"S": "Gemini"},
         "validation_status": {"S": "valid"}},
        {"key": {"S": "k4"}, "provider": {"S": "Gemini"},
         "validation_status": {"S": "invalid"}},
        # Missing provider — skipped
        {"key": {"S": "k5"}, "validation_status": {"S": "valid"}},
        # Missing key — skipped
        {"provider": {"S": "OpenAI"}, "validation_status": {"S": "valid"}},
    ])

    out = dynamodb_key_loader._scan_valid_keys_by_provider(
        fake_client, "chroma-llm-keys"
    )
    assert out == {
        "Anthropic": ["k1", "k2"],
        "Gemini": ["k3"],
    }
    fake_client.scan.assert_called_once()


def test_scan_paginates_through_last_evaluated_key():
    """Scan continues while LastEvaluatedKey is present."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.scan.side_effect = [
        _scan_response(
            [
                {"key": {"S": "k1"}, "provider": {"S": "Anthropic"},
                 "validation_status": {"S": "valid"}},
            ],
            last_key={"key": {"S": "k1"}},
        ),
        _scan_response(
            [
                {"key": {"S": "k2"}, "provider": {"S": "Gemini"},
                 "validation_status": {"S": "valid"}},
            ],
        ),  # no LastEvaluatedKey → terminates
    ]

    out = dynamodb_key_loader._scan_valid_keys_by_provider(
        fake_client, "chroma-llm-keys"
    )
    assert out == {"Anthropic": ["k1"], "Gemini": ["k2"]}
    assert fake_client.scan.call_count == 2

    # Second call must have passed the cursor.
    second_call_kwargs = fake_client.scan.call_args_list[1].kwargs
    assert second_call_kwargs.get("ExclusiveStartKey") == {"key": {"S": "k1"}}


def test_scan_swallows_aws_error_and_returns_partial(caplog):
    """A Scan error mid-pagination logs a warning and returns what was
    accumulated so far. The caller never sees an exception."""
    from agent import dynamodb_key_loader

    fake_client = MagicMock()
    fake_client.scan.side_effect = [
        _scan_response(
            [
                {"key": {"S": "k1"}, "provider": {"S": "Anthropic"},
                 "validation_status": {"S": "valid"}},
            ],
            last_key={"key": {"S": "k1"}},
        ),
        RuntimeError("Throttled"),
    ]

    with caplog.at_level("WARNING", logger="agent.dynamodb_key_loader"):
        out = dynamodb_key_loader._scan_valid_keys_by_provider(
            fake_client, "chroma-llm-keys"
        )

    assert out == {"Anthropic": ["k1"]}
    assert any("Scan failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# apply_dynamodb_overrides — integration with the scan
# ---------------------------------------------------------------------------

def test_apply_overrides_writes_env_var_for_each_provider(_no_aws_env, monkeypatch):
    """Rows for Anthropic and Gemini cause their env vars to be written
    with the first valid key returned."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-anthropic")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    fake_client = MagicMock()
    fake_client.scan.return_value = _scan_response([
        {"key": {"S": "sk-ant-fresh"}, "provider": {"S": "Anthropic"},
         "validation_status": {"S": "valid"}},
        {"key": {"S": "AIza-fresh"}, "provider": {"S": "Gemini"},
         "validation_status": {"S": "valid"}},
    ])

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-fresh"
    assert os.environ["GOOGLE_API_KEY"] == "AIza-fresh"


def test_apply_overrides_preserves_env_var_when_provider_absent(_no_aws_env, monkeypatch):
    """No row for a provider → env var untouched (DynamoDB does not win
    when there is nothing to win with)."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "preserved")

    fake_client = MagicMock()
    fake_client.scan.return_value = _scan_response([])

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    assert os.environ["OPENROUTER_API_KEY"] == "preserved"


def test_apply_overrides_uses_first_key_when_multiple_present(_no_aws_env, monkeypatch):
    """When multiple valid keys exist for a provider, the first one
    encountered during the scan wins."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    fake_client = MagicMock()
    fake_client.scan.return_value = _scan_response([
        {"key": {"S": "first"}, "provider": {"S": "Anthropic"},
         "validation_status": {"S": "valid"}},
        {"key": {"S": "second"}, "provider": {"S": "Anthropic"},
         "validation_status": {"S": "valid"}},
    ])

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    assert os.environ["ANTHROPIC_API_KEY"] == "first"


def test_apply_overrides_uses_custom_table_name(_no_aws_env, monkeypatch):
    """HERMES_DYNAMODB_KEY_TABLE overrides the default table name."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HERMES_DYNAMODB_KEY_TABLE", "staging-llm-keys")

    fake_client = MagicMock()
    fake_client.scan.return_value = _scan_response([])

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    seen_tables = {
        call.kwargs.get("TableName") for call in fake_client.scan.call_args_list
    }
    assert seen_tables == {"staging-llm-keys"}


def test_apply_overrides_logs_summary_to_stderr(_no_aws_env, monkeypatch, capsys):
    """The summary line is printed to stderr regardless of logging config."""
    from agent import dynamodb_key_loader

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    fake_client = MagicMock()
    fake_client.scan.return_value = _scan_response([
        {"key": {"S": "sk-ant"}, "provider": {"S": "Anthropic"},
         "validation_status": {"S": "valid"}},
    ])

    with patch.object(
        dynamodb_key_loader, "_build_client", return_value=fake_client
    ):
        dynamodb_key_loader.apply_dynamodb_overrides()

    captured = capsys.readouterr()
    assert "dynamodb_key_loader: applied" in captured.err
    assert "from chroma-llm-keys" in captured.err
