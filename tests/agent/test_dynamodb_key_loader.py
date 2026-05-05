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
