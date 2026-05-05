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
