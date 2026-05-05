# DynamoDB LLM Key Loader — Design

**Date:** 2026-05-05
**Status:** Approved (brainstorming → writing-plans)
**Owner:** Hermeregildo Telegram bot deployment (Railway)

## Goal

Replace the practice of injecting LLM provider API keys into the Hermes-Agent
process through individual environment variables (`OPENROUTER_API_KEY`,
`ANTHROPIC_API_KEY`, …) with a single startup-time lookup against the
DynamoDB table `chroma-llm-keys`. Env vars remain a fallback when the
DynamoDB row is absent or the lookup fails, so local development and
emergency overrides keep working.

The immediate motivation is the Hermeregildo Telegram bot deployment on
Railway, which currently calls `deepseek/deepseek-v4-flash` via OpenRouter
using `OPENROUTER_API_KEY`. After this change, the OpenRouter key (and any
other provider key the operator stores in the table) is sourced from
DynamoDB on every container start.

## Scope

**In scope**

- A new module `agent/dynamodb_key_loader.py` exposing
  `apply_dynamodb_overrides()`.
- Wiring that function into the three startup entry points that resolve
  provider clients: `run_agent.py`, `gateway/run.py`, `cli.py`.
- Reading rows for every provider listed in
  `hermes_cli/auth.py:PROVIDER_REGISTRY`.
- Writing the fetched secret into the provider's primary
  `api_key_env_vars` entry.
- Configuration via env vars: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_REGION` (default `us-east-1`), `HERMES_DYNAMODB_KEY_TABLE` (default
  `chroma-llm-keys`), `HERMES_DYNAMODB_KEY_DISABLED`.
- Unit tests with `botocore.stub.Stubber`.

**Out of scope**

- Telegram allowlist enforcement and daily group summary features (separate
  specs if requested).
- Hot rotation of keys mid-process. This loader is one-shot at startup.
- Replacing or modifying the existing credential pool subsystem
  (`hermes_cli/auth.py:read_credential_pool`).
- Storing AWS credentials anywhere other than env vars on the Railway
  service.
- Logging key contents or any partial fingerprint of a fetched secret.

## Architecture

```
                 ┌─────────────────────────────┐
 Railway env →   │  AWS_ACCESS_KEY_ID          │
                 │  AWS_SECRET_ACCESS_KEY      │
                 │  AWS_REGION (opt)           │
                 │  HERMES_DYNAMODB_KEY_*      │
                 └──────────────┬──────────────┘
                                │
                  startup       ▼
   run_agent.py ─►  apply_dynamodb_overrides()
   gateway/run.py ─►       │
   cli.py ─►               │
                           ▼
            ┌──────────────────────────────┐
            │ DynamoDB chroma-llm-keys     │
            │  GetItem key=<provider_id>   │  per provider in
            │  → Item.value.S              │  PROVIDER_REGISTRY
            └──────────────┬───────────────┘
                           ▼
              os.environ[<primary env var>] = value
                           │
                           ▼
   resolve_provider_client(...) — unchanged, reads env as today
```

The loader sits *before* `agent/auxiliary_client.py:resolve_provider_client`
in the call graph and only mutates `os.environ`. After the loader runs,
every existing code path that reads an LLM env var sees the DynamoDB-sourced
value (or, on miss/failure, the original env var the operator set).

## Components

### `agent/dynamodb_key_loader.py` (new)

Public surface:

```python
def apply_dynamodb_overrides() -> None: ...
```

Internal helpers (private):

- `_should_run() -> bool` — checks `HERMES_DYNAMODB_KEY_DISABLED`,
  presence of `AWS_ACCESS_KEY_ID`, and the once-flag.
- `_build_client()` — constructs a `boto3.client('dynamodb', region_name=…)`,
  catching `ImportError` and `botocore.exceptions.NoCredentialsError`.
- `_fetch_provider_key(client, table, provider_id) -> Optional[str]` —
  performs the `GetItem` and extracts `Item['value']['S']`. Returns
  `None` on any failure or absence.
- `_provider_targets() -> Iterable[Tuple[str, str]]` — yields
  `(provider_id, primary_env_var)` from
  `hermes_cli.auth.PROVIDER_REGISTRY`, skipping providers with empty
  `api_key_env_vars`.

A module-level `_applied = False` flag makes `apply_dynamodb_overrides()`
idempotent.

### Entry-point patches

In each of `run_agent.py`, `gateway/run.py`, `cli.py`, before any call to
`resolve_provider_client` (i.e. before any LLM provider resolution can
read env vars), add:

```python
from agent.dynamodb_key_loader import apply_dynamodb_overrides
apply_dynamodb_overrides()
```

In practice this means putting the call at the top of the `main()` (or
equivalent CLI entry function) once `argparse` / config parsing has run
but before any agent-side imports trigger provider resolution. The
import-and-call pair is the only edit those files receive.

## Behavior / data flow

1. Entry point calls `apply_dynamodb_overrides()`.
2. If `HERMES_DYNAMODB_KEY_DISABLED` is truthy → log debug, return.
3. If `AWS_ACCESS_KEY_ID` is unset → log debug, return. (Local dev path.)
4. If `boto3` import or client construction fails → log warning, return.
5. Resolve table name: `os.environ.get('HERMES_DYNAMODB_KEY_TABLE',
   'chroma-llm-keys')`.
6. For each `(provider_id, primary_env_var)` from `_provider_targets()`:
   1. Call `client.get_item(TableName=table,
      Key={'key': {'S': provider_id}})`.
   2. On any `ClientError`, `EndpointConnectionError`, or other
      `BotoCoreError`: log warning with provider id and error class,
      continue.
   3. If response has no `Item` → continue.
   4. If `Item.get('value', {}).get('S')` is empty → continue.
   5. Otherwise: `os.environ[primary_env_var] = value`. Increment
      applied counter.
7. After the loop, log `dynamodb_key_loader: applied N/<total> provider
   keys from <table>` at INFO level.
8. Set `_applied = True`.

The function never raises. The agent must boot even when DynamoDB is
completely unreachable.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | (none) | Boto3 standard. If absent, loader is skipped. |
| `AWS_SECRET_ACCESS_KEY` | (none) | Boto3 standard. |
| `AWS_REGION` | `us-east-1` | Region for the DynamoDB client. The table lives in `us-east-1`. |
| `HERMES_DYNAMODB_KEY_TABLE` | `chroma-llm-keys` | Override the table name (e.g. for staging). |
| `HERMES_DYNAMODB_KEY_DISABLED` | unset | Skips the loader entirely when set to `1`, `true`, `yes`, or `on` (case-insensitive, leading/trailing whitespace ignored). Empty string and any other value are treated as unset. |

No additional Hermes config files are touched.

## Error handling

- **Missing AWS creds** — silent skip. Local development continues with
  whatever env vars the developer has set.
- **`boto3` not installed** — single warning log, then skip. The Bedrock
  adapter already pulls boto3 in via `pyproject.toml`, so this should not
  occur in a normal install.
- **`GetItem` fails with `AccessDeniedException`** — per-provider warning
  with provider id and error class. Other providers continue.
- **Network / endpoint errors** — same: per-call warning, continue.
- **Item present but `value` attribute missing or empty** — debug log
  ("provider X has no value attribute"), continue. Env var stays as set.
- **Item absent** — debug log, continue. Env var stays as set.
- **Idempotency** — second and subsequent calls in the same process are
  no-ops (flag).
- **Logging hygiene** — under no condition is the secret value, any prefix,
  or any suffix of it logged.

## Testing

Unit tests live in `tests/agent/test_dynamodb_key_loader.py` and use
`botocore.stub.Stubber` against a mocked `dynamodb` client. Coverage:

1. Happy path — one provider, returns expected env var write.
2. Multiple providers — mix of present/absent rows; verifies counter and
   that absent rows leave env vars unchanged.
3. `Item['value']['S']` empty string — env var unchanged.
4. `Item` missing `value` attribute — env var unchanged.
5. `GetItem` raises `ClientError` on one provider — others still applied.
6. `HERMES_DYNAMODB_KEY_DISABLED=1` — no boto3 client constructed.
7. `AWS_ACCESS_KEY_ID` unset — no boto3 client constructed.
8. `boto3` import failure (monkeypatch `sys.modules`) — function returns
   without raising.
9. Idempotency — two consecutive calls hit DynamoDB once.
10. Pre-existing env var is overwritten when DynamoDB returns a value.
11. Pre-existing env var is preserved when DynamoDB row is absent.

No live AWS in CI. Manual smoke is documented but not required for merge:
set the four env vars on a workstation, run
`python -c "from agent.dynamodb_key_loader import apply_dynamodb_overrides; apply_dynamodb_overrides()"`,
and verify the expected env var has been written via `os.environ`.

## Interaction with the existing credential pool

`agent/auxiliary_client.py:_try_openrouter` (and its peers for other
providers) consult the Hermes credential pool *before* the env var:

```
pool_present, entry = _select_pool_entry("openrouter")
if entry:
    or_key = _pool_runtime_api_key(entry)
    ...
or_key = os.getenv("OPENROUTER_API_KEY")
```

If the operator has explicitly configured a pool entry for a provider
(via `hermes auth ...` or by editing the credential pool file), that pool
entry continues to win over the DynamoDB-sourced env var. This is the
correct behavior: pool entries are an explicit operator choice, and the
DynamoDB loader is a deployment-time default. The Railway deployment for
Hermeregildo is not expected to have any pool entries configured, so in
practice the env var (set from DynamoDB) is what drives provider
resolution.

## Open questions

None at spec time.

## Implementation handoff

After this spec is approved, the writing-plans skill produces a step-by-step
implementation plan covering: module skeleton, provider iteration, error
paths, entry-point patches, unit tests, and a Railway deployment checklist
(set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; verify
`OPENROUTER_API_KEY` resolution; verify the bot still answers).
