# Guide: using a shared tokenjuggler from your service

For developers whose platform team already runs a shared tokenjuggler (a central
Redis with a published config). You don't write any YAML - you need three things
from your platform admin:

1. **The Redis URL**, e.g. `rediss://user:pass@llm-quota.internal:6379/0`
2. **The namespace** (often just `tj`)
3. **Your project name**, e.g. `search-svc` - this is what your quota and usage are
   tracked under. Use the exact name the admin gave you; an unknown name still works
   but only gets the shared pool.

Plus the provider credentials your service is allowed to use (as env vars - the
admin will tell you which, e.g. `DATABRICKS_TOKEN`). The central config names the
variables; the values stay in your service's environment.

## 1. Install

```bash
uv add "tokenjuggler[openai,gemini,anthropic]"   # only the providers your service calls
```

See [the extras table](single-project.md#1-install). Until the first PyPI release:
`uv add "tokenjuggler[all] @ git+ssh://git@github.com/ishu0505/token_juggler.git"`.

## 2. Connect

```python
import tokenjuggler as juggle

tj = await juggle.connect(
    "rediss://user:pass@llm-quota.internal:6379/0",   # or leave out and set REDIS_URL
    namespace="tj",
    project="search-svc",
)
```

`connect()` loads the current config from Redis. Every 30 s it checks for a new
version (one tiny read) and switches to it without a restart - so when the admin
adds a model or changes your quota, you get it automatically. Change the interval
with `refresh_seconds=`, or `0` to pin the version you started with.

Create **one** `TokenJuggler` per process at startup (it holds connection pools) and
`await tj.aclose()` on shutdown. With FastAPI, for example:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    app.state.tj = await juggle.connect(project="search-svc")
    yield
    await app.state.tj.aclose()
```

## 3. Call models

Ask for models by the names in the central config (`tokenjuggler --central check`
lists them):

```python
r = await tj.generate("gpt-5.6-terra", "Classify this ticket: ...", max_output_tokens=200)
r.text, r.deployment, r.usage, r.cost_usd
```

Everything in the single-project guide applies from here - files, structured
output, `thinking`, native SDK clients (`tj.openai()`, `tj.genai()`,
`tj.anthropic()`), and the exceptions to catch. See
[single-project.md, steps 4 and 6](single-project.md#4-call-models).

## 4. What your quota means

The admin may have given your project:

* **a reserve** - a slice of each quota kept for you. Other projects can't use it
  (unless the admin set `lend_idle`, in which case they may borrow it while you're
  idle; when you're busy again it refills for you within about a minute).
* **a cap** - the most your project may use, even when others are idle.
* **nothing** - you share the unreserved pool with everyone else.

See your slice for every model:

```bash
REDIS_URL=... uv run tokenjuggler --central --namespace tj projects
```

When you're out of quota, `generate()` waits up to `max_wait_seconds` (set by the
admin; override per call with `max_wait_seconds=`) and then raises `QuotaExceeded`.
Its message says which limit is full and when capacity returns.

## 5. See your usage

```bash
REDIS_URL=... uv run tokenjuggler --central --project search-svc usage --since 24h
```

or in code, `await tj.usage()` (your project) - requests, errors, tokens, cost and
average latency per deployment.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ConfigError: no config has been pushed to namespace ...` | Wrong namespace or Redis URL, or the admin hasn't published yet |
| `check` shows your model's deployments as `--  env var X is not set` | Your service lacks that provider credential; set it or ask the admin which routes you may use |
| `QuotaExceeded: ... reserved for other projects` | All of that deployment is reserved for others and your project has no slice - ask the admin |
| `QuotaExceeded: ... project search-svc cap ...` | You hit your own cap |
| Frequent `QuotaExceeded` with big `max_output_tokens` | Each call reserves its full output cap up front; request only what you need |
