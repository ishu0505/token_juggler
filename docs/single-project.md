# Guide: one project with a local config

For a single application (one or many processes of it) that owns its provider
accounts. Everything is configured in one YAML file inside your repo.

## 1. Install

tokenjuggler isn't on PyPI yet; install it from git:

```bash
uv add git+ssh://git@github.com/ishu0505/token_juggler.git
# or: pip install git+ssh://git@github.com/ishu0505/token_juggler.git
```

Python 3.12+.

## 2. Credentials

Put secrets in environment variables (or a `.env` file - the CLI loads `.env` from
the working directory; in your app, load it yourself, e.g. `python-dotenv`).
The YAML only ever names the variables.

```bash
OPENAI_API_KEY=sk-...
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
BEDROCK_API_KEY=...                    # Bedrock API key (GPT via the OpenAI SDK)
GOOGLE_AI_STUDIO_API_KEY=...
VERTEX_PROJECT_ID=my-gcp-project       # Vertex auth = ADC, or a key via credentials_json_env
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://localhost:6379/0     # optional, see step 5
```

## 3. Write `tokenjuggler.yaml`

Start from `tokenjuggler.yaml.example` in the repo (all Mark 1 models and providers).
A minimal one:

```yaml
namespace: myapp

defaults:
  limits: {rpm: 500, input_tpm: 200000, output_tpm: 20000}  # per deployment
  max_output_tokens: 2048

accounts:
  databricks: {provider: databricks, host_env: DATABRICKS_HOST, token_env: DATABRICKS_TOKEN}
  openai:     {provider: openai, api_key_env: OPENAI_API_KEY}

models:
  gpt-5.6-terra:
    family: gpt
    price: {input_per_mtok: 2.0, output_per_mtok: 12.0}
    deployments:                        # tried in this order (strategy: priority)
      - {account: databricks, model_id: system.ai.gpt-5-6-terra}
      - {account: openai,     model_id: gpt-5.6-terra}
    routing: {fallbacks: [gemini-3.8-flash]}
  # ... gemini-3.8-flash defined the same way
```

Check it - no calls are made:

```bash
uv run tokenjuggler check
```

Every deployment shows `ok` or the reason it's skipped (usually a missing env var).
Then prove each route works with one tiny billed call each:

```bash
uv run tokenjuggler verify              # or: verify -m gpt-5.6-terra
```

## 4. Call models

```python
import tokenjuggler as juggle
from tokenjuggler import Text, File, QuotaExceeded

tj = juggle.from_config("tokenjuggler.yaml", project="myapp")

r = await tj.generate("gpt-5.6-terra", "Summarise our Q3 results in 3 bullets.")
print(r.text)

# Files, structured output, reasoning depth:
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float

r = await tj.generate(
    "gemini-3.8-flash",
    [Text("Extract the invoice"), File.from_path("invoice.pdf")],
    response_schema=Invoice,        # r.parsed is an Invoice
    max_output_tokens=1000,
    thinking="low",                 # minimal | low | medium | high
)
print(r.parsed, r.deployment, r.usage, r.cost_usd, r.timings)

await tj.aclose()                   # on shutdown
```

Not async? `tj.generate_sync(model, parts, ...)` takes the same arguments.

**What comes back** (`Response`): `text`, `parsed`, `usage` (input / output /
reasoning / cached tokens), `cost_usd` (None when no price is configured),
`deployment` and `provider` that served it, `attempts` (every route tried and why it
failed), `timings` (provider vs tokenjuggler time, ms), `truncated` (hit the output cap).

**Errors worth catching:**

| Exception | Meaning | Typical action |
|---|---|---|
| `QuotaExceeded` | every route stayed full longer than `max_wait_seconds` | back off, queue, or raise limits |
| `AllRoutesFailed` | every capable route errored (5xx, auth, ...) | alert; `exc.attempts` says why |
| `NoCapableDeployment` | no route can take this input (e.g. audio to GPT) | pick another model |
| `tokenjuggler.adapters.BadRequest` | the request itself is invalid (400) | fix the request |

## 5. Do you need Redis?

* **One process:** no. Without `REDIS_URL`, limits are enforced in memory.
* **Several processes/containers sharing the same provider accounts:** yes. Point them
  all at one Redis (`REDIS_URL`), otherwise each process enforces the full quota on
  its own and together they overshoot.

```bash
docker compose up -d redis           # the repo's compose file runs one on :6379
```

## 6. Use the provider SDKs you already know (optional)

Existing code written for the official SDKs keeps working, with tokenjuggler's
limiting, tracking and failover underneath:

```python
oai = tj.openai("gpt-5.6-terra")         # openai.AsyncOpenAI
resp = await oai.responses.create(model="gpt-5.6-terra", input="hi")

gem = tj.genai("gemini-3.8-flash")       # google.genai.Client
out = await gem.aio.models.generate_content(model="gemini-3.8-flash", contents="hi")

ant = tj.anthropic("claude-sonnet-5")    # anthropic.AsyncAnthropic
msg = await ant.messages.create(model="claude-sonnet-5", max_tokens=500,
                                messages=[{"role": "user", "content": "hi"}])
```

Failover here stays within one API family: GPT across its providers, Claude across
its providers, Gemini `generate_content` across Vertex and Databricks.

## 7. Watch usage and cost

```bash
uv run tokenjuggler usage --since 24h      # per deployment: requests, errors, tokens, cost, avg latency
```

In code: `await tj.usage()` returns the same rows; `await tj.recent_calls(50)` shows
the latest individual calls. For long-term storage, pass `on_call=` to `TokenJuggler`
- it receives a record for every attempt, which you can write to your own database.

## Tuning checklist

* **Calls refused with `QuotaExceeded` while providers are idle?** Your limits are
  tighter than the real quotas, or `max_output_tokens` is large: with the default
  `reservation: strict`, every call reserves its full `max_output_tokens` up front.
  Lower `max_output_tokens`, or switch that model to `reservation: estimate`.
* **Seeing provider 429s?** Your configured limits are higher than the real ones, or
  something else uses the same account. Lower the limits or `headroom`; if the
  provider counts strictly per rolling window, set `window: sliding`.
* **Latency budget:** tokenjuggler adds ~1-4 ms per call (two Redis round trips).
  Keep Redis in the same region as your app.
