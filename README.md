# token_daddy

One interface to GPT, Gemini and Claude models across several providers each,
with shared quota enforcement in Redis, automatic failover and cost tracking.

| Family | Providers (wire API) |
|---|---|
| GPT | OpenAI Platform, Databricks, AWS Bedrock (OpenAI Responses API) |
| Gemini | AI Studio (Interactions API), Vertex AI + Databricks (generateContent) |
| Claude | Claude Platform, Databricks (Anthropic Messages API) |

## How it works

* **Model** - what you ask for (`gpt-5.6-sol`). **Deployment** - one route to
  it on one account (`gpt-5.6-sol@bedrock`), with its own quota and price.
* Before every call, the limiter reserves the call's cost on the best
  deployment that has room - request count, input tokens, and output tokens
  (by default the full `max_output_tokens`, so output can never overshoot).
  Route choice and reservation are one atomic Lua script in Redis.
* A 429, 5xx, timeout or route error moves the call to the next deployment,
  then to the model's `fallbacks`. It never retries the same route unless
  `defaults.retry.enabled` is on. A 429 benches the route for its
  `Retry-After`; with the `priority` strategy traffic returns to it as soon as
  it has room again.
* When every route is merely full, the call waits for the first one to have
  room (`max_wait_seconds`).
* Every attempt is recorded - tokens, cost, latency, outcome - per project
  and deployment.

## Setup

```bash
cp token_daddy.yaml.example token_daddy.yaml   # models, accounts, limits, prices
cp .env.example .env                           # credentials
docker compose up -d redis
uv run token-daddy check                       # which deployments are routable
uv run token-daddy verify -m gpt-5.4           # one tiny live (billed) call each
```

## Unified interface

```python
from token_daddy import TokenDaddy, Text, File

td = TokenDaddy.from_config("token_daddy.yaml", project="search-svc")

r = await td.generate(
    "gpt-5.6-sol",
    [Text("Summarise this"), File.from_path("report.pdf")],
    response_schema=Summary,          # pydantic model -> r.parsed
    max_output_tokens=2000,
    thinking="low",
)
r.text, r.parsed, r.usage, r.cost_usd, r.deployment, r.attempts

await td.usage()                      # tokens + cost per deployment, this project
```

Payloads can mix text, images, PDFs, audio and video; deployments that can't
carry a part (audio to GPT, say) are skipped.

## Native SDKs

The real `openai`, `google-genai` and `anthropic` clients, routed the same way:

```python
oai = td.openai("gpt-5.6-sol")
await oai.responses.create(model="gpt-5.6-sol", input="hi")

gem = td.genai("gemini-3.8-flash")
await gem.aio.models.generate_content(model="gemini-3.8-flash", contents="hi")

ant = td.anthropic("claude-opus-5.5")
await ant.messages.create(model="claude-opus-5.5", max_tokens=1000,
                          messages=[{"role": "user", "content": "hi"}])
```

Failover is between deployments that speak the same wire API: GPT across all
three providers, Claude across both, Gemini generateContent across Vertex and
Databricks. Streaming passes through, settled at the reserved amount.

## Sharing quotas across services and projects

* **One service**: leave `REDIS_URL` unset - limits are enforced in-process.
* **Many services, one quota**: point them all at the same Redis. Each passes
  its own `project=`, which tags usage.
* **Dedicated quotas per project** on a shared Redis: give projects a `share`
  in the YAML (`search-svc: {share: 0.6}`); a project is then capped at its
  slice while also counting against the deployment's whole quota.
* **Fully separate**: a separate Redis (or a different `namespace`) per project.

## Limits: token bucket vs sliding window

`defaults.window` (or per deployment) picks how a quota is interpreted:

* `token_bucket` (default) - continuous refill with a full-size burst, which
  is how OpenAI and Anthropic describe their limits. Long-run rate never
  exceeds the limit, but a rolling window can see up to ~2x it (a full burst
  plus a window of refill).
* `sliding` - never more than the limit in *any* rolling window, for
  providers that count that way, at the cost of half the throughput.

`headroom` (default 0.95) keeps a margin under every quota either way.
