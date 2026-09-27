# tokenjuggler

One interface to GPT, Gemini and Claude models across several providers each,
with shared quota enforcement in Redis, automatic failover and cost tracking.

| Family | Providers (wire API) |
|---|---|
| GPT | OpenAI Platform, Databricks, AWS Bedrock (OpenAI Responses API) |
| Gemini | AI Studio (Interactions API), Vertex AI + Databricks (generateContent) |
| Claude | Claude Platform, Databricks (Anthropic Messages API) |

**Docs:** [which guide is for you](src/tokenjuggler/docs/README.md) - [one project](src/tokenjuggler/docs/single-project.md) -
[developer on a shared setup](src/tokenjuggler/docs/app-developer.md) - [platform admin](src/tokenjuggler/docs/platform-admin.md) -
[config reference](src/tokenjuggler/docs/configuration.md)

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

## Install

```bash
uv add "tokenjuggler[openai,gemini,anthropic]"   # core + the provider SDKs you use
uv add --dev "tokenjuggler[ui]"                  # optional: the config web UI
```

Extras: `openai`, `gemini`, `anthropic`, `ui`, `all`. Not on PyPI yet - until then:
`uv add "tokenjuggler[all] @ git+ssh://git@github.com/ishu0505/token_juggler.git"`.
Run `tokenjuggler` with no arguments for a status summary and next steps.

## Setup

```bash
uv run tokenjuggler init -t minimal             # starter tokenjuggler.yaml (minimal | full | central)
uv run tokenjuggler ui                          # edit it in a local web page - or edit the YAML directly
cp .env.example .env                            # credentials
docker run -d -p 6379:6379 redis:7-alpine      # optional: shared limits across processes
uv run tokenjuggler check                       # which deployments are routable
uv run tokenjuggler verify -m gpt-5.6-terra     # one tiny live (billed) call each
```

The config is one YAML file either way: the UI (`tokenjuggler ui`) is a form over
that file with live validation, and anything it writes you can keep editing by hand.
`tokenjuggler.yaml.example` shows every Mark 1 model on every provider.

## Unified interface

```python
import tokenjuggler as juggle
from tokenjuggler import Text, File

tj = juggle.from_config("tokenjuggler.yaml", project="search-svc")
# or, with a central config in Redis:  tj = await juggle.connect(redis_url, project="search-svc")

r = await tj.generate(
    "gpt-5.6-sol",
    [Text("Summarise this"), File.from_path("report.pdf")],
    response_schema=Summary,          # pydantic model -> r.parsed
    max_output_tokens=2000,
    thinking="low",
)
r.text, r.parsed, r.usage, r.cost_usd, r.deployment, r.attempts

await tj.usage()                      # tokens + cost per deployment, this project
```

Payloads can mix text, images, PDFs, audio and video; deployments that can't
carry a part (audio to GPT, say) are skipped.

## Native SDKs

The real `openai`, `google-genai` and `anthropic` clients, routed the same way:

```python
oai = tj.openai("gpt-5.6-sol")
await oai.responses.create(model="gpt-5.6-sol", input="hi")

gem = tj.genai("gemini-3.8-flash")
await gem.aio.models.generate_content(model="gemini-3.8-flash", contents="hi")

ant = tj.anthropic("claude-opus-5.5")
await ant.messages.create(model="claude-opus-5.5", max_tokens=1000,
                          messages=[{"role": "user", "content": "hi"}])
```

Failover is between deployments that speak the same wire API: GPT across all
three providers, Claude across both, Gemini generateContent across Vertex and
Databricks. Streaming passes through, settled at the reserved amount.

## Sharing quotas across services and projects

* **One service**: a local `tokenjuggler.yaml`; leave `REDIS_URL` unset for in-process limits.
* **Several processes of one app**: same YAML, one Redis for all of them.
* **Many projects on shared accounts**: an admin publishes the config to a central
  Redis (`tokenjuggler config push`); services call
  `await TokenJuggler.connect(redis_url, project="...")` and follow new versions
  automatically. Projects get a `cap` (ceiling) and/or a `reserve` (guaranteed slice,
  optionally lent out while idle). See [the admin guide](src/tokenjuggler/docs/platform-admin.md).

## Limits: token bucket vs sliding window

`defaults.window` (or per deployment) picks how a quota is interpreted:

* `token_bucket` (default) - continuous refill with a full-size burst, which
  is how OpenAI and Anthropic describe their limits. Long-run rate never
  exceeds the limit, but a rolling window can see up to ~2x it (a full burst
  plus a window of refill).
* `sliding` - never more than the limit in *any* rolling window, for
  providers that count that way, at the cost of half the throughput.

`headroom` (default 0.95) keeps a margin under every quota either way.
