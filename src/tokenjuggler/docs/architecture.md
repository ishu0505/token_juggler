# Architecture

tokenjuggler is a Python library. It runs **inside your service's process** - there
is no proxy or gateway between your code and the providers. The only shared piece is
an optional Redis, which holds quota counters (and, in the central setup, the config).

## The pieces

```
 your service (Python)
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  your code                                                                  │
 │     │  tj.generate("gpt-5.6-terra", ...)      or   tj.openai().responses... │
 │     ▼                                              (native SDK clients)     │
 │  ┌───────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
 │  │  Router   │──▶│  Limiter    │   │  Adapters    │   │  Tracker         │  │
 │  │ strategy, │   │ reserve /   │   │ OpenAI Resp. │   │ tokens, cost,    │  │
 │  │ failover, │   │ settle quota│   │ Gemini       │   │ latency per call │  │
 │  │ fallbacks │   │             │   │ Anthropic    │   │                  │  │
 │  └─────┬─────┘   └──────┬──────┘   └──────┬───────┘   └────────┬─────────┘  │
 │        │  config: Registry (models, deployments, limits, prices, projects)  │
 └────────┼────────────────┼─────────────────┼────────────────────┼────────────┘
          │                │ 2 round trips   │ HTTPS              │ same round trip
          │                ▼ per call        ▼                    ▼ as the settle
          │        ┌───────────────┐   ┌───────────────────────────────────────┐
          │        │    Redis      │   │ Providers                             │
          │        │ quota buckets │   │ OpenAI · Databricks · Bedrock         │
          │        │ usage rollups │   │ AI Studio · Vertex AI · Anthropic     │
          └───────▶│ config (opt.) │   └───────────────────────────────────────┘
                   └───────────────┘
```

* **Registry** - the config, resolved: every *deployment* (a model on one provider
  account) with its endpoint, credentials (from env vars), limits and price.
* **Router** - picks the deployment for a request, fails over, falls back.
* **Limiter** - reserves each call's cost against every quota before the call and
  corrects it afterwards. Two Lua scripts in Redis (or the same logic in memory).
* **Adapters** - one per wire API (OpenAI Responses, Gemini Interactions /
  generateContent, Anthropic Messages); translate one request shape to each SDK.
* **Tracker** - records every attempt: tokens, cost, latency, outcome.

## One request, step by step

```
generate("gpt-5.6-terra", [text, pdf])
 │
 ├─ 1. which deployments can carry this?   (pdf needs "pdf" capability)
 │       gpt-5.6-terra@databricks, gpt-5.6-terra@openai
 │
 ├─ 2. estimate cost: input ≈ text/3.5 + pages×1500, output = max_output_tokens
 │
 ├─ 3. acquire (1 Redis round trip, atomic):
 │       first deployment, in strategy order, whose buckets ALL have room
 │       (rpm, input_tpm, output_tpm, ... + account + project cap/reserve)
 │       ─ none has room? → wait until one does (max_wait_seconds) or QuotaExceeded
 │
 ├─ 4. call the provider through its adapter
 │       ─ 429 → bench this deployment (Retry-After), go back to 3 without it
 │       ─ 5xx / timeout / auth error → same, next deployment
 │       ─ 400 → raise: every provider would reject it the same way
 │       ─ every deployment failed → the model's fallbacks, in order
 │
 └─ 5. settle (1 Redis round trip): refund unused reservation, record usage + cost
```

## Quota buckets

Every limit (`rpm`, `input_tpm`, `output_tpm`, ...) is a token bucket that refills
continuously - no minute boundary to burst across. A call spends from every bucket
that applies to it, all at once or not at all:

```
deployment gpt-5.6-terra@openai          ← the provider's real quota (× headroom)
 ├─ account openai                        ← optional account-wide limits
 ├─ project cap: batch-etl ≤ 30%          ← optional ceiling per project
 └─ one quota source (projects with reservations):
       own reserve → shared pool → reserves lent while idle
```

## The two ways to deploy it

```
Simple (one project)                      Central (many projects)
────────────────────                      ────────────────────────────────────────
your app + tokenjuggler.yaml              admin ── tokenjuggler config push ──┐
        │                                                                      ▼
        └── Redis (optional:                 service A (project=search) ─┐   Redis
            only needed when several         service B (project=batch)  ─┼─▶ config v7
            processes share accounts)        service C (project=chat)   ─┘   buckets
                                                                             usage
```

In the central setup every service loads the config from Redis at startup and checks
for a new version every 30 s, so a change the admin publishes reaches every service
without a redeploy.

## What lives where

| Thing | Where | Notes |
|---|---|---|
| Config | `tokenjuggler.yaml`, or Redis (central) | never contains secrets |
| Credentials | each service's environment | the config only names the env vars |
| Quota counters | Redis, or process memory | tiny, expire on their own |
| Usage rollups | Redis | per minute (2 h), hour (8 d), day (90 d) |
| Long-term call records | your `on_call` sink | e.g. a database or metrics system |
