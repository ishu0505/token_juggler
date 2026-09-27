# tokenjuggler docs

tokenjuggler gives you one interface to GPT, Gemini and Claude models across several
providers each (OpenAI, Databricks, AWS Bedrock, Google AI Studio, Vertex AI,
Anthropic). It keeps you under every provider's quota, fails over between providers
and fallback models, and tracks tokens, cost and latency.

## Which guide is for you?

| You are... | Setup | Read |
|---|---|---|
| Building **one app** (or a few processes of one app) that calls LLMs | Local YAML file, optional Redis | [single-project.md](single-project.md) |
| A **developer on a team** whose platform team already runs a shared tokenjuggler Redis | Connect with a Redis URL and a project name | [app-developer.md](app-developer.md) |
| The **platform owner/admin** running tokenjuggler for several projects | Central config in Redis, per-project quotas | [platform-admin.md](platform-admin.md) |
| Anyone looking up a config field | - | [configuration.md](configuration.md) |
| Anyone wondering how it works inside | - | [architecture.md](architecture.md) |

Creating a config: `tokenjuggler init` writes a starter file (`minimal`, `full` or
`central`), and `tokenjuggler ui` opens a local web page to edit it with live
validation. You can always edit the YAML by hand instead - it's the same file.

## The two setups at a glance

```
Simple (one project)                       Central (many projects)
─────────────────────                      ──────────────────────────────────────────
 your app                                   service A (project=search)  ─┐
   └─ tokenjuggler                           service B (project=batch)   ─┼─ tokenjuggler
        ├─ reads tokenjuggler.yaml           service C (project=chat)    ─┘      │
        └─ Redis (optional)                                                     ▼
                                                                  shared Redis
                                                                  ├─ config (versioned)
                                                                  ├─ quota buckets
                                                                  └─ usage + cost
                                            admin: tokenjuggler config push tokenjuggler.yaml
```

Both setups use the same package and the same YAML format; the only difference is
where the YAML lives. You can start simple and move to central later without
changing application code beyond one line (`from_config(...)` -> `connect(...)`).

## Core ideas (2 minutes)

* **Model** - what you ask for, e.g. `gpt-5.6-terra`.
* **Deployment** - that model on one provider account, e.g. `gpt-5.6-terra@databricks`.
  Each deployment has its own quota and price.
* **Routing** - a request goes to the first deployment (by the model's strategy) that
  has quota room. A 429, 5xx, timeout or auth error moves it to the next deployment,
  then to the model's `fallbacks`. It never retries the same deployment unless you
  turn retries on.
* **Quota enforcement** - before every call, tokenjuggler reserves the call's requests,
  input tokens and output tokens against the deployment's limits (atomically, in
  Redis). A route without room is never called, so you don't cause provider 429s.
* **Projects** - on a shared Redis, each service passes a project name. Projects can
  get a `cap` (a ceiling) and/or a `reserve` (a guaranteed slice) of each quota.
