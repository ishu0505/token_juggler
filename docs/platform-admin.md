# Guide: running tokenjuggler for several projects

For the person or team that owns the provider accounts and wants several services
to share them fairly: one Redis, one central config, per-project quotas.

## How it fits together

* **Redis** holds three things per namespace: the published config (versioned),
  the quota buckets every call reserves against, and usage/cost rollups.
* **Services** install the package and call
  `await TokenJuggler.connect(redis_url, namespace=..., project=...)`. They load the
  config from Redis and follow new versions automatically (checked every 30 s).
* **You** edit one YAML file and publish it with `tokenjuggler config push`. Nothing
  needs redeploying.
* Credentials are never stored in Redis. The config names env vars; each service
  provides the values.

## 1. Run Redis

Any Redis 6.2+ or Valkey works: managed (ElastiCache, Memorystore, Upstash, ...)
or self-hosted. Requirements:

* **Close to the services** - every LLM call makes two round trips; same region
  keeps that at ~1 ms.
* **Reachable from every service**, with auth and TLS (`rediss://user:pass@host:6379/0`).
* **Standalone or a single primary.** Redis Cluster works only because all keys of a
  namespace share one hash slot; there's no sharding across nodes yet.
* **Small.** Data is counters with TTLs plus the config; losing it resets counters
  (brief risk of overshooting a quota) and loses usage history - keep a sink via
  `on_call` if you need long-term records. Persistence is nice to have, not required.

## 2. Write the config

Start from `tokenjuggler.yaml.example`. On top of what a single project needs
(accounts, models, deployments, limits, prices - see [configuration.md](configuration.md)),
add your projects:

```yaml
namespace: tj                    # services connect with namespace="tj"

projects:
  search-svc:                    # user-facing: guaranteed 40%, may be borrowed while idle
    reserve: 0.4
    lend_idle: true
  chat-svc:
    reserve: 0.2                 # guaranteed 20%, never lent out
  batch-etl:
    cap: 0.3                     # background jobs: at most 30%, nothing guaranteed
    deployments:
      gpt-5.6-terra@openai: {cap: 0.1}   # tighter on one expensive route
```

### Choosing quota modes

Fractions apply to **each deployment's** limits (rpm, tpm, ...), per deployment.

| Setting | Guarantees | Cost |
|---|---|---|
| `cap: X` | the project never uses more than X | nothing is guaranteed to it |
| `reserve: X` | X is kept for the project; others can't take it | idle reserve is wasted |
| `reserve: X` + `lend_idle: true` | X is the project's first; others borrow it only when it's idle | after others borrow it, the owner waits up to ~1 minute for it to refill |
| (not listed) | nothing | draws only from the shared pool |

How a call finds quota: its own **reserve** first, then the **shared pool**
(1 - sum of reserves), then reserves that are **lent**. Every call also checks the
deployment's overall limit and the project's cap, so the provider's real quota is
never exceeded however the slices are set.

Rules the config validator enforces: reserves on one deployment add up to at most
1.0; a reserve can't exceed the same project's cap; caps may add up to more than 1.0
(they're ceilings, overcommitting is fine). If reserves add up to exactly 1.0 there's
no shared pool, and unlisted projects are refused on that deployment.

Rule of thumb: **reserve** for latency-sensitive, user-facing services; **cap** for
batch and experiments; `lend_idle: true` unless a project must have its full slice
instantly at any moment.

## 3. Validate and publish

```bash
export REDIS_URL=rediss://user:pass@llm-quota.internal:6379/0

uv run tokenjuggler --config tokenjuggler.yaml check        # routes + missing credentials (local)
uv run tokenjuggler --config tokenjuggler.yaml projects     # how each quota is split
uv run tokenjuggler config push tokenjuggler.yaml           # validate + publish as a new version
```

`push` validates the whole file first - a config that doesn't parse, references an
unknown account or model, or over-reserves a deployment is never published. The
namespace comes from the file.

Keep the YAML in git and publish from CI or by hand. To avoid two admins
overwriting each other:

```bash
uv run tokenjuggler config show                              # note the version, e.g. 7
uv run tokenjuggler config push tokenjuggler.yaml --expect-version 7
# -> refused if someone published v8 in the meantime
```

Other commands (add `--namespace X` if not `tj`):

```bash
uv run tokenjuggler config pull -o current.yaml    # download what's live
uv run tokenjuggler config history                 # last 20 versions: who, when, hash
uv run tokenjuggler config rollback 6              # re-publish v6 as the newest version
```

Services switch to a new version within their refresh interval (30 s by default).
A service that finds an invalid version in Redis keeps running on its last good one
and logs an error.

## 4. Onboard a project

1. Add it under `projects:` (or don't, and it shares the pool) and push.
2. Give the team: Redis URL, namespace, project name, and which provider
   credentials they need (the `*_env` names in `accounts:`).
3. Point them to [app-developer.md](app-developer.md).

Every service can verify its own view with
`tokenjuggler --central --project <name> check`.

## 5. Monitor

```bash
uv run tokenjuggler --central usage --all --since 24h
```

Requests, errors, tokens, cost and average latency per project and deployment,
from the Redis rollups (kept 2 h per minute, 8 days per hour, 90 days per day).

What to watch:

* **errors > 0 with `rate_limited`** - a provider returned a real 429, so its real
  limit is lower than configured (or something outside tokenjuggler shares the
  account). Lower that deployment's limits or `headroom`.
* **many `QuotaExceeded` in services while `usage` shows headroom** - reservations
  too strict or `max_output_tokens` too large under `reservation: strict`.
* **unpriced (`*`) in cost** - add `price:` to those models.

For dashboards or long-term records, have services pass `on_call=` (every attempt's
`CallRecord`: project, deployment, tokens, cost, latency, outcome) and ship it to your
metrics or warehouse.

## 6. Change management notes

* Removing a model or deployment affects services on their next refresh: calls
  already running finish; later calls for a removed model raise `KeyError`.
* Changing `namespace` means a new, empty set of buckets and a separate config -
  services must be moved to it explicitly.
* The simple single-project setup (`TokenJuggler.from_config(...)`) remains fully
  supported; you can migrate projects one at a time.
