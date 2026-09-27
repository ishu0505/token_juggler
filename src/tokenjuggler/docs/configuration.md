# Configuration reference

One YAML format for both setups: a local `tokenjuggler.yaml`, or the central config
published with `tokenjuggler config push`. Unknown keys are rejected, so typos fail
loudly. Secrets never go in the file - fields ending in `_env` name environment
variables.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `namespace` | `tj` | Prefix for every Redis key. Different namespaces never share quotas, usage or config. |
| `redis.url` / `redis.url_env` | `url_env: REDIS_URL` | Where the limiter keeps state (local-config setup). Unset -> in-memory, per process. Ignored by `connect()`, which already has a URL. |
| `defaults` | see below | Applied to every deployment unless overridden. |
| `accounts` | required | Provider credentials, by name. |
| `models` | required | Logical models and their deployments. |
| `estimation` | see below | How payloads become a token estimate before the call. |
| `projects` | `{}` | Per-project caps and reservations. |

## `defaults`

| Key | Default | Meaning |
|---|---|---|
| `limits` | none | Quota per deployment - see [Limits](#limits). |
| `headroom` | `0.95` | Fraction of every quota tokenjuggler lets itself use. |
| `reservation` | `strict` | `strict`: reserve `max_output_tokens` before a call (can't overshoot output quotas). `estimate`: reserve a guess (more throughput, can overshoot). |
| `window` | `token_bucket` | `token_bucket`: continuous refill with a full-size burst (how OpenAI/Anthropic describe limits; a rolling window may see up to ~2x). `sliding`: never above the limit in any rolling window, at half the throughput. |
| `max_output_tokens` | `4096` | Output cap sent with each call when the caller doesn't set one. |
| `max_wait_seconds` | `60` | How long a call waits when every route is full, before `QuotaExceeded`. |
| `request_timeout_seconds` | `180` | Per provider call. |
| `cooldown_seconds` | `20` | How long a deployment is benched after a real 429 without `Retry-After`. |
| `error_cooldown_seconds` | `300` | Bench time after auth/unknown-model errors. |
| `retry.enabled` | `false` | Retry the **same** deployment on 5xx/timeouts before failing over. |
| `retry.max_attempts` / `base_delay_seconds` / `jitter` | `2` / `0.5` / `true` | Retry shape when enabled. |

## Limits

Usable in `defaults.limits`, a deployment's `limits`, and an account's `limits`
(account limits are a separate bucket shared by all deployments on that account).
Unset or `0` = no limit of that kind.

| Key | Counts | Window |
|---|---|---|
| `rps`, `rpm`, `rph`, `rpd` | requests | second, minute, hour, day |
| `input_tpm` | input tokens | minute |
| `output_tpm` | output tokens (incl. reasoning) | minute |
| `tpm`, `tpd` | input + output tokens | minute, day |
| `max_concurrent` | requests in flight | - |

A deployment's `limits` override `defaults.limits` key by key.

## `accounts.<name>`

| Key | Used by | Meaning |
|---|---|---|
| `provider` | all | `openai`, `databricks`, `bedrock`, `google_ai_studio`, `vertex`, `anthropic` |
| `api_key_env` | openai, bedrock, google_ai_studio, anthropic | Env var with the API key (Bedrock: a Bedrock API key) |
| `host_env`, `token_env` | databricks | Workspace URL and token env vars |
| `region` | bedrock | e.g. `us-east-1` |
| `project_env` | vertex | Env var with the GCP project (optional when the key has one) |
| `credentials_json_env` | vertex | Env var holding a service-account JSON; omit to use ADC |
| `location` | vertex | Default `global` |
| `base_url` | any | Override the endpoint |
| `limits` | any | Account-wide quota across all its deployments |

Default wire API per provider and family: GPT on openai/databricks/bedrock ->
OpenAI Responses API; Gemini on google_ai_studio -> Interactions API, on
vertex/databricks -> generateContent; Claude on anthropic/databricks -> Messages API.

## `models.<name>`

| Key | Default | Meaning |
|---|---|---|
| `family` | required | `gpt`, `gemini`, `claude` (drives request format and estimation) |
| `capabilities` | text, image, pdf, json_schema | Also `audio`, `video`, `reasoning`. Deployments that lack a capability a request needs are skipped. |
| `max_output_tokens` | `defaults.max_output_tokens` | |
| `price` | none | `{input_per_mtok, output_per_mtok, cached_input_per_mtok}` USD per 1M tokens. Without it, cost is reported as unknown, never $0. |
| `deployments` | required | List, in priority order - see below |
| `routing.strategy` | `priority` | `priority` (first with room), `round_robin`, `weighted` (by deployment `weight`), `least_used` |
| `routing.fallbacks` | `[]` | Other models to try, in order, when every deployment of this one is unavailable |

### `deployments[]`

| Key | Default | Meaning |
|---|---|---|
| `account` | required | An `accounts` name. The deployment's id is `<model>@<account>`. |
| `model_id` | required | The provider's id, e.g. `system.ai.gpt-5-6-terra`, `global.openai.gpt-5.6-sol` |
| `api` | per provider | Override the wire API |
| `limits` | `defaults.limits` | Overrides, key by key |
| `price` | model's price | E.g. `{input_per_mtok: 0, output_per_mtok: 0}` for a free tier |
| `reservation`, `window` | defaults | Per-route override |
| `exclude_capabilities` | `[]` | Capabilities this route lacks, e.g. `[audio]` |
| `weight` | `1` | For `weighted` routing; `0` = last resort |
| `enabled` | `true` | |
| `extra_params` | `{}` | Merged into every request body, e.g. `{store: false}` for Bedrock |

## `estimation`

| Key | Default |
|---|---|
| `chars_per_token` | `3.5` |
| `tokens_per_pdf_page` | gpt 1500, gemini 560, claude 2000 |
| `tokens_per_image` | gpt 1100, gemini 1100, claude 1600 |
| `tokens_per_audio_second` | gpt 10, gemini 32, claude 32 |
| `audio_bytes_per_second` | `16000` (when the duration isn't in the file header) |
| `tokens_per_video_mb` | `10000` |
| `safety_multiplier` | `1.15` |
| `output_ratio`, `output_floor` | `0.4`, `256` (`reservation: estimate` only) |

Estimates only size the up-front reservation; the provider's real counts replace
them when the call returns.

## `projects.<name>`

Fractions of each deployment's limits. See the
[admin guide](platform-admin.md#choosing-quota-modes) for how to choose.

| Key | Meaning |
|---|---|
| `cap` (old name: `share`) | At most this fraction |
| `reserve` | This fraction is kept for the project |
| `lend_idle` | Others may borrow the reserve while it's unused |
| `deployments.<model@account>` | Override for one deployment: a number (a cap) or `{cap, reserve, lend_idle}` |

Validation: reserves per deployment sum to at most 1.0; `reserve <= cap`; project
deployment ids must exist.
