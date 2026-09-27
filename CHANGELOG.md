# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) (while at 0.x, minor versions may break).

## [0.1.0] - 2026-09-27

First release.

### Added
- One interface (`generate`) to GPT, Gemini and Claude models across OpenAI,
  Databricks, AWS Bedrock, Google AI Studio, Vertex AI and Anthropic, using each
  provider's current API (OpenAI Responses, Gemini Interactions / generateContent,
  Anthropic Messages). Text, images, PDFs, audio and video; structured output;
  reasoning depth.
- Native SDK clients (`openai`, `google-genai`, `anthropic`) with the same limiting,
  tracking and failover underneath.
- Quota enforcement across processes: token buckets for requests, input and output
  tokens per second/minute/hour/day, concurrency, account-wide limits; atomic
  reservation in Redis via Lua, or in memory for a single process.
- Failover to the next deployment on 429/5xx/timeout/auth errors, then to fallback
  models; routing strategies priority, round robin, weighted and least used;
  optional same-route retries.
- Usage and cost tracking per project and deployment, with latency and overhead.
- Central config in Redis with versions, history, rollback, compare-and-set
  publishing and hot reload in every connected service.
- Per-project quotas: caps, reservations and lending idle reservations.
- CLI: `init`, `ui`, `check`, `projects`, `verify`, `usage`, `config push/show/pull/history/rollback`.
- Local web UI for building and validating configs, with the docs built in.
- Optional extras: `openai`, `gemini`, `anthropic`, `ui`, `all`.
