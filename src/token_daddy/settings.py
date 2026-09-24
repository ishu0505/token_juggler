"""The YAML config schema: accounts, models, deployments, limits, prices.

Secrets never live in the YAML. Accounts name the environment variables that
hold them (`api_key_env: OPENAI_API_KEY`), so the same file can be committed
and shared by every service that draws from the same quotas.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from token_daddy.types import Capability


class Provider(StrEnum):
    OPENAI = "openai"
    DATABRICKS = "databricks"
    BEDROCK = "bedrock"
    GOOGLE_AI_STUDIO = "google_ai_studio"
    VERTEX = "vertex"
    ANTHROPIC = "anthropic"


class Family(StrEnum):
    GPT = "gpt"
    GEMINI = "gemini"
    CLAUDE = "claude"


class WireApi(StrEnum):
    """The protocol a deployment speaks. Deployments with the same wire API
    can stand in for each other under a native SDK client."""

    OPENAI_RESPONSES = "openai_responses"
    GENAI_INTERACTIONS = "genai_interactions"
    GENAI_GENERATE_CONTENT = "genai_generate_content"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class Strategy(StrEnum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_USED = "least_used"


class Window(StrEnum):
    # Continuous refill with a full-size burst - how OpenAI and Anthropic
    # document their limits. A rolling window can see up to ~2x the limit
    # (a full burst plus a full window of refill).
    TOKEN_BUCKET = "token_bucket"
    # At most the limit in EVERY rolling window, for providers that count
    # that way: burst and refill are each half the limit.
    SLIDING = "sliding"


class Reservation(StrEnum):
    # Reserve max_output_tokens up front: can never overshoot the output quota.
    STRICT = "strict"
    # Reserve an estimate and correct afterwards: more throughput, can overshoot.
    ESTIMATE = "estimate"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# name -> (window seconds, what it counts). Every configured limit becomes one
# token bucket that refills continuously over its window.
LIMIT_KINDS: dict[str, tuple[int, Literal["requests", "input", "output", "total"]]] = {
    "rps": (1, "requests"),
    "rpm": (60, "requests"),
    "rph": (3600, "requests"),
    "rpd": (86400, "requests"),
    "input_tpm": (60, "input"),
    "output_tpm": (60, "output"),
    "tpm": (60, "total"),
    "tpd": (86400, "total"),
}


class Limits(_Strict):
    """Quota ceilings. Unset means no limit of that kind; 0 also disables."""

    rps: int | None = None
    rpm: int | None = None
    rph: int | None = None
    rpd: int | None = None
    input_tpm: int | None = None
    output_tpm: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    max_concurrent: int | None = None

    def merged(self, override: Limits | None) -> Limits:
        """`override`'s explicitly-set fields win over ours."""
        if override is None:
            return self
        return self.model_copy(update=override.model_dump(exclude_unset=True))

    def buckets(self) -> dict[str, int]:
        return {
            name: value
            for name in LIMIT_KINDS
            if (value := getattr(self, name)) is not None and value > 0
        }


class Price(_Strict):
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None


class RetryPolicy(_Strict):
    """Retrying the SAME deployment. Off by default: failing over is the plan."""

    enabled: bool = False
    max_attempts: int = 2
    base_delay_seconds: float = 0.5
    jitter: bool = True


class Defaults(_Strict):
    limits: Limits = Field(default_factory=Limits)
    # Fraction of each quota we allow ourselves. The rest absorbs drift between
    # our token estimate and the provider's count.
    headroom: float = Field(default=0.95, gt=0, le=1)
    reservation: Reservation = Reservation.STRICT
    window: Window = Window.TOKEN_BUCKET
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    # How long a call may wait for capacity when every route is full.
    max_wait_seconds: float = 60.0
    request_timeout_seconds: float = 180.0
    # How long a deployment is benched after a real 429 with no Retry-After.
    cooldown_seconds: float = 20.0
    # Benching after auth failures / unknown model ids - a config problem, so long.
    error_cooldown_seconds: float = 300.0
    max_output_tokens: int = 4096


class AccountConfig(_Strict):
    provider: Provider
    api_key_env: str | None = None
    host_env: str | None = None
    token_env: str | None = None
    region: str | None = None
    project_env: str | None = None
    location: str | None = None
    # Overrides the provider's default endpoint.
    base_url: str | None = None
    # Optional account-wide quota, shared by every deployment on this account.
    limits: Limits | None = None


class DeploymentConfig(_Strict):
    account: str
    model_id: str
    api: WireApi | None = None
    limits: Limits | None = None
    price: Price | None = None
    reservation: Reservation | None = None
    window: Window | None = None
    # Capabilities this route lacks even though the model has them.
    exclude_capabilities: list[Capability] = Field(default_factory=list)
    weight: int = Field(default=1, ge=0)
    enabled: bool = True
    # Merged into every request body for this deployment (e.g. {"store": false}).
    extra_params: dict = Field(default_factory=dict)


class RoutingConfig(_Strict):
    strategy: Strategy = Strategy.PRIORITY
    # Models tried, in order, once every deployment of this one is unavailable.
    fallbacks: list[str] = Field(default_factory=list)


class ModelConfig(_Strict):
    family: Family
    capabilities: list[Capability] = Field(
        default_factory=lambda: [Capability.TEXT, Capability.IMAGE, Capability.PDF,
                                 Capability.JSON_SCHEMA]
    )
    max_output_tokens: int | None = None
    price: Price | None = None
    deployments: list[DeploymentConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


class EstimationConfig(_Strict):
    """How payloads turn into a token guess before the call. Tune against real
    `usage` numbers; the reconcile step corrects each call either way."""

    chars_per_token: float = 3.5
    tokens_per_pdf_page: dict[Family, int] = Field(
        default_factory=lambda: {Family.GPT: 1500, Family.GEMINI: 560, Family.CLAUDE: 2000}
    )
    tokens_per_image: dict[Family, int] = Field(
        default_factory=lambda: {Family.GPT: 1100, Family.GEMINI: 1100, Family.CLAUDE: 1600}
    )
    tokens_per_audio_second: dict[Family, int] = Field(
        default_factory=lambda: {Family.GPT: 10, Family.GEMINI: 32, Family.CLAUDE: 32}
    )
    # Used when an audio file's duration can't be read from its header.
    audio_bytes_per_second: int = 16_000
    tokens_per_video_mb: int = 10_000
    # Unknown binary formats: a byte-based guess.
    bytes_per_token_other: int = 4
    safety_multiplier: float = Field(default=1.15, ge=1)
    # Reservation.ESTIMATE only: expected output as a share of input, floored.
    output_ratio: float = 0.4
    output_floor: int = 256


class ProjectConfig(_Strict):
    """A slice of the shared quota. `share` applies to every deployment and
    account; `deployments` overrides it for specific deployment ids."""

    share: float | None = Field(default=None, gt=0, le=1)
    deployments: dict[str, float] = Field(default_factory=dict)


class RedisConfig(_Strict):
    url: str | None = None
    url_env: str | None = "REDIS_URL"

    def resolve(self) -> str | None:
        if self.url:
            return self.url
        return os.environ.get(self.url_env) if self.url_env else None


class Config(_Strict):
    namespace: str = "td"
    redis: RedisConfig = Field(default_factory=RedisConfig)
    defaults: Defaults = Field(default_factory=Defaults)
    accounts: dict[str, AccountConfig]
    models: dict[str, ModelConfig]
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    projects: dict[str, ProjectConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_references(self) -> Config:
        for name, model in self.models.items():
            for dep in model.deployments:
                if dep.account not in self.accounts:
                    raise ValueError(f"model {name!r}: unknown account {dep.account!r}")
            for fallback in model.routing.fallbacks:
                if fallback not in self.models:
                    raise ValueError(f"model {name!r}: unknown fallback {fallback!r}")
                if fallback == name:
                    raise ValueError(f"model {name!r} lists itself as a fallback")
        self._check_project_shares()
        return self

    def _check_project_shares(self) -> None:
        """Shares on one Redis must not promise more than the quota holds."""
        total = sum(p.share or 0 for p in self.projects.values())
        if total > 1 + 1e-9:
            raise ValueError(f"project shares add up to {total:.2f}, over 1.0")
        per_deployment: dict[str, float] = {}
        for project in self.projects.values():
            for dep_id, share in project.deployments.items():
                if not 0 < share <= 1:
                    raise ValueError(f"project share for {dep_id!r} must be in (0, 1]")
        for dep_id in {d for p in self.projects.values() for d in p.deployments}:
            per_deployment[dep_id] = sum(
                p.deployments.get(dep_id, p.share or 0) for p in self.projects.values()
            )
            if per_deployment[dep_id] > 1 + 1e-9:
                raise ValueError(
                    f"project shares for {dep_id!r} add up to "
                    f"{per_deployment[dep_id]:.2f}, over 1.0"
                )


def load_config(path: str | Path) -> Config:
    with open(path) as handle:
        return Config.model_validate(yaml.safe_load(handle))
