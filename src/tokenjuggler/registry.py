"""Turns a validated `Config` into ready-to-route deployments.

This is where env vars are read, default endpoints are filled in and each
deployment's limits are merged from defaults. A deployment whose credentials
are missing is kept out of routing - with the reason recorded, so
`tokenjuggler config check` can say why - rather than failing at call time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from tokenjuggler.settings import (
    AccountConfig,
    Config,
    Family,
    Limits,
    Price,
    ProjectQuota,
    Provider,
    Reservation,
    RoutingConfig,
    Window,
    WireApi,
)
from tokenjuggler.types import Capability
from tokenjuggler.utils.logger import get_logger

log = get_logger("tokenjuggler.registry")

# Which wire API a provider speaks for each model family, when not overridden.
DEFAULT_API: dict[tuple[Provider, Family], WireApi] = {
    (Provider.OPENAI, Family.GPT): WireApi.OPENAI_RESPONSES,
    (Provider.DATABRICKS, Family.GPT): WireApi.OPENAI_RESPONSES,
    (Provider.DATABRICKS, Family.GEMINI): WireApi.GENAI_GENERATE_CONTENT,
    (Provider.DATABRICKS, Family.CLAUDE): WireApi.ANTHROPIC_MESSAGES,
    (Provider.BEDROCK, Family.GPT): WireApi.OPENAI_RESPONSES,
    # The Interactions API is Google's recommended surface, but only on AI
    # Studio so far; Vertex still serves generateContent only.
    (Provider.GOOGLE_AI_STUDIO, Family.GEMINI): WireApi.GENAI_INTERACTIONS,
    (Provider.VERTEX, Family.GEMINI): WireApi.GENAI_GENERATE_CONTENT,
    (Provider.ANTHROPIC, Family.CLAUDE): WireApi.ANTHROPIC_MESSAGES,
}

_DATABRICKS_GATEWAY_PATH = {
    WireApi.OPENAI_RESPONSES: "/ai-gateway/openai/v1",
    WireApi.GENAI_GENERATE_CONTENT: "/ai-gateway/gemini",
    WireApi.ANTHROPIC_MESSAGES: "/ai-gateway/anthropic",
}


class CredentialsMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    api_key: str | None = None
    # Sent as `Authorization: Bearer` (Databricks gateway).
    bearer_token: str | None = None
    project: str | None = None
    location: str | None = None
    # Vertex service-account key, when not using Application Default Credentials.
    service_account_info: dict | None = field(default=None, repr=False, hash=False)


@dataclass(frozen=True)
class Deployment:
    id: str  # "<model>@<account>"
    model: str
    family: Family
    account: str
    provider: Provider
    api: WireApi
    model_id: str
    base_url: str | None
    credentials: Credentials = field(repr=False)
    limits: Limits
    account_limits: Limits | None
    price: Price | None
    reservation: Reservation
    window: Window
    capabilities: frozenset[Capability]
    max_output_tokens: int
    weight: int = 1
    extra_params: dict = field(default_factory=dict)

    def supports(self, required: set[Capability]) -> bool:
        return required <= self.capabilities


@dataclass(frozen=True)
class Model:
    name: str
    family: Family
    capabilities: frozenset[Capability]
    deployments: tuple[Deployment, ...]
    routing: RoutingConfig
    max_output_tokens: int


class Registry:
    def __init__(self, config: Config, *, environ: dict[str, str] | None = None):
        self.config = config
        self._env = os.environ if environ is None else environ
        self.models: dict[str, Model] = {}
        self.deployments: dict[str, Deployment] = {}
        # deployment id -> why it isn't routable.
        self.unavailable: dict[str, str] = {}
        self._build()

    def model(self, name: str) -> Model:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(
                f"unknown model {name!r}; configured: {', '.join(sorted(self.models))}"
            ) from None

    def project_quota(self, project: str | None, deployment_id: str) -> ProjectQuota | None:
        """This project's cap/reserve on a deployment, or None when it has none."""
        if project is None or project not in self.config.projects:
            return None
        return self.config.projects[project].quota_for(deployment_id)

    def reservations(self, deployment_id: str) -> dict[str, ProjectQuota]:
        """Every project holding a reserve on this deployment."""
        out = {}
        for name, cfg in self.config.projects.items():
            quota = cfg.quota_for(deployment_id)
            if quota.reserve:
                out[name] = quota
        return out

    def _build(self) -> None:
        defaults = self.config.defaults
        for name, model_cfg in self.config.models.items():
            model_caps = frozenset(model_cfg.capabilities) | {Capability.TEXT}
            max_out = model_cfg.max_output_tokens or defaults.max_output_tokens
            built: list[Deployment] = []
            for dep_cfg in model_cfg.deployments:
                dep_id = f"{name}@{dep_cfg.account}"
                if not dep_cfg.enabled:
                    self.unavailable[dep_id] = "disabled in config"
                    continue
                account = self.config.accounts[dep_cfg.account]
                api = dep_cfg.api or DEFAULT_API.get((account.provider, model_cfg.family))
                if api is None:
                    self.unavailable[dep_id] = (
                        f"{account.provider} has no default API for {model_cfg.family} "
                        "models; set `api:` on the deployment"
                    )
                    continue
                try:
                    creds = self._credentials(account)
                    base_url = self._base_url(account, api)
                except CredentialsMissing as exc:
                    self.unavailable[dep_id] = str(exc)
                    continue
                dep = Deployment(
                    id=dep_id,
                    model=name,
                    family=model_cfg.family,
                    account=dep_cfg.account,
                    provider=account.provider,
                    api=api,
                    model_id=dep_cfg.model_id,
                    base_url=base_url,
                    credentials=creds,
                    limits=defaults.limits.merged(dep_cfg.limits),
                    account_limits=account.limits,
                    price=dep_cfg.price or model_cfg.price,
                    reservation=dep_cfg.reservation or defaults.reservation,
                    window=dep_cfg.window or defaults.window,
                    capabilities=model_caps - set(dep_cfg.exclude_capabilities),
                    max_output_tokens=max_out,
                    weight=dep_cfg.weight,
                    extra_params=dict(dep_cfg.extra_params),
                )
                built.append(dep)
                self.deployments[dep_id] = dep
            self.models[name] = Model(
                name=name,
                family=model_cfg.family,
                capabilities=model_caps,
                deployments=tuple(built),
                routing=model_cfg.routing,
                max_output_tokens=max_out,
            )
        if self.unavailable:
            log.warning(
                "%d deployment(s) not routable (run `tokenjuggler check` for details): %s",
                len(self.unavailable), ", ".join(sorted(self.unavailable)),
            )

    def _env_value(self, var: str | None, what: str, account: AccountConfig) -> str:
        if not var:
            raise CredentialsMissing(f"{account.provider} account needs `{what}` set")
        value = self._env.get(var)
        if not value:
            raise CredentialsMissing(f"env var {var} is not set")
        return value

    def _credentials(self, account: AccountConfig) -> Credentials:
        match account.provider:
            case Provider.OPENAI | Provider.BEDROCK | Provider.GOOGLE_AI_STUDIO | Provider.ANTHROPIC:
                return Credentials(api_key=self._env_value(account.api_key_env, "api_key_env", account))
            case Provider.DATABRICKS:
                return Credentials(bearer_token=self._env_value(account.token_env, "token_env", account))
            case Provider.VERTEX:
                # Auth is a service-account JSON from the env when configured,
                # else Application Default Credentials.
                info = None
                if account.credentials_json_env:
                    import json

                    info = json.loads(
                        self._env_value(account.credentials_json_env, "credentials_json_env", account)
                    )
                if account.project_env:
                    project = self._env_value(account.project_env, "project_env", account)
                elif info and info.get("project_id"):
                    project = info["project_id"]
                else:
                    raise CredentialsMissing("vertex account needs `project_env` or a key with project_id")
                return Credentials(
                    project=project,
                    location=account.location or "global",
                    service_account_info=info,
                )

    def _base_url(self, account: AccountConfig, api: WireApi) -> str | None:
        if account.base_url:
            return account.base_url.rstrip("/")
        match account.provider:
            case Provider.OPENAI:
                return "https://api.openai.com/v1"
            case Provider.DATABRICKS:
                host = self._env_value(account.host_env, "host_env", account).rstrip("/")
                if not host.startswith("http"):
                    host = f"https://{host}"
                return host + _DATABRICKS_GATEWAY_PATH[api]
            case Provider.BEDROCK:
                if not account.region:
                    raise CredentialsMissing("bedrock account needs `region`")
                return f"https://bedrock-runtime.{account.region}.amazonaws.com/openai/v1"
            case _:
                return None  # the SDK's own default endpoint
