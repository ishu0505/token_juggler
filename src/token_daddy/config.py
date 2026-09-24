from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Redis (token/RPM/TPM tracking, shared across apps using the same pool) ---
    # Unset (empty) falls back to a per-process in-memory limiter - see
    # `token_daddy.llm.limits.InProcessRedis`.
    redis_url: str | None = "redis://localhost:6379/0"

    # --- OpenAI Platform (platform.openai.com) ---
    openai_api_key: str | None = None
    openai_org_id: str | None = None
    openai_project_id: str | None = None

    # --- Anthropic Platform (console.anthropic.com) ---
    anthropic_api_key: str | None = None

    # --- Google AI Studio (Gemini API) ---
    google_ai_studio_api_key: str | None = None

    # --- Google Vertex AI ---
    vertex_project_id: str | None = None
    vertex_region: str | None = None
    google_application_credentials: str | None = None

    # --- AWS Bedrock ---
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region: str | None = None
    bedrock_anthropic_model_id: str | None = None

    # --- Databricks (model serving endpoints, e.g. hosted Gemini/Llama/etc.) ---
    databricks_host: str | None = None
    databricks_token: str | None = None

    # --- Direct Gemini key (used by token_daddy.llm.providers.gemini when no gateway) ---
    google_api_key: str | None = None

    @property
    def openai_base_url(self) -> str | None:
        """Where the OpenAI SDK should point, or None for the vendor default."""
        if not (self.databricks_host and self.databricks_token):
            return None
        return f"{self.databricks_host.rstrip('/')}/ai-gateway/openai/v1"

    @property
    def gemini_base_url(self) -> str | None:
        """Where the google-genai SDK should point, or None for the default."""
        if not (self.databricks_host and self.databricks_token):
            return None
        return f"{self.databricks_host.rstrip('/')}/ai-gateway/gemini"

    # --- Sampling, for reproducibility -------------------------------------
    # Available levers differ sharply by model:
    #
    #   gpt-5 family  REJECTS any temperature but 1 ("Unsupported value:
    #                 'temperature' does not support 0.0 with this model").
    #                 `seed` IS accepted, and is the only lever there.
    #   gemini        accepts temperature 0 and seed.
    #
    # So temperature is per-provider and OPTIONAL - None means do not send it,
    # which is what a model that refuses to be told wants.
    #
    # A seed is best-effort on every provider. Inference stays
    # non-deterministic even at temperature 0 because batching and hardware
    # scheduling introduce real variation; expect improvement, not zero.
    llm_seed: int | None = 7
    openai_temperature: float | None = None
    # Low reasoning won the production extraction/indexing trade-off and
    # avoids spending output quota on unnecessary deliberation.
    openai_reasoning_effort: str | None = "low"
    # VERIFIED against the live API: deployed Gemini 3.x models accept
    # temperature.
    gemini_temperature: float | None = 0.0
    # How hard Gemini 3.x thinks before answering. Thinking tokens are billed
    # as output - which is the quantity the workspace quota actually limits.
    gemini_thinking_level: str | None = "low"

    # --- Rate limit ceilings (enforced by token_daddy.llm.gate) -------------
    # Shared Redis counters enforce these across every worker. The hourly
    # allowance (360K) is tighter than 1,000 RPS, hence 6,000 RPM. Input and
    # output quotas differ by model and are therefore configured separately.
    llm_request_timeout_seconds: float = 180.0

    openai_tpm: int = 10_000_000
    openai_output_tpm: int = 1_000_000
    openai_rpm: int = 6_000
    openai_max_concurrent: int = 64
    gemini_tpm: int = 50_000_000
    gemini_output_tpm: int = 15_000_000
    gemini_rpm: int = 6_000
    gemini_max_concurrent: int = 200

    # Tokens one job may consume before it is failed. Stops a single runaway
    # job from starving every job queued behind it - failing one job is
    # much better than degrading all of them.
    job_token_budget: int = 2_000_000


settings = Settings()
