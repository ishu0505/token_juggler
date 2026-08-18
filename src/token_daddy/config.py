from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Redis (token/RPM/TPM tracking, shared across apps using the same pool) ---
    redis_url: str = "redis://localhost:6379/0"

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


settings = Settings()
