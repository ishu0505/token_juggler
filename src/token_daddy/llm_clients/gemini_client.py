"""Gemini client - routed through the Databricks AI Gateway, not Google AI
Studio directly. Uses DATABRICKS_TOKEN + DATABRICKS_HOST.

Implements the shared BaseLLMClient interface, so this is interchangeable with
OpenAIClient at the call site. Still the google-genai SDK, just pointed at
Databricks: the gateway wants an ``api_key`` value in the request (its own
literal, not a real Google key) with the actual bearer token carried in the
Authorization header instead, via a custom ``http_options.base_url``.

Databricks names most models ``system.ai.<our-name-with-dashes-not-dots>``
(e.g. our ``gemini-3.7-flash`` is its ``system.ai.gemini-3-7-flash``), but
this is not a reliable rule to derive on the fly - ``gemini-3.1-pro-preview``
is registered as ``system.ai.gemini-3-1-pro``, dropping "-preview" - so every
entry in ``_DATABRICKS_MODEL_IDS`` was confirmed against the live gateway
rather than assumed. Callers still pass our clean names everywhere; the
gateway id is only used for the wire call. If Databricks adds/renames a
model, confirm the new id against the gateway before adding it here - see
tests/test_llm_clients.py.
"""

from google import genai
from google.genai import types
from pydantic import BaseModel

from token_daddy.config import settings
from token_daddy.llm_clients.base_client import (
    Attachment,
    BaseLLMClient,
    LLMResponse,
    ThinkingLevel,
    TokenUsage,
    UnsupportedAttachmentError,
)

# Only the three models we've standardised on.
GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
GEMINI_3_7_FLASH = "gemini-3.7-flash"
GEMINI_3_6_FLASH = "gemini-3.6-flash"

# Our name -> the Databricks AI Gateway's "system.ai.*" model id. Confirmed
# against the live gateway, not derived - see the module docstring.
_DATABRICKS_MODEL_IDS: dict[str, str] = {
    GEMINI_3_1_PRO_PREVIEW: "system.ai.gemini-3-1-pro",
    GEMINI_3_7_FLASH: "system.ai.gemini-3-7-flash",
    GEMINI_3_6_FLASH: "system.ai.gemini-3-6-flash",
}

# Gemini takes text, images, audio, video and PDF - but NOT office formats
# (xlsx/docx/pptx). Convert those to PDF or CSV first, or send them to GPT,
# which does accept them natively.
SUPPORTED_MIME_PREFIXES = ("image/", "audio/", "video/", "text/")
SUPPORTED_MIME_TYPES = {"application/pdf", "application/json"}

# Anything bigger than this must go through the Files API rather than inline.
MAX_INLINE_BYTES = 20 * 1024 * 1024

# Our four levels happen to match Gemini's own enum one-for-one.
_THINKING_LEVELS: dict[str, types.ThinkingLevel] = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


class GeminiClient(BaseLLMClient):
    provider = "gemini"
    default_model = GEMINI_3_7_FLASH
    supported_models = (
        GEMINI_3_1_PRO_PREVIEW,
        GEMINI_3_7_FLASH,
        GEMINI_3_6_FLASH,
    )

    def __init__(
        self,
        model: str | None = None,
        databricks_token: str | None = None,
        databricks_host: str | None = None,
    ):
        super().__init__(model)
        token = databricks_token or settings.databricks_token
        host = databricks_host or settings.databricks_host
        if not token:
            raise RuntimeError("DATABRICKS_TOKEN is not set")
        if not host:
            raise RuntimeError("DATABRICKS_HOST is not set")

        self._client = genai.Client(
            api_key="databricks",  # gateway placeholder - the real auth is the header below
            http_options=types.HttpOptions(
                base_url=f"{host.rstrip('/')}/ai-gateway/gemini",
                headers={"Authorization": f"Bearer {token}"},
            ),
        )

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        attachments: list[Attachment] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking_level: ThinkingLevel | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model = self._validate_model(model or self.model)

        # Files first, then the prompt - Gemini responds better with the
        # question asked after the material it refers to.
        parts: list = [self._to_part(item) for item in (attachments or [])]
        parts.append(prompt)

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        if thinking_level is not None:
            config.thinking_config = types.ThinkingConfig(
                thinking_level=_THINKING_LEVELS[thinking_level],
            )

        if response_schema is not None:
            # The SDK reads the pydantic model directly - no hand-rolled
            # JSON Schema needed, unlike the OpenAI path.
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        response = self._client.models.generate_content(
            model=_DATABRICKS_MODEL_IDS[model],
            contents=parts,
            config=config,
        )

        usage = response.usage_metadata

        # Gemini 3.x models think, and `candidates_token_count` counts only the
        # visible answer - thinking tokens are reported separately but are still
        # billed as output, so add them in.
        reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + reasoning_tokens

        return self._finish(
            text=response.text or "",
            model=model,
            usage=TokenUsage.build(
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=output_tokens,
                total_tokens=getattr(usage, "total_token_count", None),
                reasoning_tokens=reasoning_tokens,
            ),
            raw=response,
        )

    def _to_part(self, attachment: Attachment) -> types.Part:
        """Turn an Attachment into an inline Gemini Part, or explain why we can't."""
        if not self._is_supported(attachment.mime_type):
            raise UnsupportedAttachmentError(
                f"Gemini can't read '{attachment.filename}' ({attachment.mime_type}). "
                "Convert it to PDF/CSV first, or send it to the OpenAI client instead."
            )

        if attachment.size_bytes > MAX_INLINE_BYTES:
            raise UnsupportedAttachmentError(
                f"'{attachment.filename}' is {attachment.size_bytes} bytes, over the "
                f"{MAX_INLINE_BYTES}-byte inline limit. Upload it via the Gemini Files "
                "API and pass the file reference instead."
            )

        return types.Part.from_bytes(
            data=attachment.data,
            mime_type=attachment.mime_type,
        )

    @staticmethod
    def _is_supported(mime_type: str) -> bool:
        return mime_type in SUPPORTED_MIME_TYPES or mime_type.startswith(
            SUPPORTED_MIME_PREFIXES
        )
