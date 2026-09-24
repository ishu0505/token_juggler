"""GPT client - routed through the Databricks AI Gateway, not OpenAI directly.
Uses DATABRICKS_TOKEN + DATABRICKS_HOST.

Implements the shared BaseLLMClient interface, so this is interchangeable with
GeminiClient at the call site. Still the OpenAI SDK's Responses API, just
pointed at Databricks via `base_url` - the gateway accepts the same bearer
token as the SDK's own `api_key`, so no custom headers needed here (unlike
the Gemini client, whose SDK insists on a real-looking `api_key` value).

Databricks names each model ``system.ai.<our-name-with-dashes-not-dots>``
(e.g. our ``gpt-5.4`` is its ``system.ai.gpt-5-4``) - see
``_DATABRICKS_MODEL_IDS``. Callers still pass our clean names everywhere;
the gateway id is only used for the wire call.
"""

import base64

from openai import OpenAI
from pydantic import BaseModel

from token_daddy.config import settings
from token_daddy.llm_clients.base_client import (
    Attachment,
    BaseLLMClient,
    LLMResponse,
    ThinkingLevel,
    TokenUsage,
    strict_json_schema,
)

# Only the five models we've standardised on.
GPT_5_4 = "gpt-5.4"
GPT_5_4_MINI = "gpt-5.4-mini"
GPT_5_4_NANO = "gpt-5.4-nano"
GPT_5_6_TERRA = "gpt-5.6-terra"
GPT_5_6_LUNA = "gpt-5.6-luna"

# Our name -> the Databricks AI Gateway's "system.ai.*" model id.
_DATABRICKS_MODEL_IDS: dict[str, str] = {
    GPT_5_4: "system.ai.gpt-5-4",
    GPT_5_4_MINI: "system.ai.gpt-5-4-mini",
    GPT_5_4_NANO: "system.ai.gpt-5-4-nano",
    GPT_5_6_TERRA: "system.ai.gpt-5-6-terra",
    GPT_5_6_LUNA: "system.ai.gpt-5-6-luna",
}


class OpenAIClient(BaseLLMClient):
    provider = "openai"
    default_model = GPT_5_4
    supported_models = (
        GPT_5_4,
        GPT_5_4_MINI,
        GPT_5_4_NANO,
        GPT_5_6_TERRA,
        GPT_5_6_LUNA,
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

        self._client = OpenAI(
            api_key=token, base_url=f"{host.rstrip('/')}/ai-gateway/openai/v1"
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

        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for item in attachments or []:
            content.append(self._to_content_block(item))

        # Only send knobs that were actually asked for - the reasoning models
        # reject an explicit temperature, so passing a default would break
        # every call that didn't need one.
        kwargs: dict = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if thinking_level is not None:
            kwargs["reasoning"] = {"effort": thinking_level}
        if response_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_schema.__name__,
                    "schema": strict_json_schema(response_schema),
                    "strict": True,
                }
            }

        response = self._client.responses.create(
            model=_DATABRICKS_MODEL_IDS[model],
            instructions=system,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)

        return self._finish(
            text=response.output_text or "",
            model=model,
            usage=TokenUsage.build(
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", None),
                # Already counted inside output_tokens; surfaced for visibility.
                reasoning_tokens=getattr(details, "reasoning_tokens", 0),
            ),
            raw=response,
        )

    @staticmethod
    def _to_content_block(attachment: Attachment) -> dict:
        """Images go in as input_image; everything else as input_file.

        input_file covers pdf, xlsx, csv, docx, pptx, txt, md, json and more,
        so there's no allow-list here - we let the API reject anything odd.
        """
        encoded = base64.b64encode(attachment.data).decode("utf-8")
        data_url = f"data:{attachment.mime_type};base64,{encoded}"

        if attachment.mime_type.startswith("image/"):
            return {"type": "input_image", "image_url": data_url}

        return {
            "type": "input_file",
            "filename": attachment.filename,
            "file_data": data_url,
        }
