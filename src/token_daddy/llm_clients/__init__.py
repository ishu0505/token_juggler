"""LLM clients - import everything you need from here.

    from token_daddy.llm_clients import Attachment, get_client

    client = get_client("gemini")     # or "openai" / "anthropic"
    reply = client.generate("What is this?", attachments=[Attachment.from_path("a.pdf")])
    print(reply.text, reply.usage.input_tokens, reply.usage.output_tokens)

Providers are imported lazily so you only need the SDK for the one you use.
"""

from token_daddy.llm_clients.base_client import (
    Attachment,
    BaseLLMClient,
    LLMResponse,
    Provider,
    ThinkingLevel,
    TokenUsage,
    UnsupportedAttachmentError,
    strict_json_schema,
)

__all__ = [
    "Attachment",
    "BaseLLMClient",
    "LLMResponse",
    "Provider",
    "ThinkingLevel",
    "TokenUsage",
    "UnsupportedAttachmentError",
    "get_client",
    "strict_json_schema",
]

PROVIDERS = tuple(p.value for p in Provider)


def get_client(
    provider: Provider | str, model: str | None = None, **kwargs
) -> BaseLLMClient:
    """Build a client by provider name. Same interface whichever you pick."""
    provider = str(provider).lower()

    if provider == Provider.GEMINI:
        from token_daddy.llm_clients.gemini_client import GeminiClient

        return GeminiClient(model=model, **kwargs)

    if provider == Provider.OPENAI:
        from token_daddy.llm_clients.openai_client import OpenAIClient

        return OpenAIClient(model=model, **kwargs)

    if provider == Provider.ANTHROPIC:
        from token_daddy.llm_clients.anthropic_client import AnthropicClient

        return AnthropicClient(model=model, **kwargs)

    raise ValueError(f"Unknown provider '{provider}'. Pick one of: {', '.join(PROVIDERS)}")
