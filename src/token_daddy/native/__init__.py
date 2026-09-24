"""Native SDK clients whose traffic runs through token_daddy.

    oai = td.openai("gpt-5.6-sol")
    r = await oai.responses.create(model="gpt-5.6-sol", input="hi")

Each is the real SDK client. Its requests are limited, tracked and failed
over between every deployment that speaks the same wire API. `model` is the
default logical model for requests that don't name one we know; requests may
name any configured model. Pass `upstream=` (a transport) to replace the real
network, as the tests do.
"""

from __future__ import annotations

from token_daddy.native.transport import RoutingCore, httpx2_transport, httpx_transport

__all__ = ["anthropic_client", "genai_client", "openai_client"]

# The SDKs need a base URL; the transport replaces it on every request.
_PLACEHOLDER = "https://token-daddy.invalid"


def openai_client(td, model: str | None = None, *, upstream=None):
    import httpx2
    from openai import AsyncOpenAI

    transport = httpx2_transport(RoutingCore(td, model))
    if upstream is not None:
        transport._upstream = upstream
    return AsyncOpenAI(
        api_key="token-daddy",
        base_url=f"{_PLACEHOLDER}/v1",
        max_retries=0,  # failover happens below the SDK
        timeout=td.config.defaults.request_timeout_seconds,
        http_client=httpx2.AsyncClient(transport=transport),
    )


def anthropic_client(td, model: str | None = None, *, upstream=None):
    import httpx2
    from anthropic import AsyncAnthropic

    transport = httpx2_transport(RoutingCore(td, model))
    if upstream is not None:
        transport._upstream = upstream
    return AsyncAnthropic(
        api_key="token-daddy",
        base_url=_PLACEHOLDER,
        max_retries=0,
        timeout=td.config.defaults.request_timeout_seconds,
        http_client=httpx2.AsyncClient(transport=transport),
    )


def genai_client(td, model: str | None = None, *, upstream=None):
    import httpx
    from google import genai
    from google.genai import types

    transport = httpx_transport(RoutingCore(td, model))
    if upstream is not None:
        transport._upstream = upstream
    return genai.Client(
        api_key="token-daddy",
        http_options=types.HttpOptions(
            base_url=f"{_PLACEHOLDER}/",
            timeout=int(td.config.defaults.request_timeout_seconds * 1000),
            httpx_async_client=httpx.AsyncClient(transport=transport),
        ),
    )
