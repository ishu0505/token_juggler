"""tokenjuggler - one interface to GPT, Gemini and Claude across many providers,
with shared quota enforcement, failover and cost tracking.

    import tokenjuggler as juggle

    tj = juggle.from_config("tokenjuggler.yaml", project="search-svc")    # local config
    tj = await juggle.connect("redis://...", project="search-svc")        # central config
    r = await tj.generate("gpt-5.6-terra", "hello")
"""

from tokenjuggler.client import TokenJuggler
from tokenjuggler.router import (
    AllRoutesFailed,
    NoCapableDeployment,
    QuotaExceeded,
    RoutingError,
)
from tokenjuggler.types import Capability, File, Response, Text, Usage

# Shorter name for the client class.
Juggler = TokenJuggler

__all__ = [
    "AllRoutesFailed", "Capability", "File", "Juggler", "NoCapableDeployment",
    "QuotaExceeded", "Response", "RoutingError", "Text", "TokenJuggler", "Usage",
    "connect", "from_config", "main",
]


def from_config(path, **kwargs) -> TokenJuggler:
    """A client configured from a local YAML file. See TokenJuggler.from_config."""
    return TokenJuggler.from_config(path, **kwargs)


async def connect(redis_url: str | None = None, **kwargs) -> TokenJuggler:
    """A client configured from the central config in Redis. See TokenJuggler.connect."""
    return await TokenJuggler.connect(redis_url, **kwargs)


def main() -> None:
    from tokenjuggler.cli import main as cli_main

    cli_main()
