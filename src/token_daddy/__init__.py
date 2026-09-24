"""token_daddy - one interface to GPT, Gemini and Claude across many providers,
with shared quota enforcement, failover and cost tracking."""

from token_daddy.client import TokenDaddy
from token_daddy.router import (
    AllRoutesFailed,
    NoCapableDeployment,
    QuotaExceeded,
    RoutingError,
)
from token_daddy.types import Capability, File, Response, Text, Usage

__all__ = [
    "AllRoutesFailed", "Capability", "File", "NoCapableDeployment", "QuotaExceeded",
    "Response", "RoutingError", "Text", "TokenDaddy", "Usage", "main",
]


def main() -> None:
    from token_daddy.cli import main as cli_main

    cli_main()
