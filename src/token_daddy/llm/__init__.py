"""Structured model access and the shared rate-limit gate.

Pipeline stages obtain clients through :func:`factory.client_for`. Provider
adapters translate the common call shape into each provider SDK, while every
actual call still passes through :mod:`token_daddy.llm.gate`.
"""

from token_daddy.llm.factory import client_for

__all__ = ["client_for"]
