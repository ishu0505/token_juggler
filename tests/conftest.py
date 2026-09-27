import fakeredis
import pytest

from tokenjuggler.limiter import InProcessBackend, RedisBackend
from tokenjuggler.registry import Registry
from tokenjuggler.settings import Config

ENV = {"KEY_A": "a", "KEY_B": "b", "KEY_C": "c"}


def make_registry(*, limits=None, deployments=None, accounts=None, projects=None,
                  headroom=1.0, models=None) -> Registry:
    """A small registry: model `m` on accounts a and b (both OpenAI-style)."""
    accounts = accounts or {
        "a": {"provider": "openai", "api_key_env": "KEY_A"},
        "b": {"provider": "openai", "api_key_env": "KEY_B"},
    }
    models = models or {
        "m": {
            "family": "gpt",
            "deployments": deployments or [
                {"account": "a", "model_id": "m-a"},
                {"account": "b", "model_id": "m-b"},
            ],
        }
    }
    raw = {
        "defaults": {"limits": limits or {}, "headroom": headroom},
        "accounts": accounts,
        "models": models,
        "projects": projects or {},
    }
    return Registry(Config.model_validate(raw), environ=ENV)


@pytest.fixture(params=["inprocess", "redis"])
def backend(request):
    if request.param == "inprocess":
        return InProcessBackend()
    return RedisBackend(fakeredis.FakeAsyncRedis(decode_responses=True))
