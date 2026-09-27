"""The shared config in Redis: publishing, versions, and services following it."""

import asyncio

import fakeredis
import pytest
import yaml
from pydantic import ValidationError

from tests.test_router import FakeAdapter
from tokenjuggler import TokenJuggler
from tokenjuggler.central import CentralConfig, ConfigError
from tokenjuggler.limiter import RedisBackend
from tokenjuggler.settings import WireApi

ENV = {"KEY_A": "a", "KEY_B": "b"}


def config_yaml(rpm: int, namespace: str = "tj", projects: str = "", headroom: float = 0.95) -> str:
    return f"""
namespace: {namespace}
defaults:
  headroom: {headroom}
  limits: {{rpm: {rpm}}}
accounts:
  a: {{provider: openai, api_key_env: KEY_A}}
  b: {{provider: openai, api_key_env: KEY_B}}
models:
  m:
    family: gpt
    deployments:
      - {{account: a, model_id: m-a}}
      - {{account: b, model_id: m-b}}
{projects}"""


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


async def connect(redis, **kw):
    fake = FakeAdapter()
    tj = await TokenJuggler.connect(
        "redis://unused", backend=RedisBackend(redis), environ=ENV,
        adapters={api: fake for api in WireApi}, **kw,
    )
    return tj, fake


async def test_push_then_connect_loads_that_version(redis):
    central = CentralConfig(redis, "tj")
    version, _ = await central.push(config_yaml(rpm=5), by="alice")
    assert version == 1
    tj, _ = await connect(redis, project="web", refresh_seconds=0)
    assert tj.config_version == 1
    assert tj.registry.deployments["m@a"].limits.rpm == 5
    info = await central.info()
    assert info["updated_by"] == "alice" and "yaml" not in info


async def test_connect_without_a_pushed_config_explains_what_to_do(redis):
    with pytest.raises(ConfigError, match="config push"):
        await connect(redis)


async def test_an_invalid_config_is_never_published(redis):
    central = CentralConfig(redis, "tj")
    with pytest.raises(ValidationError):
        await central.push("namespace: tj\naccounts: {}\nmodels: {m: {family: nope}}")
    assert await central.version() == 0


async def test_a_config_for_another_namespace_is_refused(redis):
    with pytest.raises(ConfigError, match="namespace"):
        await CentralConfig(redis, "tj").push(config_yaml(5, namespace="other"))


async def test_pushing_the_same_file_twice_does_not_bump_the_version(redis):
    central = CentralConfig(redis, "tj")
    assert (await central.push(config_yaml(5)))[0] == 1
    assert (await central.push(config_yaml(5)))[0] == 1


async def test_concurrent_edits_are_caught_by_expected_version(redis):
    central = CentralConfig(redis, "tj")
    await central.push(config_yaml(5))
    await central.push(config_yaml(6), expected_version=1)          # alice, from v1
    with pytest.raises(ConfigError, match="changed underneath"):
        await central.push(config_yaml(7), expected_version=1)      # bob, also from v1


async def test_rollback_republishes_an_old_version(redis):
    central = CentralConfig(redis, "tj")
    await central.push(config_yaml(5))
    await central.push(config_yaml(9))
    assert await central.rollback(1) == 3
    _, text = await central.fetch()
    assert "rpm: 5" in text
    assert [h["version"] for h in await central.history()] == [3, 2, 1]


async def test_services_pick_up_a_new_version_without_restarting(redis):
    central = CentralConfig(redis, "tj")
    await central.push(config_yaml(rpm=1))
    tj, _ = await connect(redis, project="web", refresh_seconds=0.05)

    await tj.generate("m", "hi")                         # m@a's one request/min
    assert (await tj.generate("m", "hi")).deployment == "m@b"

    await central.push(config_yaml(rpm=100), by="admin")  # raise the limit centrally
    await asyncio.sleep(0.2)
    assert tj.config_version == 2
    assert (await tj.generate("m", "hi")).deployment == "m@a"  # room again under the new limit
    await tj.aclose()


async def test_a_bad_version_in_redis_is_ignored_and_the_old_one_kept(redis):
    central = CentralConfig(redis, "tj")
    await central.push(config_yaml(rpm=5))
    tj, _ = await connect(redis, refresh_seconds=0)
    # Someone writes garbage straight into Redis, bypassing push() validation.
    await redis.hset("{tj}:config", mapping={"version": 2, "yaml": "not: [valid"})
    with pytest.raises(yaml.YAMLError):
        await tj.reload_config()
    assert tj.config_version == 1 and tj.registry.deployments["m@a"].limits.rpm == 5


async def test_project_quotas_from_the_central_config_apply(redis):
    await CentralConfig(redis, "tj").push(
        config_yaml(rpm=4, headroom=1.0, projects="projects:\n  batch: {reserve: 0.5}\n")
    )
    web, _ = await connect(redis, project="web", refresh_seconds=0)
    batch, _ = await connect(redis, project="batch", refresh_seconds=0)
    only_a = [web.registry.deployments["m@a"]]
    from tokenjuggler.limiter import Cost

    grants = 0
    while (await web.limiter.acquire(only_a, [Cost(1, 1)]))[0]:
        grants += 1
    assert grants == 2  # the other half is batch's
    assert (await batch.limiter.acquire([batch.registry.deployments["m@a"]], [Cost(1, 1)]))[0]


async def test_the_juggle_shorthand(redis, tmp_path):
    import tokenjuggler as juggle

    path = tmp_path / "tokenjuggler.yaml"
    path.write_text(config_yaml(rpm=5))
    local = juggle.from_config(path, environ=ENV)
    assert isinstance(local, juggle.Juggler)
    await CentralConfig(redis, "tj").push(config_yaml(rpm=5))
    central = await juggle.connect("redis://unused", backend=RedisBackend(redis),
                                   environ=ENV, refresh_seconds=0)
    assert central.config_version == 1
