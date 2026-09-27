from pathlib import Path

import pytest
from pydantic import ValidationError

from tokenjuggler.registry import Registry
from tokenjuggler.settings import Config, WireApi, load_config
from tokenjuggler.types import Capability

EXAMPLE = Path(__file__).parent.parent / "tokenjuggler.yaml.example"

ALL_CREDS = {
    "OPENAI_API_KEY": "sk-test",
    "DATABRICKS_HOST": "dbc-123.cloud.databricks.com",
    "DATABRICKS_TOKEN": "dapi-test",
    "BEDROCK_API_KEY": "bedrock-test",
    "GOOGLE_AI_STUDIO_API_KEY": "aistudio-test",
    "VERTEX_PROJECT_ID": "my-proj",
    "ANTHROPIC_API_KEY": "sk-ant-test",
}


def test_example_config_has_the_whole_mark1_catalog():
    registry = Registry(load_config(EXAMPLE), environ=ALL_CREDS)
    assert len(registry.models) == 14  # 9 gpt + 2 gemini + 3 claude
    assert len(registry.deployments) == 39
    assert registry.unavailable == {}


def test_each_provider_gets_its_recommended_api_and_endpoint():
    registry = Registry(load_config(EXAMPLE), environ=ALL_CREDS)
    d = registry.deployments
    assert d["gpt-5.6-sol@bedrock"].api is WireApi.OPENAI_RESPONSES
    assert d["gpt-5.6-sol@bedrock"].base_url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
    )
    assert d["gpt-5.6-sol@databricks"].base_url == (
        "https://dbc-123.cloud.databricks.com/ai-gateway/openai/v1"
    )
    assert d["gemini-3.8-flash@aistudio"].api is WireApi.GENAI_INTERACTIONS
    assert d["gemini-3.8-flash@vertex"].api is WireApi.GENAI_GENERATE_CONTENT
    assert d["gemini-3.8-flash@databricks"].base_url.endswith("/ai-gateway/gemini")
    assert d["claude-sonnet-5@databricks"].api is WireApi.ANTHROPIC_MESSAGES
    assert d["claude-sonnet-5@databricks"].base_url.endswith("/ai-gateway/anthropic")
    assert d["claude-sonnet-5@databricks"].credentials.bearer_token == "dapi-test"


def test_defaults_merge_under_deployment_overrides():
    registry = Registry(load_config(EXAMPLE), environ=ALL_CREDS)
    limits = registry.deployments["gpt-5.4@openai"].limits
    assert limits.buckets() == {
        "rps": 1000, "rph": 360000, "input_tpm": 200000, "output_tpm": 20000,
    }


def test_missing_credentials_make_a_deployment_unroutable_not_an_error():
    env = {k: v for k, v in ALL_CREDS.items() if k != "BEDROCK_API_KEY"}
    registry = Registry(load_config(EXAMPLE), environ=env)
    assert "gpt-5.4@bedrock" not in registry.deployments
    assert "BEDROCK_API_KEY" in registry.unavailable["gpt-5.4@bedrock"]
    assert [d.account for d in registry.model("gpt-5.4").deployments] == ["databricks", "openai"]


def test_gemini_routes_carry_audio_gpt_routes_do_not():
    registry = Registry(load_config(EXAMPLE), environ=ALL_CREDS)
    assert registry.deployments["gemini-3.7-flash@vertex"].supports({Capability.AUDIO})
    assert not registry.deployments["gpt-5.4@openai"].supports({Capability.AUDIO})


def _minimal(**extra) -> dict:
    return {
        "accounts": {"a": {"provider": "openai", "api_key_env": "K"}},
        "models": {"m": {"family": "gpt", "deployments": [{"account": "a", "model_id": "x"}]}},
        **extra,
    }


def test_reserves_over_one_are_rejected_but_caps_may_overcommit():
    with pytest.raises(ValidationError, match="reserves .* over 1.0"):
        Config.model_validate(_minimal(projects={"p1": {"reserve": 0.7}, "p2": {"reserve": 0.5}}))
    # Caps are ceilings, so 0.7 + 0.5 is fine; `share` is the old name for cap.
    cfg = Config.model_validate(_minimal(projects={"p1": {"cap": 0.7}, "p2": {"share": 0.5}}))
    assert cfg.projects["p2"].cap == 0.5


def test_per_deployment_overrides_merge_over_project_defaults():
    cfg = Config.model_validate(_minimal(projects={
        "p": {"reserve": 0.2, "lend_idle": True, "deployments": {"m@a": {"reserve": 0.5}}},
        "q": {"cap": 0.5, "deployments": {"m@a": 0.3}},
    }))
    assert cfg.projects["p"].quota_for("m@a").reserve == 0.5
    assert cfg.projects["p"].quota_for("m@a").lend_idle is True
    assert cfg.projects["q"].quota_for("m@a").cap == 0.3


def test_reserve_larger_than_cap_is_rejected():
    with pytest.raises(ValidationError, match="larger than cap"):
        Config.model_validate(_minimal(projects={"p": {"cap": 0.2, "reserve": 0.5}}))


def test_unknown_deployment_in_a_project_is_rejected():
    with pytest.raises(ValidationError, match="unknown deployment"):
        Config.model_validate(_minimal(projects={"p": {"deployments": {"nope@a": 0.5}}}))


def test_unknown_account_is_rejected():
    raw = _minimal()
    raw["models"]["m"]["deployments"][0]["account"] = "nope"
    with pytest.raises(ValidationError, match="unknown account"):
        Config.model_validate(raw)


def test_typos_in_keys_are_rejected():
    raw = _minimal()
    raw["defaults"] = {"limts": {"rps": 1}}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
