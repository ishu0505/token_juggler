"""The config UI's API: what the page relies on, against a file and against Redis."""

from pathlib import Path

import fakeredis
import httpx
import pytest
import yaml

from tokenjuggler.central import CentralConfig
from tokenjuggler.templates import TEMPLATES, read_template
from tokenjuggler.ui import create_app

ROOT = Path(__file__).parent.parent
ENV = {"OPENAI_API_KEY": "sk-x", "GOOGLE_AI_STUDIO_API_KEY": "k"}


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ui")


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_is_a_valid_config(name):
    from tokenjuggler.settings import parse_config

    parse_config(read_template(name))


def test_the_repo_example_matches_the_packaged_full_template():
    assert (ROOT / "tokenjuggler.yaml.example").read_text() == read_template("full")


async def test_page_and_meta(tmp_path):
    async with client(create_app(config_path=str(tmp_path / "c.yaml"), environ=ENV)) as c:
        page = await c.get("/")
        assert page.status_code == 200 and "tokenjuggler config" in page.text
        meta = (await c.get("/api/meta")).json()
        assert meta["source"]["mode"] == "file"
        assert "vertex" in meta["enums"]["provider"] and "rpm" in meta["enums"]["limit"]
        assert set(meta["templates"]) == set(TEMPLATES)
        assert (await c.get("/api/config")).json()["exists"] is False


async def test_analyze_reports_routes_env_and_never_env_values(tmp_path):
    raw = yaml.safe_load(read_template("minimal"))
    async with client(create_app(config_path=str(tmp_path / "c.yaml"), environ=ENV)) as c:
        out = (await c.post("/api/analyze", json={"raw": raw})).json()
    assert out["ok"] is True
    assert {d["id"] for d in out["routable"]} == {"gpt-5.6-terra@openai", "gemini-3.8-flash@aistudio"}
    assert out["env"]["openai"]["api_key_env"] == {"var": "OPENAI_API_KEY", "set": True}
    assert "sk-x" not in str(out)  # values are never sent to the page
    assert "gpt-5.6-terra" in out["yaml"]


async def test_analyze_explains_errors(tmp_path):
    raw = yaml.safe_load(read_template("minimal"))
    raw["models"]["gpt-5.6-terra"]["deployments"][0]["account"] = "nope"
    raw["projects"] = {"a": {"reserve": 0.8}, "b": {"reserve": 0.5}}
    async with client(create_app(config_path=str(tmp_path / "c.yaml"))) as c:
        out = (await c.post("/api/analyze", json={"raw": raw})).json()
    assert out["ok"] is False
    assert any("unknown account" in e["msg"] for e in out["errors"])


async def test_missing_credentials_show_as_unroutable(tmp_path):
    raw = yaml.safe_load(read_template("minimal"))
    async with client(create_app(config_path=str(tmp_path / "c.yaml"), environ={})) as c:
        out = (await c.post("/api/analyze", json={"raw": raw})).json()
    assert out["ok"] and out["routable"] == []
    assert "OPENAI_API_KEY" in out["unavailable"]["gpt-5.6-terra@openai"]


async def test_parse_yaml_and_syntax_errors(tmp_path):
    async with client(create_app(config_path=str(tmp_path / "c.yaml"))) as c:
        ok = (await c.post("/api/parse", json={"yaml": read_template("central")})).json()
        assert ok["ok"] and ok["projects_split"]
        bad = await c.post("/api/parse", json={"yaml": "models: [unclosed"})
        assert bad.status_code == 422 and "syntax" in bad.json()["detail"]


async def test_save_keeps_text_verbatim_and_backs_up(tmp_path):
    path = tmp_path / "tokenjuggler.yaml"
    path.write_text("# old\n" + read_template("minimal"))
    text = "# my comment survives\n" + read_template("minimal")
    async with client(create_app(config_path=str(path))) as c:
        loaded = (await c.get("/api/config")).json()
        assert loaded["exists"] and loaded["yaml"].startswith("# old")
        saved = (await c.post("/api/save", json={"yaml": text})).json()
    assert path.read_text() == text
    assert Path(saved["backup"]).read_text().startswith("# old")


async def test_save_refuses_an_invalid_config(tmp_path):
    path = tmp_path / "tokenjuggler.yaml"
    async with client(create_app(config_path=str(path))) as c:
        r = await c.post("/api/save", json={"yaml": "accounts: {}\nmodels: {m: {family: nope}}"})
    assert r.status_code == 422 and not path.exists()


async def test_central_mode_loads_and_publishes():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    central = CentralConfig(redis, "tj")
    text = read_template("central")
    await central.push(text)
    async with client(create_app(central=central)) as c:
        meta = (await c.get("/api/meta")).json()
        assert meta["source"] == {"mode": "central", "path": None, "namespace": "tj", "version": 1}
        assert (await c.get("/api/config")).json()["version"] == 1
        changed = text.replace("max_wait_seconds: 30", "max_wait_seconds: 10")
        r = (await c.post("/api/push", json={"yaml": changed, "expected_version": 1})).json()
        assert r == {"version": 2, "changed": True}
        stale = await c.post("/api/push", json={"yaml": text, "expected_version": 1})
        assert stale.status_code == 409  # someone (we) published v2 meanwhile
        assert (await c.post("/api/save", json={"yaml": text})).status_code == 400
