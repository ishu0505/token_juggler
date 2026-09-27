"""The shared config, stored in the same Redis that holds the quotas.

An admin pushes a YAML file once; every service connects with just a Redis URL
and a project name, loads the current version, and picks up new versions on
its own. Each push is validated first and kept in a short history, so a bad
change can be rolled back with one command.

Keys, under the config's own `{namespace}` hash tag:

    {ns}:config          hash: version, yaml, sha256, updated_at, updated_by
    {ns}:config:history  list of past versions, newest first (last 20)

Only the YAML is stored - it names the env vars holding credentials, never the
credentials themselves.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from tokenjuggler.settings import Config, parse_config

HISTORY_LENGTH = 20

# Compare-and-set, so two admins pushing at once cannot silently overwrite
# each other: ARGV[1] is the version the pusher last saw ("" to skip the check).
_PUSH = """
local current = tonumber(redis.call('HGET', KEYS[1], 'version') or '0')
if ARGV[1] ~= '' and tonumber(ARGV[1]) ~= current then return {-1, current} end
local version = current + 1
redis.call('HSET', KEYS[1], 'version', version, 'yaml', ARGV[2], 'sha256', ARGV[3],
           'updated_at', ARGV[4], 'updated_by', ARGV[5])
redis.call('LPUSH', KEYS[2], cjson.encode({version = version, sha256 = ARGV[3],
           updated_at = ARGV[4], updated_by = ARGV[5], yaml = ARGV[2]}))
redis.call('LTRIM', KEYS[2], 0, tonumber(ARGV[6]) - 1)
return {version, current}
"""


class ConfigError(RuntimeError):
    pass


class CentralConfig:
    def __init__(self, redis: Any, namespace: str):
        self._redis = redis
        self.namespace = namespace
        self._key = f"{{{namespace}}}:config"
        self._history = f"{{{namespace}}}:config:history"
        self._push = redis.register_script(_PUSH)

    async def version(self) -> int:
        return int(await self._redis.hget(self._key, "version") or 0)

    async def fetch(self) -> tuple[int, str] | None:
        """(version, yaml) of the current config, or None if none was pushed."""
        data = await self._redis.hgetall(self._key)
        if not data:
            return None
        return int(data["version"]), data["yaml"]

    async def info(self) -> dict | None:
        data = await self._redis.hgetall(self._key)
        if not data:
            return None
        return {k: v for k, v in data.items() if k != "yaml"}

    async def push(
        self, text: str, *, by: str = "", expected_version: int | None = None,
        force: bool = False,
    ) -> tuple[int, Config]:
        """Validate and publish a config. Returns (new version, parsed config).

        Refuses a config for a different namespace, and - unless `force` - one
        identical to the current version (so a re-push doesn't churn every
        service's reload).
        """
        config = parse_config(text)  # raises before anything is written
        if config.namespace != self.namespace:
            raise ConfigError(
                f"the file is for namespace {config.namespace!r}, not {self.namespace!r}"
            )
        sha = hashlib.sha256(text.encode()).hexdigest()
        current = await self.info()
        if current and current.get("sha256") == sha and not force:
            return int(current["version"]), config
        version, seen = await self._push(
            keys=[self._key, self._history],
            args=[
                "" if expected_version is None else str(expected_version),
                text, sha, datetime.now(UTC).isoformat(timespec="seconds"),
                by, HISTORY_LENGTH,
            ],
        )
        if int(version) < 0:
            raise ConfigError(
                f"config changed underneath you: expected version {expected_version}, "
                f"it is now {seen}. Pull, re-apply your change, and push again."
            )
        return int(version), config

    async def history(self, limit: int = HISTORY_LENGTH) -> list[dict]:
        rows = await self._redis.lrange(self._history, 0, limit - 1)
        return [json.loads(r) for r in rows]

    async def rollback(self, to_version: int, *, by: str = "") -> int:
        """Re-publish an earlier version as a new version."""
        for entry in await self.history():
            if entry["version"] == to_version:
                version, _ = await self.push(entry["yaml"], by=by or f"rollback to v{to_version}",
                                             force=True)
                return version
        raise ConfigError(f"version {to_version} is not in the last {HISTORY_LENGTH} versions")
