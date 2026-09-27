# Releasing tokenjuggler

Releases are published by GitHub Actions (`.github/workflows/release.yml`) when a
version tag is pushed. Nothing is uploaded from a laptop and no API token is stored:
PyPI trusts this repository's workflow directly ("trusted publishing").

## One-time setup

1. **PyPI and TestPyPI accounts** with 2FA: <https://pypi.org>, <https://test.pypi.org>.
2. **Add a pending trusted publisher on each** (the project doesn't exist yet, so
   use "pending"):
   - PyPI: <https://pypi.org/manage/account/publishing/> - TestPyPI: <https://test.pypi.org/manage/account/publishing/>
   - PyPI project name: `tokenjuggler`
   - Owner: `ishu0505`, repository: `token_juggler`
   - Workflow name: `release.yml`
   - Environment name: `pypi` on PyPI, `testpypi` on TestPyPI
3. **Create the two environments in GitHub**: repo Settings -> Environments ->
   `testpypi` and `pypi`. On `pypi`, add yourself under "Required reviewers" so every
   real release waits for a click.

## Each release

1. Update `CHANGELOG.md` (move items under the new version, with today's date).
2. Bump the version: `uv version 0.1.1` (or `uv version --bump patch|minor`).
3. Commit, then tag and push:
   ```bash
   git commit -am "Release 0.1.1"
   git tag v0.1.1
   git push origin main v0.1.1
   ```
4. Watch the "Release" workflow: tests -> build -> TestPyPI -> install check from
   TestPyPI -> (your approval) -> PyPI.

The workflow refuses to run if the tag doesn't match the version in `pyproject.toml`.

## Before the first release

- **Visibility:** a PyPI release publishes the source code (the sdist and wheel are
  downloadable by anyone) under Apache-2.0, even while the GitHub repo is private.
- **Try it locally:** `uv build` then install the wheel in a fresh venv:
  `uv venv /tmp/t && VIRTUAL_ENV=/tmp/t uv pip install "tokenjuggler[all] @ file://$PWD/dist/tokenjuggler-0.1.0-py3-none-any.whl"`.
