"""Starter configs shipped with the package (`tokenjuggler init`, and the UI)."""

from importlib import resources

TEMPLATES = {
    "minimal": "One model on two providers with a fallback - the smallest useful config.",
    "full": "Every Mark 1 model on every supported provider.",
    "central": "A shared config for several projects, with caps and reservations.",
}


def read_template(name: str) -> str:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; choose from {', '.join(TEMPLATES)}")
    return resources.files(__name__).joinpath(f"{name}.yaml").read_text()
