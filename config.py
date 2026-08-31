"""Loads config.yaml + .env once, shared across CLI and UI."""
import os
import functools
import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache(maxsize=1)
def load_config(path: str = None) -> dict:
    path = path or os.path.join(_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def env(key: str, default=None, required: bool = False):
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val
