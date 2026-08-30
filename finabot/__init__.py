"""Finabot, a Chinese-market financial assistant."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("finabot")
except PackageNotFoundError:
    __version__ = "0.1.4.post5"

__logo__ = "🤖"