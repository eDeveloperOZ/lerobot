"""Lightweight stubs used for tests without the real `draccus` dependency."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json


class ChoiceRegistry:
    """Simple registry for configuration subclasses.

    This stub mimics the small subset of the real library used in tests.  It is
    implemented as a normal base class so it can be combined with ``abc.ABC``
    without metaclass conflicts.
    """

    registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "__name__", "")
        ChoiceRegistry.registry[name] = cls

    @classmethod
    def register_subclass(cls, name: str):
        def deco(subcls: type) -> type:
            cls.registry[name] = subcls
            return subcls
        return deco


CHOICE_TYPE_KEY = "type"


def wrap():
    def deco(fn):
        return fn

    return deco


def config_type(_):
    @contextmanager
    def cm():
        yield

    return cm()


def dump(obj, f, **kwargs):
    json.dump(asdict(obj), f, **kwargs)


def parse(cls, config_file, args=None):
    return cls()


def set_config_type(_):
    pass
