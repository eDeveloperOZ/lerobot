class ChoiceRegistry(type):
    registry = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, '__name__', '')
        ChoiceRegistry.registry[name] = cls
    @classmethod
    def register_subclass(cls, name):
        def deco(subcls):
            return subcls
        return deco

CHOICE_TYPE_KEY = 'type'

def wrap():
    def deco(fn):
        return fn
    return deco

from contextlib import contextmanager

def config_type(_):
    @contextmanager
    def cm():
        yield
    return cm()

def dump(obj, f, **kwargs):
    import json, dataclasses
    json.dump(dataclasses.asdict(obj), f, **kwargs)

def parse(cls, config_file, args=None):
    return cls()

def set_config_type(_):
    pass
