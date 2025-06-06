import importlib


def test_server_app():
    mod = importlib.import_module("server.main")
    assert hasattr(mod, "app")
