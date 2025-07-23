import importlib
import os

import pytest


@pytest.mark.cloud
def test_cloud_mode(monkeypatch):
    monkeypatch.setenv("LEROBOT_MODE", "cloud")
    mod = importlib.import_module("lerobot_remote")
    assert mod.mode == "cloud"


@pytest.mark.edge
def test_edge_mode(monkeypatch):
    monkeypatch.setenv("LEROBOT_MODE", "edge")
    mod = importlib.import_module("lerobot_remote")
    assert mod.mode == "edge"
