# lerobot/cli/commands/model/base.py
from abc import abstractmethod
import argparse

from ...base import BaseCommand

class ModelCommand(BaseCommand):
    def __init__(self, name: str, help_text: str):
        self._name = name
        self._help = help_text
    
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def help(self) -> str:
        return self._help