from abc import ABC, abstractmethod
import argparse
from typing import Optional

class BaseCommand(ABC):
    """An abstract class to serve LeRobot CLI commands"""

    @property
    @abstractmethod
    def name(self)-> str:
        """Commands name in the CLI"""
        pass

    @property
    @abstractmethod
    def help(self)-> str:
        """Command help shown in CLI help"""
        pass

    @property
    def description(self)-> str:
        """"Detailed command description for command help"""
        return None
    
    @abstractmethod
    def register_parser(self, subparser: argparse._SubParsersAction)-> None:
        """Register command-specific arguments"""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command logic"""
        pass