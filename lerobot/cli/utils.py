import argparse
from typing import Dict, Type, Optional
from pathlib import Path

from .base import BaseCommand


class LeRobotCLI:
    
    def __init__(self) -> None:
        self.commands: Dict[str, BaseCommand] = {}
        self.parser: Optional[argparse.ArgumentParser] = None
        self.init_parser()
        self.register_commands()

    def init_parser(self):
        self.parser = argparse.ArgumentParser(
            description="LeRobot CLI - State-of-the-art Machine Learning for Real-World Robotics"
        )
        self.subparsers = self.parser.add_subparsers(
            title="commands",
            dest="command",
            required=True
        )

    def register_commands(self) -> None:
        """Register all available commands"""
        # Robot commands
        self.register_command(CalibrateCommand())
        self.register_command(TelepoperateCommand())
        self.register_command(RecordCommand())
        # We'll add more commands as we implement them

    def register_command(self, command: BaseCommand) -> None:
        """Register a single command"""
        command.register_parser(self.subparsers)
        self.commands[command.name] = command

    def run(self, args=None) -> int:
        """Execute CLI with given args"""
        parsed_args = self.parser.parse_args(args)
        command = self.commands[parsed_args.command]
        try:
            return command.execute(parsed_args)
        except Exception as e:
            self.parser.error(str(e))
            return 1
        
def main() -> int:
    cli = LeRobotCLI()
    return cli.run()

if __name__ == "__main__":
    exit(main())