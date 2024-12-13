import argparse
from typing import Dict, Optional, List
from importlib import import_module
from pathlib import Path
from .base import BaseCommand


class LeRobotCLI:
    """Command Line Interface for LeRobot.
    
    Handles dynamic loading and execution of CLI commands from the commands directory.
    Each command should follow the {name}Command naming convention and inherit from BaseCommand.
    """

    def __init__(self) -> None:
        """Initialize the CLI with commands and argument parser."""

        self.commands: Dict[str, BaseCommand] = self._load_command_objects()
        self.parser: Optional[argparse.ArgumentParser] = None
        self.init_parser()                  

    def _load_command_objects(self) -> Dict[str, BaseCommand]:
        """Load all commands from the /commands directory.
        
        Commands must follow the {name}Command naming convention and inherit from BaseCommand.
        
        Returns:
            Dict[str, BaseCommand]: Dictionary of command name to command instance mappings.
        """
        commands = {}
        try:
            base_path = Path(__file__).parent / "commands"
            for command_path in base_path.rglob("*.py"):
                if "base.py" in str(command_path):
                    continue
                    
                try:
                    relative_path = command_path.relative_to(base_path)
                    module_name = f"lerobot.cli.commands.{relative_path.parent}.{command_path.stem}"
                    module = import_module(module_name)
                    
                    command_class_name = f"{command_path.stem}Command"
                    command_class = getattr(module, command_class_name, None)
                    
                    if (command_class and 
                        isinstance(command_class, type) and 
                        issubclass(command_class, BaseCommand) and 
                        command_class != BaseCommand):
                        
                        command_instance = command_class()
                        commands[command_class.name] = command_instance
                        
                except Exception as e:
                    print(f"Warning: Failed to load command from {command_path}: {e}")
                    continue

            if not commands:
                raise ValueError("No valid commands were found")

            return commands
            
        except Exception as e:
            raise RuntimeError(f"Error loading commands: {e}")

    def init_parser(self) -> None:
        """Initialize the argument parser and register command subparsers."""
        if not self.commands:
            raise ValueError("No commands were loaded")
            
        self.parser = argparse.ArgumentParser(
            description="LeRobot CLI - State-of-the-art Machine Learning for Real-World Robotics",
        )
        
        self.subparsers = self.parser.add_subparsers(
            help="Available commands",
            dest="command",
            required=True
        )
        
        for name, cmd in self.commands.items():
            try:
                cmd.register_parser(self.subparsers)
            except Exception as e:
                raise RuntimeError(f"Failed to register parser for command {name}: {e}")

    def run(self, args: Optional[List[str]] = None) -> int:
        """Execute CLI command with given args."""
        try:
            if self.parser is None:
                raise RuntimeError("Parser not initialized")
                
            parsed_args = self.parser.parse_args(args)
            command = self.commands.get(parsed_args.command)
            
            if not command:
                raise(f"Unknown command: {parsed_args.command}")
                
            return command.execute(parsed_args)
            
        except argparse.ArgumentError as e:
            self.parser.error(str(e))
        except Exception as e:
            self.parser.error(f"Command failed: {str(e)}")

def main() -> int:
    """CLI entry point."""
    try:
        cli = LeRobotCLI()
        return cli.run()
    except Exception as e:
        print(f"Error in executing CLI:\n {e}")
        return 1

if __name__ == "__main__":
    exit(main())