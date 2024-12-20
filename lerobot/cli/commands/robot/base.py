# lerobot/cli/commands/robot/base.py
import argparse
import importlib
import gymnasium as gym

from ...base import BaseCommand
from ...uitls import *
from lerobot.common.utils.utils import init_hydra_config

class RobotCommand(BaseCommand):
    """Base class for robot-related commands"""
    def __init__(self, name: str, help_text: str, description: str | None = None):
        super().__init__(name, help_text, description)

    def _register_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add common robot arguments"""
        parser.add_argument(
            "--robot-path",
            type=str,
            default="lerobot/configs/robot/koch.yaml",
            help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
        )
        parser.add_argument(
            "--robot-overrides",
            type=str,
            nargs="*",
            help="Any key=value arguments to override config values (use dots for.nested=overrides)",
        )
        parser.add_argument(
            "-sim",
            type=str,
            default=None,
            help="Path to yaml config for simulation environment. If provided, runs in simulation mode."
        )

    def _init_simulation(self, sim):
        """Initialize simulation environment if sim mode is enabled"""
        if sim:
            env_cfg = init_hydra_config(sim)
            importlib.import_module(f"gym_{env_cfg.env.name}")

            def env_constructor():
                return gym.make(env_cfg.env.handle, disable_env_checker=True, **env_cfg.env.gym)
                
            return env_constructor, env_cfg
        return None, None

    def _get_sim_calibration(self, robot, env_cfg):
        """Get simulation calibration if in sim mode"""
        if env_cfg:
            return init_sim_calibration(robot, env_cfg.calibration)
        return None