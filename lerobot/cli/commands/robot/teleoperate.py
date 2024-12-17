import argparse

from .base import RobotCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.robot_devices.robots.factory import make_robot
from ...uitls import *
from lerobot.scripts.control_robot import teleoperate

class TeleoperateCommand(RobotCommand):
    COMMAND='teleoperate'

    def __init__(self, description: str | None = None):
        
        super().__init__(
            name= self.COMMAND,
            help_text='Teleoperate the robot arms'
        )
        self.exec = teleoperate

    def execute(self, args:argparse.Namespace) -> int:
        init_logging()
        robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
        robot = make_robot(robot_cfg)
        self.exec(robot, args)
        return 0

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description,
        )
        self._register_parser(parser=parser)
        parser.add_argument(
            "--fps",
            type=none_or_int,
            default=1,
            help="Display all cameras on screen (set to 1 to display or 0).",
        )
        parser.add_argument(
            "--display-cameras",
            type=int,
            default=1,
            help="Display all cameras on screen (set to 1 to display or 0).",
        )   

    