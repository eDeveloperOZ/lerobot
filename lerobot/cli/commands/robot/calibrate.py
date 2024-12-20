# lerobot/cli/commands/robot/calibrate.py
import argparse

from .base import RobotCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.scripts.control_robot import calibrate 

class CalibrateCommand(RobotCommand):
    COMMAND='calibrate'

    def __init__(self, description: str | None = None):
        super().__init__(
            name=self.COMMAND,
            help_text="Calibrate robot arms",
        )
        self.exec = calibrate

    def execute(self, args: argparse.Namespace) -> int:
        if args.sim:
            raise ValueError("Calibration is not supported in simulation mode")
        
        init_logging()
        robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
        robot = make_robot(robot_cfg)
        self.exec(robot, args.arms)
        # Finsih the program successfuly 
        return 0

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description,
        )
        self._register_parser(parser)
        parser.add_argument(
            "--arms",
            type=str,
            nargs="*",
            help="List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)",
        )

    