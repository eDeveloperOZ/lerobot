# lerobot/cli/commands/robot/calibrate.py
import argparse
from .base import RobotCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.scripts.control_robot import calibrate as calibrate_robot

class CalibrateCommand(RobotCommand):
    def __init__(self):
        super().__init__(
            name="calibrate",
            help_text="Calibrate robot arms",
        )

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description,
        )
        self.add_robot_args(parser)
        parser.add_argument(
            "--arms",
            type=str,
            nargs="*",
            help="List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        init_logging()
        robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
        robot = make_robot(robot_cfg)
        calibrate_robot(robot, args.arms)
        return 0