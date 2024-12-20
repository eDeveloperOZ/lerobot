import argparse
from pathlib import Path

from .base import RobotCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.robot_devices.robots.factory import make_robot
from ...uitls import *
from lerobot.scripts.control_robot import replay
from lerobot.scripts.control_sim_robot import replay as replay_sim


class ReplayCommand(RobotCommand):
    COMMAND='replay'

    def __init__(self, description: str | None = None):
        super().__init__(
            name = self.COMMAND, 
            help_text = 'Replay an episode from the dataSet',
        )
        self.exec = replay

    def execute(self, args:argparse.Namespace) -> int:
        init_logging()
        robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
        robot = make_robot(robot_cfg)
        env_constructor, env_cfg = self._init_simulation(args.sim)
        if env_cfg:
            # Simulation mode
            replay_sim(env_constructor, **vars(args))
        else:
            # Real robot mode
            self.exec(robot, **vars(args))

        return 0
    
    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        self.parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description,
        )
        self._register_parser(parser=self.parser)
        self.parser.add_argument(
            "--fps",
            type=none_or_int,
            default=None, 
            help="Frames per second (set to None to disable)"
        )
        self.parser.add_argument(
            "--root",
            type=Path,
            default=None,
            help="Root directory where the dataset will be stored (e.g. 'dataset/path').",
        )
        self.parser.add_argument(
            "--repo-id",
            type=str,
            default="lerobot/test",
            help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
        )
        self.parser.add_argument(
            "--local-files-only",
            type=int,
            default=0,
            help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
        )
        self.parser.add_argument(
            "--episode",    
            type=int, 
            default=0, 
            help="Index of the episode to replay."
        )
