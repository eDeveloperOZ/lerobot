import argparse
from pathlib import Path

from .base import RobotCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.robot_devices.robots.factory import make_robot
from ...uitls import *
from lerobot.scripts.control_robot import record


class RecoredCommand(RobotCommand):
    COMMAND='record'

    def __init__(self, description: str | None = None):
        super().__init__(
            name = self.COMMAND,
            help_text = 'Record an episode'
        )
        self.exec = record

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
        self._register_parser(parser)
        task_args = parser.add_mutually_exclusive_group(required=True)
        task_args.add_argument(
            "--single-task",
            type=str,
            help="A short but accurate description of the task performed during the recording.",
        )

        # TODO(aliberts): add multi-task support
        # task_args.add_argument(
        #     "--multi-task",
        #     type=int,
        #     help="You will need to enter the task performed at the start of each episode.",
        # )
        parser.add_argument(
            "--fps",
            type=none_or_int,
            default=None,
            help="Frames per second (set to None to disable)",
        )
        parser.add_argument(
            "--root",
            type=Path,
            default=None,
            help="Root directory where the dataset will be stored (e.g. 'dataset/path').",
        )
        parser.add_argument(
            "--repo-id",
            type=str,
            default="lerobot/test",
            help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
        )
        parser.add_argument(
            "--local-files-only",
            type=int,
            default=0,
            help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
        )
        parser.add_argument(
            "--warmup-time-s",
            type=int,
            default=10,
            help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
        )
        parser.add_argument(
            "--episode-time-s",
            type=int,
            default=60,
            help="Number of seconds for data recording for each episode.",
        )
        parser.add_argument(
            "--reset-time-s",
            type=int,
            default=60,
            help="Number of seconds for resetting the environment after each episode.",
        )
        parser.add_argument(
            "--num-episodes",
            type=int,
            default=50,
            help="Number of episodes to record."
        )
        parser.add_argument(
            "--run-compute-stats",
            type=int,
            default=1,
            help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
        )
        parser.add_argument(
            "--push-to-hub",
            type=int,
            default=1,
            help="Upload dataset to Hugging Face hub.",
        )
        parser.add_argument(
            "--tags",
            type=str,
            nargs="*",
            help="Add tags to your dataset on the hub.",
        )
        parser.add_argument(
            "--num-image-writer-processes",
            type=int,
            default=0,
            help=(
                "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
                "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
                "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
                "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
            ),
        )
        parser.add_argument(
            "--num-image-writer-threads-per-camera",
            type=int,
            default=4,
            help=(
                "Number of threads writing the frames as png images on disk, per camera. "
                "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
                "Not enough threads might cause low camera fps."
            ),
        )
        parser.add_argument(
            "--resume",
            type=int,
            default=0,
            help="Resume recording on an existing dataset.",
        )
        parser.add_argument(
            "-p",
            "--pretrained-policy-name-or-path",
            type=str,
            help=(
                "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
                "saved using `Policy.save_pretrained`."
            ),
        )
        parser.add_argument(
            "--policy-overrides",
            type=str,
            nargs="*",
            help="Any key=value arguments to override config values (use dots for.nested=overrides)",
        )
        