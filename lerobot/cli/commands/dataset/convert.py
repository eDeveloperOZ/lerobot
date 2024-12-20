import argparse
from pathlib import Path

from .base import DatasetCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
# TODO: remove this, either to utils for commands or when removing the hydra Necessity 
from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import parse_robot_config
from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset
# TODO: import batch convert and call it with a -b --batch flag

class ConvertCommand(DatasetCommand):
    COMMAND='convert'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'Convert a dataset from V1 to V2',
            descriptio=description)
        self.exec = convert_dataset

    def execute(self, args: argparse.Namespace) -> int:
        init_logging()
        if not args.local_dir:
            args.local_dir = Path("/tmp/lerobot_dataset_v2")

        robot_cfg = None
        if args.robot_config:
            robot_cfg = parse_robot_config(args.robot_config, args.robot_overrides)
        del args.robot_config, args.robot_overrides
        self.exec(**vars(args), robot_config=robot_cfg)
        return 0
    
    def register_parser(self, subparser: argparse._SubParsersAction) -> None:
        self.parser = subparser.add_parser(
            self.name,
            help=self.help,
            description=self.description
        )
        self._register_parser(self.parser)
        task_args = self.parser.add_mutually_exclusive_group(required=True)
        task_args.add_argument(
            "--single-task",
            type=str,
            help="A short but accurate description of the single task performed in the dataset.",
        )
        task_args.add_argument(
            "--tasks-col",
            type=str,
            help="The name of the column containing language instructions",
        )
        task_args.add_argument(
            "--tasks-path",
            type=Path,
            help="The path to a .json file containing one language instruction for each episode_index",
        )
        self.parser.add_argument(
            "--repo-id",
            type=str,
            required=True,
            help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
        )
        self.parser.add_argument(
            "--robot-config",
            type=Path,
            default=None,
            help="Path to the robot's config yaml the dataset during conversion.",
        )
        self.parser.add_argument(
            "--robot-overrides",
            type=str,
            nargs="*",
            help="Any key=value arguments to override the robot config values (use dots for.nested=overrides)",
        )
        self.parser.add_argument(
            "--local-dir",
            type=Path,
            default=None,
            help="Local directory to store the dataset during conversion. Defaults to /tmp/lerobot_dataset_v2",
        )
        self.parser.add_argument(
            "--license",
            type=str,
            default="apache-2.0",
            help="Repo license. Must be one of https://huggingface.co/docs/hub/repositories-licenses. Defaults to mit.",
        )
        self.parser.add_argument(
            "--test-branch",
            type=str,
            default=None,
            help="Repo branch to test your conversion first (e.g. 'v2.0.test')",
        )