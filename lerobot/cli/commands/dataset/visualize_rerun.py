# lerobot/cli/commands/dataset/visualize.py

import argparse
from pathlib import Path

from .base import DatasetCommand
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.visualize_dataset import visualize_dataset

class ConvertCommand(DatasetCommand):
    COMMAND='visualize-rerun'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.',
            descriptio=description)
        self.exec = visualize_dataset

    def execute(self, args: argparse.Namespace) -> int:
        init_logging()
        dataset = LeRobotDataset(args.repo_id, root=args.root, local_files_only=args.local_files_only)

        self.exec(dataset, **vars(args))
        return 0
    
    def register_parser(self, subparser: argparse._SubParsersAction) -> None:
        self.parser = subparser.add_parser(
            self.name,
            help=self.help,
            description=self.description
        )
        self._register_parser(self.parser)
        self.parser.add_argument(
            "--repo-id",
            type=str,
            required=True,
            help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
        )
        self.parser.add_argument(
            "--episode-index",
            type=int,
            required=True,
            help="Episode to visualize.",
        )
        self.parser.add_argument(
            "--local-files-only",
            type=int,
            default=0,
            help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
        )
        self.parser.add_argument(
            "--root",
            type=Path,
            default=None,
            help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
        )
        self.parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory path to write a .rrd file when `--save 1` is set.",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size loaded by DataLoader.",
        )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of processes of Dataloader for loading the data.",
        )
        self.parser.add_argument(
            "--mode",
            type=str,
            default="local",
            help=(
                "Mode of viewing between 'local' or 'distant'. "
                "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
                "'distant' creates a server on the distant machine where the data is stored. "
                "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
            ),
        )
        self.parser.add_argument(
            "--web-port",
            type=int,
            default=9090,
            help="Web port for rerun.io when `--mode distant` is set.",
        )
        self.parser.add_argument(
            "--ws-port",
            type=int,
            default=9087,
            help="Web socket port for rerun.io when `--mode distant` is set.",
        )
        self.parser.add_argument(
            "--save",
            type=int,
            default=0,
            help=(
                "Save a .rrd file in the directory provided by `--output-dir`. "
                "It also deactivates the spawning of a viewer. "
                "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
            ),
        )