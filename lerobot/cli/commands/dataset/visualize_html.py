# lerobot/cli/commands/dataset/visualize_html.py
import argparse
from pathlib import Path

from .base import DatasetCommand
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.visualize_dataset_html import visualize_dataset_html

class VisualizeHtmlCommand(DatasetCommand):
    COMMAND = 'visualize-html'

    def __init__(self, description: str | None = None):
        super().__init__(
            name=self.COMMAND,
            help_text='Visualize dataset in HTML format with interactive plots and video playback.',
            description=description
        )
        self.exec = visualize_dataset_html

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
        
        # Add all the arguments from visualize_dataset_html.py
        self.parser.add_argument(
            "--repo-id",
            type=str,
            required=True,
            help="Name of hugging face repository containing a LeRobotDataset dataset",
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
            help="Root directory for a dataset stored locally (e.g. `--root data`).",
        )
        self.parser.add_argument(
            "--episodes",
            type=int,
            nargs="*",
            default=None,
            help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6).",
        )
        self.parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory path to write html files and kickoff a web server.",
        )
        self.parser.add_argument(
            "--serve",
            type=int,
            default=1,
            help="Launch web server.",
        )
        self.parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Web host used by the http server.",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            default=9090,
            help="Web port used by the http server.",
        )
        self.parser.add_argument(
            "--force-override",
            type=int,
            default=0,
            help="Delete the output directory if it exists already.",
        )