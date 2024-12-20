import argparse
from pathlib import Path

from .base import DatasetCommand
from lerobot.common.utils.utils import init_logging
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw

class DownloadRawCommand(DatasetCommand):
    COMMAND='download-raw'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'This file contains download scripts for raw datasets.',
            descriptio=description)
        self.exec = download_raw

    def execute(self, args: argparse.Namespace) -> int:
        init_logging()
        self.exec(**vars(args))
        return 0
    
    def register_parser(self, subparser: argparse._SubParsersAction) -> None:
        self.parser = subparser.add_parser(
            self.name,
            help=self.help,
            description=self.description
        )
        self._register_parser(self.parser)
        self.parser.add_argument(
            "--raw-dir",
            type=Path,
            required=True,
            help="Directory containing input raw datasets (e.g. `data/aloha_mobile_chair_raw` or `data/pusht_raw).",
        )
        self.parser.add_argument(
            "--repo-id",
            type=str,
            required=True,
            help="""Repositery identifier on Hugging Face: a community or a user name `/` the name of
            the dataset (e.g. `lerobot/pusht_raw`, `cadene/aloha_sim_insertion_human_raw`).""",
        )