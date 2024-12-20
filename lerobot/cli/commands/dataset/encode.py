import argparse
from pathlib import Path

from .base import DatasetCommand
from lerobot.common.utils.utils import init_logging
# TODO: remove this, either to utils for commands or when removing the hydra Necessity 
from lerobot.common.datasets.push_dataset_to_hub._encode_datasets import encode_datasets

class ConvertCommand(DatasetCommand):
    COMMAND='encode'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'batch encode lerobot dataset from their raw format to LeRobotDataset and push their updated version to the hub',
            descriptio=description)
        self.exec = encode_datasets

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
            default=Path("data"),
            help="Directory where raw datasets are located.",
        )
        self.parser.add_argument(
            "--raw-repo-ids",
            type=str,
            nargs="*",
            default=["lerobot-raw"],
            help="""Raw dataset repo ids. if 'lerobot-raw', the keys from `AVAILABLE_RAW_REPO_IDS` will be
                used and raw datasets will be fetched from the 'lerobot-raw/' repo and pushed with their
                associated format. It is assumed that each dataset is located at `raw_dir / raw_repo_id` """,
        )
        self.parser.add_argument(
            "--raw-format",
            type=str,
            default=None,
            help="""Raw format to use for the raw repo-ids. Must be specified if --raw-repo-ids is not
                'lerobot-raw'""",
        )
        self.parser.add_argument(
            "--local-dir",
            type=Path,
            default=None,
            help="""When provided, writes the dataset converted to LeRobotDataset format in this directory
            (e.g. `data/lerobot/aloha_mobile_chair`).""",
        )
        self.parser.add_argument(
            "--push-repo",
            type=str,
            default="lerobot",
            help="Repo to upload datasets to",
        )
        self.parser.add_argument(
            "--vcodec",
            type=str,
            default="libsvtav1",
            help="Codec to use for encoding videos",
        )
        self.parser.add_argument(
            "--pix-fmt",
            type=str,
            default="yuv420p",
            help="Pixel formats (chroma subsampling) to be used for encoding",
        )
        self.parser.add_argument(
            "--g",
            type=int,
            default=2,
            help="Group of pictures sizes to be used for encoding.",
        )
        self.parser.add_argument(
            "--crf",
            type=int,
            default=30,
            help="Constant rate factors to be used for encoding.",
        )
        self.parser.add_argument(
            "--tests-data-dir",
            type=Path,
            default=None,
            help=(
                "When provided, save tests artifacts into the given directory "
                "(e.g. `--tests-data-dir tests/data` will save to tests/data/{--repo-id})."
            ),
        )
        self.parser.add_argument(
            "--dry-run",
            type=int,
            default=0,
            help="If not set to 0, this script won't download or upload anything.",
        )
        