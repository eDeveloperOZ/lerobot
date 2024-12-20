import argparse
from pathlib import Path

from .base import DatasetCommand
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub

class ConvertCommand(DatasetCommand):
    COMMAND='push'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'Convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub,or store it locally.',
            descriptio=description)
        self.exec = push_dataset_to_hub

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
        # TODO(rcadene): add automatic detection of the format
        self.parser.add_argument(
            "--raw-format",
            type=str,
            required=True,
            help="Dataset type (e.g. `pusht_zarr`, `umi_zarr`, `aloha_hdf5`, `xarm_pkl`, `dora_parquet`, `rlds`, `openx`).",
        )
        self.parser.add_argument(
            "--repo-id",
            type=str,
            required=True,
            help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
        )
        self.parser.add_argument(
            "--local-dir",
            type=Path,
            help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
        )
        self.parser.add_argument(
            "--push-to-hub",
            type=int,
            default=1,
            help="Upload to hub.",
        )
        self.parser.add_argument(
            "--fps",
            type=int,
            help="Frame rate used to collect videos. If not provided, use the default one specified in the code.",
        )
        self.parser.add_argument(
            "--video",
            type=int,
            default=1,
            help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size loaded by DataLoader for computing the dataset statistics.",
        )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of processes of Dataloader for computing the dataset statistics.",
        )
        self.parser.add_argument(
            "--episodes",
            type=int,
            nargs="*",
            help="When provided, only converts the provided episodes (e.g `--episodes 2 3 4`). Useful to test the code on 1 episode.",
        )
        self.parser.add_argument(
            "--force-override",
            type=int,
            default=0,
            help="When set to 1, removes provided output directory if it already exists. By default, raises a ValueError exception.",
        )
        self.parser.add_argument(
            "--resume",
            type=int,
            default=0,
            help="When set to 1, resumes a previous run.",
        )
        self.parser.add_argument(
            "--cache-dir",
            type=Path,
            required=False,
            default="/tmp",
            help="Directory to store the temporary videos and images generated while creating the dataset.",
        )
        self.parser.add_argument(
            "--tests-data-dir",
            type=Path,
            help=(
                "When provided, save tests artifacts into the given directory "
                "(e.g. `--tests-data-dir tests/data` will save to tests/data/{--repo-id})."
            ),
        )