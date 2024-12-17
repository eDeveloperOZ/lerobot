# lerobot/cli/commands/model/train.py
import hydra
from omegaconf import DictConfig
import logging
import argparse

from .base import ModelCommand
from lerobot.scripts.train import train

class TrainCommand(ModelCommand):
    COMMAND='train'

    def __init__(self):
        super().__init__(
            name=self.COMMAND,
            help_text="Train a policy model using hydra configuration"
        )
    
    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.help
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            # Use hydra's CLI 
            @hydra.main(version_base="1.2", config_name="default", config_path="../configs")
            def train_cli(cfg: DictConfig):
                return train(
                    cfg,
                    out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
                    job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
                )
            
            train_cli()
            return 0
        except Exception as e:
            logging.error(str(e))
            return 1