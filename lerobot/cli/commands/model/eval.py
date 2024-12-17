# lerobot/cli/commands/model/eval.py
import argparse
import logging

from .base import ModelCommand
from lerobot.scripts.eval import get_pretrained_policy_path, main as eval_main

class EvalCommand(ModelCommand):
    COMMAND='eval'
    
    def __init__(self):
        super().__init__(
            name=self.COMMAND,
            help_text="Evaluate a policy model on an environment by running rollouts"
        )



    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=__doc__,
        )
        # Keep original eval.py argparse arguments
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-p",
            "--pretrained-policy-name-or-path",
            help=(
                "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
                "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
                "(useful for debugging). This argument is mutually exclusive with `--config`."
            ),
        )
        group.add_argument(
            "--config",
            help=(
                "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
                "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
            ),
        )
        parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
        parser.add_argument(
            "--out-dir",
            help=(
                "Where to save the evaluation outputs. If not provided, outputs are saved in "
                "outputs/eval/{timestamp}_{env_name}_{policy_name}"
            ),
        )
        parser.add_argument(
            "overrides",
            nargs="*",
            help="Any key=value arguments to override config values (use dots for.nested=overrides)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if args.pretrained_policy_name_or_path is None:
                eval_main(
                    hydra_cfg_path=args.config,
                    out_dir=args.out_dir,
                    config_overrides=args.overrides
                )
            else:
                pretrained_policy_path = get_pretrained_policy_path(
                    args.pretrained_policy_name_or_path,
                    revision=args.revision
                )
                eval_main(
                    pretrained_policy_path=pretrained_policy_path,
                    out_dir=args.out_dir,
                    config_overrides=args.overrides
                )
            return 0
        except Exception as e:
            logging.error(str(e))
            return 1