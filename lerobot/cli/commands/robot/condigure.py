from abc import ABCMeta
import argparse

from .base import RobotCommand
from ...uitls import *
from lerobot.scripts.configure_motor import configure_motor

class ConfigureCommand(RobotCommand):
    COMMAND = 'configure-motor'

    def __init__(self, description: str | None = None):
        super().__init__(
            name=self.COMMAND,
            help_text='Configure motor id'
        )
        self.exec = configure_motor

    def execute(self, args: argparse.Namespace) -> int:
        self.exec(
            port=self.args.port,
            brand=self.args.brand,
            model=self.args.model,
            id=self.args.ID,
            baudrate_des=self.args.baudrate
        )

    def register(self,  subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name,
            self.help,
            description=self.description
        )
        self._register_parser(parser=parser)
        parser.add_argument(
            "--port", 
            type=str, 
            required=True, 
            help="Motors bus port (e.g. dynamixel,feetech)"
        )
        parser.add_argument(
            "--brand", 
            type=str, 
            required=True, 
            help="Motor brand (e.g. dynamixel,feetech)"
        )
        parser.add_argument(
            "--model", 
            type=str, 
            required=True, 
            help="Motor model (e.g. xl330-m077,sts3215)"
        )
        parser.add_argument(
            "--ID", 
            type=int, 
            required=True, 
            help="Desired ID of the current motor (e.g. 1,2,3)"
        )
        parser.add_argument(
            "--baudrate", 
            type=int, 
            default=1000000, 
            help="Desired baudrate for the motor (default: 1000000)"
        )
        self.args = parser.parse_args()