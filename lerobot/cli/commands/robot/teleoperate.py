import argparse
import gymnasium as gym

from .base import RobotCommand
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.robot_devices.robots.factory import make_robot
from ...uitls import *
from lerobot.scripts.control_robot import teleoperate
from lerobot.scripts.control_sim_robot import teleoperate as teleoperate_sim

class TeleoperateCommand(RobotCommand):
    COMMAND='teleoperate'

    def __init__(self, description: str | None = None):
        
        super().__init__(
            name= self.COMMAND,
            help_text='Teleoperate the robot arms'
        )
        self.exec = teleoperate

    def execute(self, args:argparse.Namespace) -> int:
        init_logging()
        robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
        robot = make_robot(robot_cfg)

        env_constructor, env_cfg = self._init_simulation(args.sim)
        if env_cfg:
            # Simulation mode
            calib_kwgs = self._get_sim_calibration(robot, env_cfg)
            
            def process_actions(action):
                return real_positions_to_sim(action, **calib_kwgs)
                
            teleoperate_sim(env_constructor, robot, process_actions, **vars(args))
        else:
            # Real mode.
            self.exec(robot, **vars(args))

        return 0

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        self.parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description,
        )
        self._register_parser(parser=self.parser)
        self.parser.add_argument(
            "--fps",
            type=none_or_int,
            default=1,
            help="Display all cameras on screen (set to 1 to display or 0).",
        )
        self.parser.add_argument(
            "--display-cameras",
            type=int,
            default=1,
            help="Display all cameras on screen (set to 1 to display or 0).",
        )   

    