import inspect
import sys
from argparse import ArgumentError
from functools import wraps
from pathlib import Path
from typing import Sequence
import sys

import draccus

from lerobot.common.utils.utils import has_method

PATH_KEY = "path"
draccus.set_config_type("json")


def get_cli_overrides(
    field_name: str, args: Sequence[str] | None = None
) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (
        f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=",
        f"--{field_name}.{PATH_KEY}=",
    )
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(
    fields_to_filter: str | list[str], args: Sequence[str] | None = None
) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = args
    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [
                arg for arg in filtered_args if not arg.startswith(f"--{field}.")
            ]

    return filtered_args


def filter_dict_args(
    fields_to_filter: str | list[str], args: Sequence[str] | None = None
) -> list[str]:
    """Remove CLI options targeting nested dictionary keys."""
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    if args is None:
        args = sys.argv[1:]

    filtered_args = []
    for arg in args:
        if any(arg.startswith(f"--{field}.") for field in fields_to_filter):
            continue
        filtered_args.append(arg)
    return filtered_args


def parse_nested_dict_args(
    field_name: str, args: Sequence[str] | None = None
) -> dict[str, dict[str, str]]:
    """Return dict of overrides for ``field_name`` CLI arguments."""
    if args is None:
        args = sys.argv[1:]
    overrides: dict[str, dict[str, str]] = {}
    prefix = f"--{field_name}."
    for arg in args:
        if not arg.startswith(prefix):
            continue
        item = arg[len(prefix) :]
        if "=" not in item or "." not in item:
            continue
        key, value = item.split("=", 1)
        entry, subkey = key.split(".", 1)
        overrides.setdefault(entry, {})[subkey] = value
    return overrides


def apply_gateway_overrides(robot_cfg, args: Sequence[str] | None = None) -> None:
    """Apply Gateway bus CLI overrides to ``robot_cfg``."""
    from lerobot.common.robot_devices.motors.configs import GatewayMotorsBusConfig

    leader = parse_nested_dict_args("robot.leader_arms", args)
    follower = parse_nested_dict_args("robot.follower_arms", args)

    for name, fields in leader.items():
        if fields.get("type") != "gateway" or "url" not in fields:
            continue
        motors = getattr(robot_cfg.leader_arms.get(name), "motors", None)
        robot_cfg.leader_arms[name] = GatewayMotorsBusConfig(
            url=fields["url"], motors=motors, mock=robot_cfg.mock
        )

    for name, fields in follower.items():
        if fields.get("type") != "gateway" or "url" not in fields:
            continue
        motors = getattr(robot_cfg.follower_arms.get(name), "motors", None)
        robot_cfg.follower_arms[name] = GatewayMotorsBusConfig(
            url=fields["url"], motors=motors, mock=robot_cfg.mock
        )


def wrap(config_path: Path | None = None):
    """
    HACK: Similar to draccus.wrap but does two additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
    """

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)
                if has_method(argtype, "__get_dict_fields__"):
                    dict_fields = argtype.__get_dict_fields__()
                    cli_args = filter_dict_args(dict_fields, cli_args)
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    cli_args = filter_arg("config_path", cli_args)
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    cfg = draccus.parse(
                        config_class=argtype, config_path=config_path, args=cli_args
                    )
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer
