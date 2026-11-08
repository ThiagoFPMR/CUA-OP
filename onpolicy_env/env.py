"""Environment construction shared by the runner and the preflight rehearsal."""
from __future__ import annotations

import argparse

from onpolicy_env.desktop_env.desktop_env import DesktopEnv


def add_env_args(p: argparse.ArgumentParser) -> None:
    """The provider knobs (they mirror the holo3 multienv runner's)."""
    p.add_argument("--path_to_vm", type=str, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--action_space", type=str, default="pyautogui")
    p.add_argument("--observation_type",
                   choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
                   default="screenshot")
    p.add_argument("--provider_name", type=str, default="vmware",
                   choices=["vmware", "virtualbox", "docker", "aws", "azure", "aliyun",
                            "volcengine", "fastvm"])
    p.add_argument("--region", type=str, default="us-east-1")
    p.add_argument("--client_password", type=str, default="")
    p.add_argument("--enable_proxy", action="store_true")
    p.add_argument("--screen_width", type=int, default=1920)
    p.add_argument("--screen_height", type=int, default=1080)


def build_env(args: argparse.Namespace) -> DesktopEnv:
    screen_size = (args.screen_width, args.screen_height)
    snapshot_name = "init_state"
    if args.provider_name == "aws":
        from onpolicy_env.desktop_env.providers.aws.manager import IMAGE_ID_MAP
        ami = IMAGE_ID_MAP[args.region].get(
            screen_size, IMAGE_ID_MAP[args.region][(1920, 1080)])
        snapshot_name = ami
    return DesktopEnv(
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        provider_name=args.provider_name,
        region=args.region,
        snapshot_name=snapshot_name,
        screen_size=screen_size,
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type in
        ("a11y_tree", "screenshot_a11y_tree", "som"),
        enable_proxy=args.enable_proxy,
        client_password=args.client_password,
    )
