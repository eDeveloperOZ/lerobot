"""
Utilities to control a robot in simulation.

Useful to record a dataset, replay a recorded episode and record an evaluation dataset.

Examples of usage:


- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency.
  You can modify this value depending on how fast your simulation can run:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30 \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

Enable the --push-to-hub 1 to push the recorded dataset to the huggingface hub.

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/robot_sim_test \
    --episode-index 0
```

- Replay a sequence of test episodes: 
```bash
python lerobot/scripts/control_sim_robot.py replay \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --episode 0
```
Note: The seed is saved, therefore, during replay we can load the same environment state as the one during collection.

- Record a full dataset in order to train a policy,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --num-episodes 50 \
    --episode-time-s 30 \
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to reseting the environment.
- Tap right arrow key '->' to early exit while reseting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
"""

import importlib
import logging
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    init_policy,
    is_headless,
    log_control_info,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say

DEFAULT_FEATURES = {
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "seed": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
}


########################################################################################
# Utilities
########################################################################################
def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def init_sim_calibration(robot, cfg):
    # Constants necessary for transforming the joint pos of the real robot to the sim
    # depending on the robot discription used in that sim.
    start_pos = np.array(robot.leader_arms.main.calibration["start_pos"])
    axis_directions = np.array(cfg.get("axis_directions", [1]))
    offsets = np.array(cfg.get("offsets", [0])) * np.pi

    return {"start_pos": start_pos, "axis_directions": axis_directions, "offsets": offsets}


def real_positions_to_sim(real_positions, axis_directions, start_pos, offsets):
    """Counts - starting position -> radians -> align axes -> offset"""
    return axis_directions * (real_positions - start_pos) * 2.0 * np.pi / 4096 + offsets


########################################################################################
# Control modes
########################################################################################


def teleoperate(env, robot: Robot, process_action_fn, teleop_time_s=None):
    env = env()
    env.reset()
    start_teleop_t = time.perf_counter()
    while True:
        leader_pos = robot.leader_arms.main.read("Present_Position")
        action = process_action_fn(leader_pos)
        env.step(np.expand_dims(action, 0))
        if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
            print("Teleoperation processes finished.")
            break


def record(
    env,
    robot: Robot,
    process_action_from_leader,
    root: Path,
    repo_id: str,
    task: str,
    fps: int | None = None,
    tags: list[str] | None = None,
    pretrained_policy_name_or_path: str = None,
    policy_overrides: bool | None = None,
    episode_time_s: int = 30,
    num_episodes: int = 50,
    video: bool = True,
    push_to_hub: bool = True,
    num_image_writer_processes: int = 0,
    num_image_writer_threads_per_camera: int = 4,
    display_cameras: bool = False,
    play_sounds: bool = True,
    resume: bool = False,
    local_files_only: bool = False,
    run_compute_stats: bool = True,
) -> LeRobotDataset:
    # Load pretrained policy
    policy = None
    if pretrained_policy_name_or_path is not None:
        policy, policy_fps, device, use_amp = init_policy(pretrained_policy_name_or_path, policy_overrides)

        if fps is None:
            fps = policy_fps
            logging.warning(f"No fps provided, so using the fps from policy config ({policy_fps}).")

    if policy is None and process_action_from_leader is None:
        raise ValueError("Either policy or process_action_fn has to be set to enable control in sim.")

    # initialize listener before sim env
    listener, events = init_keyboard_listener()

    # create sim env
    env = env()

    # Create empty dataset or load existing saved episodes
    num_cameras = sum([1 if "image" in key else 0 for key in env.observation_space])

    # get image keys
    image_keys = [key for key in env.observation_space if "image" in key]
    state_keys_dict = env_cfg.state_keys

    if resume:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            local_files_only=local_files_only,
        )
        dataset.start_image_writer(
            num_processes=num_image_writer_processes,
            num_threads=num_image_writer_threads_per_camera * num_cameras,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, video)
    else:
        features = DEFAULT_FEATURES
        # add image keys to features
        for key in image_keys:
            shape = env.observation_space[key].shape
            if not key.startswith("observation.image."):
                key = "observation.image." + key
            features[key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}

        for key, obs_key in state_keys_dict.items():
            features[key] = {
                "dtype": "float32",
                "names": None,
                "shape": env.observation_space[obs_key].shape,
            }

        features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(repo_id, policy)
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            root=root,
            features=features,
            use_videos=video,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_image_writer_threads_per_camera * num_cameras,
        )

    recorded_episodes = 0
    while True:
        log_say(f"Recording episode {dataset.num_episodes}", play_sounds)

        if events is None:
            events = {"exit_early": False}

        if episode_time_s is None:
            episode_time_s = float("inf")

        timestamp = 0
        start_episode_t = time.perf_counter()

        seed = np.random.randint(0, 1e5)
        observation, info = env.reset(seed=seed)

        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()

            if policy is not None:
                action = predict_action(observation, policy, device, use_amp)
            else:
                leader_pos = robot.leader_arms.main.read("Present_Position")
                action = process_action_from_leader(leader_pos)

            observation, reward, terminated, _, info = env.step(action)

            success = info.get("is_success", False)
            env_timestamp = info.get("timestamp", dataset.episode_buffer["size"] / fps)

            frame = {
                "action": torch.from_numpy(action),
                "next.reward": reward,
                "next.success": success,
                "seed": seed,
                "timestamp": env_timestamp,
            }

            for key in image_keys:
                if not key.startswith("observation.image"):
                    frame["observation.image." + key] = observation[key]
                else:
                    frame[key] = observation[key]

            for key, obs_key in state_keys_dict.items():
                frame[key] = torch.from_numpy(observation[obs_key])

            dataset.add_frame(frame)

            if display_cameras and not is_headless():
                for key in image_keys:
                    cv2.imshow(key, cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_control_info(robot, dt_s, fps=fps)

            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"] or terminated:
                events["exit_early"] = False
                break

        if events["rerecord_episode"]:
            log_say("Re-record episode", play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode(task=task)
        recorded_episodes += 1

        if events["stop_recording"] or recorded_episodes >= num_episodes:
            break
        else:
            logging.info("Waiting for a few seconds before starting next episode recording...")
            busy_wait(3)

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    if run_compute_stats:
        logging.info("Computing dataset statistics")
    dataset.consolidate(run_compute_stats)

    if push_to_hub:
        dataset.push_to_hub(tags=tags)

    log_say("Exiting", play_sounds)
    return dataset


def replay(
    env, root: Path, repo_id: str, episode: int, fps: int | None = None, local_files_only: bool = True
):
    env = env()

    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root, local_files_only=local_files_only)
    items = dataset.hf_dataset.select_columns("action")
    seeds = dataset.hf_dataset.select_columns("seed")["seed"]

    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()
    env.reset(seed=seeds[from_idx].item())
    logging.info("Replaying episode")
    log_say("Replaying episode", play_sounds=True)
    for idx in range(from_idx, to_idx):
        start_episode_t = time.perf_counter()
        action = items[idx]["action"]
        env.step(action.unsqueeze(0).numpy())
        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / fps - dt_s)

