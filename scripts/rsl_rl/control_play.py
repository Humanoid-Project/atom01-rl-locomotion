# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with keyboard (WASD) velocity control.

Controls:
    W / S : forward / backward
    A / D : turn left / turn right
    Q / E : strafe left / strafe right
    SPACE : stop
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with keyboard velocity control.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import carb
import omni

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atom01_standing.tasks  # noqa: F401
import atom01_walking.tasks   # noqa: F401

# 키 입력당 명령 속도 크기 (학습 시 명령 범위 내로 설정)
LIN_VEL_X = 0.5   # 전후 속도 (m/s)
LIN_VEL_Y = 0.2   # 좌우 속도 (m/s)
ANG_VEL_Z = 0.5   # yaw 회전 속도 (rad/s)


class KeyboardCommand:
    """WASD 키 입력을 (lin_vel_x, lin_vel_y, ang_vel_z) 명령으로 변환한다.

    여러 키를 동시에 누르면 명령이 합산된다 (예: W+A = 전진하며 좌회전).
    """

    def __init__(self, device: str):
        self.device = device
        self._pressed: set[str] = set()
        # 키 -> (lin_vel_x, lin_vel_y, ang_vel_z) 기여분
        self._key_map = {
            "W": (LIN_VEL_X, 0.0, 0.0),
            "S": (-LIN_VEL_X, 0.0, 0.0),
            "Q": (0.0, LIN_VEL_Y, 0.0),
            "E": (0.0, -LIN_VEL_Y, 0.0),
            "A": (0.0, 0.0, ANG_VEL_Z),
            "D": (0.0, 0.0, -ANG_VEL_Z),
        }
        self.command = torch.zeros(3, device=self.device)
        # 키보드 이벤트 구독
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event):
        key = getattr(event.input, "name", event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._key_map:
                self._pressed.add(key)
            elif key == "SPACE":
                self._pressed.clear()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
        self._update_command()

    def _update_command(self):
        cmd = [0.0, 0.0, 0.0]
        for key in self._pressed:
            contrib = self._key_map[key]
            cmd = [c + d for c, d in zip(cmd, contrib)]
        self.command = torch.tensor(cmd, device=self.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent under keyboard control."""
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 키보드로만 명령을 주기 위해 명령 자동 리샘플링을 사실상 비활성화
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = True
    # 에피소드 타임아웃 없이 계속 조작
    env_cfg.episode_length_s = 1.0e9

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    # rsl-rl < 4.0.0에서는 recurrent state 리셋에 네트워크 객체가 직접 필요
    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    # set up keyboard listener
    keyboard = KeyboardCommand(device=env.unwrapped.device)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    print(
        "[INFO] Keyboard controls:\n"
        "  W / S : forward / backward\n"
        "  A / D : turn left / turn right\n"
        "  Q / E : strafe left / strafe right\n"
        "  SPACE : stop"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # 모든 로봇에 키보드 명령 적용
            command_term.vel_command_b[:] = keyboard.command
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # 키보드 조작감을 위해 항상 실시간으로 실행
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
