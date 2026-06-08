# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)

def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity (yaw rotation)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize x/y base linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)



def _body_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected body positions in robot base frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_pos_w = body_pos_w - asset.data.root_pos_w.unsqueeze(1)

    num_bodies = body_pos_w.shape[1]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    body_pos_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    )
    return body_pos_b.reshape(env.num_envs, num_bodies, 3)

def feet_width_l2(env: ManagerBasedRLEnv, target_width: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet width deviation from target width."""
    feet_pos_b = _body_pos_b(env, asset_cfg)
    foot_width = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    return torch.square(foot_width - target_width)

def feet_centered_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet midpoint y offset from base center."""
    feet_pos_b = _body_pos_b(env, asset_cfg)
    feet_mid_y = 0.5 * (feet_pos_b[:, 0, 1] + feet_pos_b[:, 1, 1])
    return torch.square(feet_mid_y)

def feet_x_align_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize one foot being ahead of the other."""
    feet_pos_b = _body_pos_b(env, asset_cfg)
    feet_x_diff = feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0]
    return torch.square(feet_x_diff)