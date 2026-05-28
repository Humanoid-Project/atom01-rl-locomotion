# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.sensors import ContactSensorCfg # 접촉센서
from isaaclab.sensors import ImuCfg # IMU

from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg # 센서 노이즈


LEG_JOINTS = [
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


@configclass
class RobonexTrainSceneCfg(InteractiveSceneCfg):
    """Configuration for an Atom01 training scene."""

    # ground 설정
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot 설정
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/polygon/robonex_ws/atom01_description/usd/atom01.usd",
            activate_contact_sensors=True, # 접촉센서 쓰려면 추가해야 함
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.72),  # 바닥 위 0.72m에서 시작
            
            # 기본 자세 정의
            joint_pos={
                "left_thigh_yaw_joint": 0.0,
                "left_thigh_roll_joint": 0.0,
                "left_thigh_pitch_joint": -0.35,
                "left_knee_joint": 0.7,
                "left_ankle_pitch_joint": -0.337,
                "left_ankle_roll_joint": 0.0,

                "right_thigh_yaw_joint": 0.0,
                "right_thigh_roll_joint": 0.0,
                "right_thigh_pitch_joint": -0.35,
                "right_knee_joint": 0.7,
                "right_ankle_pitch_joint": -0.337,
                "right_ankle_roll_joint": 0.0,

                "torso_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*thigh.*", ".*knee.*", ".*ankle.*"],
                stiffness=100.0,
                damping=5.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=200.0,
                damping=10.0,
            ),
        },
    )

    # 접촉센서
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # IMU
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 기본 자세에서 관절을 얼마나 움직일 것인가를 학습
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale={
            ".*yaw.*": 0.08,
            ".*roll.*": 0.08,
            ".*thigh_pitch.*": 0.20,
            ".*knee.*": 0.25,
            ".*ankle_pitch.*": 0.15,
        },
        use_default_offset=True, # 절대 관절각이 아닌 기본 자세에서의 변화량 사용
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # 관측할 값 정의 (실제 로봇으로부터 받을 값)
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 관절 각도
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )

        # 관절 속도
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.03),
        )

        # IMU 각속도
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05),
                bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05),
            ),
        )

        # 중력 방향
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )

        # 이전 Action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True # 노이즈 적용 여부
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 엑추에이터 gain 
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # base_link 초기화
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )
    # 다리 joint 초기화
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.02, 0.02),
        },
    )
    # 몸통 회전 joint 초기화
    reset_torso_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"]),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.02, 0.02),
        },
    )
    
    # 외부 충격
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0,8.0),
        params={
            "velocity_range":{
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                #"yaw": (-0.5, 0.5),
            },
        },
    )
    
    # 마찰 랜덤화
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0), # 0.0이면 미끄러운 바닥
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    # 관절 마찰 랜덤화
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    
    # 질량 랜덤화
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "torso_link"]),
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
    )


# 보상함수 정의
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 매 step마다 살아있으면 보상
    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0
    )

    # 쓰러져서 에피소드 끝나면 페널티
    terminating = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0
    )

    # 몸통 기울기 페널티
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # 수직 방향 흔들림 억제
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
    )

    # yaw 회전 억제
    ang_vel_z = RewTerm(
        func=mdp.ang_vel_z_l2,
        weight=-0.3,
    )

    # 로봇 높이 페널티
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "target_height": 0.72,
        },
    )
    
    # 다리 관절이 기본자세에서 벗어나면 페널티
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINTS,
            ),
        },
    )

    # 관절 토크 억제
    dof_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
    )

    # 지난번이랑 너무 다른 action을 내면 페널티
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    feet_width = RewTerm(
        func=mdp.feet_width_l2,
        weight=-5.0,
        params={
            "target_width": 0.145,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"]), "bounds": (-3.0, 3.0)},
    )
    fall_down = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


@configclass
class RobonexTrainEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobonexTrainSceneCfg = RobonexTrainSceneCfg(
        num_envs=4096, # 동시에 실행할 로봇 수
        env_spacing=4.0, # 각 로봇 사이 간격(m)
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""

        self.decimation = 2
        self.episode_length_s = 15

        self.viewer.eye = (8.0, 0.0, 5.0)

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation