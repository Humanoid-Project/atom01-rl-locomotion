# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 # 환경당 데이터 수집 step 수
    max_iterations = 4000 # 총 학습 반복 횟수
    save_interval = 500 # 모델 체크포인트 저장 주기 (iteration 단위)
    experiment_name = "atom01_standing_real"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, # 초기 탐색 노이즈 크기 (학습 진행에 따라 감소)
        actor_obs_normalization=True, # 입력 observation 정규화 여부 (단위가 다른 값이 섞이면 True 권장)
        critic_obs_normalization=True, # Critic 입력 정규화 여부
        actor_hidden_dims=[128, 128], # Actor 네트워크 은닉층 구조
        critic_hidden_dims=[128, 128], # Critic 네트워크 은닉층 구조
        activation="elu", # 은닉층 활성화 함수 (elu, relu, tanh)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # Critic 손실 가중치
        use_clipped_value_loss=True, # Critic 손실 클리핑 사용 여부
        clip_param=0.2, # 정책 업데이트 제한 범위
        entropy_coef=0.01, # 탐색 장려 가중치 (클수록 다양한 행동 시도)
        num_learning_epochs=5, # 수집한 데이터를 몇 번 반복 학습할지
        num_mini_batches=4, # 데이터를 몇 등분해서 학습할지
        learning_rate=1.0e-3, # 신경망 학습률
        schedule="adaptive", # 학습률 조정 방식
        gamma=0.99, # 미래 보상을 얼마나 중요하게 볼지 (1에 가까울수록 먼 미래까지 고려)
        lam=0.95, # GAE 파라미터
        desired_kl=0.01, # 목표 KL divergence
        max_grad_norm=1.0, # 그래디언트 클리핑 최대값 (학습 발산 방지)
    )