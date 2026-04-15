from gs_env.sim.envs.config.schema import (
    EnvArgs,
    GenesisInitArgs,
    MotionEnvArgs,
    WalkingEnvArgs,
)
from gs_env.sim.robots.config.registry import RobotArgsRegistry
from gs_env.sim.scenes.config.registry import SceneArgsRegistry

# ------------------------------------------------------------
# Genesis init
# ------------------------------------------------------------


GenesisInitArgsRegistry: dict[str, GenesisInitArgs] = {}


GenesisInitArgsRegistry["default"] = GenesisInitArgs(
    seed=0,
    precision="32",
    logging_level="info",
    backend=None,
)


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------


EnvArgsRegistry: dict[str, EnvArgs] = {}


# ------------------------------------------------------------
# G1 Walking
# ------------------------------------------------------------


EnvArgsRegistry["g1_walk"] = WalkingEnvArgs(
    env_name="WalkingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Velocity Tracking ###
        "LinVelXYReward": 10.0,
        "AngVelZReward": 10.0,
        "LinVelZPenalty": 20.0,
        "AngVelXYPenalty": 0.5,
        ### Pose Tracking ###
        "OrientationPenalty": 100.0,
        ### Regularization ###
        # "TorquePenalty": 0.0001,
        "ActionRatePenalty": 0.1,
        "DofPosLimitPenalty": 100.0,
        "DofVelPenalty": 0.02,
        "ActionLimitPenalty": 0.1,
        ### Motion Constraints ###
        "HipYawPenalty": 5.0,
        "HipRollPenalty": 5.0,
        "UpperBodyActionPenalty": 0.2,
        "BodyRollPenalty": 100.0,
        "WaistRollPenalty": 50.0,
        "FeetAirTimePenalty": 100.0,
        "G1FeetSlidePenalty": 2.0,
        "G1FeetHeightPenalty": 100.0,
        "FeetOrientationPenalty": 30.0,
        "FeetContactForceLimitPenalty": 1.0,
    },
    img_resolution=(480, 270),
    action_latency=0,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "base_ang_vel": 0.5,
    },
    obs_noises={
        "dof_pos": 0.01,
        "dof_vel": 0.02,
        "projected_gravity": 0.05,
        "base_ang_vel": 0.2,
    },
    actor_obs_terms=[
        "last_action",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
        "base_ang_vel",
        "commands",
    ],
    critic_obs_terms=[
        "last_action",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
        "base_lin_vel",
        "base_ang_vel",
        "commands",
        "feet_height",
        "foot_contact_weighted",
    ],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    command_resample_time=10.0,
    commands_range=(
        (-1.0, 2.0),  # Forward/Backward
        (-0.0, 0.0),  # Left/Right
        (-2.0, 2.0),  # Turn
    ),
)


EnvArgsRegistry["g1_motion_teacher"] = MotionEnvArgs(
    env_name="MotionEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_g1_mocap"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Motion Tracking ###
        "DofPosReward": 3.0,
        "DofVelReward": 0.02,
        "BaseHeightReward": 100.0,
        "BaseLinVelReward": 20.0,
        "BaseQuatReward": 20.0,
        "BaseAngVelReward": 0.2,
        "TrackingLinkPosGlobalReward": 20.0,
        "TrackingLinkPosLocalReward": 20.0,
        "TrackingLinkQuatReward": 1.0,
        "TrackingLinkLinVelReward": 1.0,
        "FootContactReward": 3.0,
        "FootContactPenalty": 5.0,
        ### Regularization ###
        "TorquePenalty": 0.0001,
        "DofVelPenalty": 0.02,
        "ActionRatePenalty": 0.2,
        "AnkleTorquePenalty": 0.003,
        "BodyAngVelXYPenalty": 1.0,
        "BodyRollPenalty": 20.0,
        "BaseRollPenalty": 20.0,
        "WaistVelPenalty": 0.2,
        "WaistRollPenalty": 10.0,
        # "HipRollPenalty": 5.0,
        # "HipYawPenalty": 5.0,
        "G1FeetSlidePenalty": 5.0,
        "MotionFeetAirTimePenalty": 200.0,
        "MotionStandStillFeetContactPenalty": 50.0,
        "FootOrientationPenalty": 50.0,
        "FeetContactForceLimitPenalty": 4.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "diff_dof_vel": 0.1,
    },
    obs_noises={},
    actor_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "base_rotation_6D",
        "diff_base_yaw",
        "projected_gravity",
        # Motion Difference
        "diff_dof_pos",
        "diff_dof_vel",
        "diff_base_rotation_6D",
        "diff_base_euler",
        "diff_base_ang_vel_local",
        "diff_base_pos_local_yaw",
        "diff_base_lin_vel_local",
        "diff_tracking_link_pos_global_local_yaw",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "diff_tracking_link_lin_vel_local_yaw",
        "diff_tracking_link_ang_vel_local_yaw",
        # Reference
        "motion_obs_history",
        # Privileged
        "dr_obs",
        "base_lin_vel_local",
        "tracking_link_pos_local_yaw",
        "foot_contact_weighted",
        "ref_foot_contact_weighted",
    ],
    critic_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "base_rotation_6D",
        "diff_base_yaw",
        "projected_gravity",
        # Motion Difference
        "diff_dof_pos",
        "diff_dof_vel",
        "diff_base_rotation_6D",
        "diff_base_euler",
        "diff_base_ang_vel_local",
        "diff_base_pos_local_yaw",
        "diff_base_lin_vel_local",
        "diff_tracking_link_pos_global_local_yaw",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "diff_tracking_link_lin_vel_local_yaw",
        "diff_tracking_link_ang_vel_local_yaw",
        # Reference
        "motion_obs_history",
        # Privileged
        "dr_obs",
        "base_lin_vel_local",
        "tracking_link_pos_local_yaw",
        "foot_contact_weighted",
        "ref_foot_contact_weighted",
    ],
    reset_yaw_range=(-0.15, 0.15),
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    tracking_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "torso_link",
        "pelvis",
    ],
    dof_weights={
        "hip_roll": 1.0,
        "hip_pitch": 0.6,
        "hip_yaw": 1.0,
        "knee": 0.6,
        "ankle": 0.3,
        "waist": 2.0,
        "shoulder": 1.0,
        "elbow": 0.6,
        "wrist": 0.3,
    },
    link_pos_global_weights={
        "ankle": 1.0,
        "wrist": 0.0,
        "torso": 1.0,
        "pelvis": 1.0,
    },
    link_pos_local_weights={
        "ankle": 1.0,
        "wrist": 2.0,
        "torso": 0.0,
        "pelvis": 0.0,
    },
    link_quat_weights={
        "ankle": 0.0,
        "wrist": 2.0,
        "torso": 5.0,
    },
    no_terminate_before_motion_time=1.0,
    no_terminate_after_reset_time=2.0,
    no_terminate_after_random_push_time=2.0,
    # [initial_threshold, [min_threshold, max_threshold]]
    terminate_after_error={
        "base_pos_error": [2.0, [0.1, 2.0]],
        "base_quat_error": [2.0, [0.1, 2.0]],
    },
    adaptive_termination_ratio=None,
    deviation_thresholds={
        "base_pos_error": 0.5,
        "base_quat_error": 1.0,
        "base_lin_vel_error": 2.0,
    },
    observed_steps={
        "base_pos": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "base_quat": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "base_lin_vel": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "base_ang_vel": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "base_ang_vel_local": [
            1,
        ],
        "dof_pos": [
            1,
        ],
        "dof_vel": [
            1,
        ],
        "link_pos_local": [
            1,
        ],
        "link_quat_local": [
            1,
        ],
        "link_lin_vel": [
            1,
        ],
        "link_ang_vel": [
            1,
        ],
        "foot_contact": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32],
    },
    motion_file="assets/motion/evaluate.pkl",
    relative_motion_obs=False,
    motion_obs_history_len=1,
)


EnvArgsRegistry["g1_motion"] = MotionEnvArgs(
    env_name="MotionEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_g1_mocap"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Motion Tracking ###
        "DofPosReward": 1.0,
        "DofVelReward": 0.01,
        "BaseHeightReward": 100.0,
        "BaseLinVelReward": 20.0,
        "BaseQuatReward": 20.0,
        "BaseAngVelReward": 0.2,
        "TrackingLinkPosGlobalReward": 20.0,
        "TrackingLinkPosLocalReward": 20.0,
        "TrackingLinkQuatReward": 1.0,
        "TrackingLinkLinVelReward": 1.0,
        "FootContactReward": 2.0,
        "FootContactPenalty": 5.0,
        ### Regularization ###
        "TorquePenalty": 0.0001,
        "DofVelPenalty": 0.02,
        "ActionRatePenalty": 0.2,
        "AnkleTorquePenalty": 0.002,
        "BodyAngVelXYPenalty": 1.0,
        "BodyRollPenalty": 20.0,
        "BaseRollPenalty": 20.0,
        "WaistVelPenalty": 0.2,
        "WaistRollPenalty": 10.0,
        # "HipRollPenalty": 5.0,
        "HipYawPenalty": 5.0,
        "G1FeetSlidePenalty": 5.0,
        "MotionFeetAirTimePenalty": 200.0,
        "MotionStandStillFeetContactPenalty": 50.0,
        # "MotionStandStillAnkleVelPenalty": 10.0,
        "FootOrientationPenalty": 100.0,
        "FeetContactForceLimitPenalty": 5.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "diff_dof_vel": 0.1,
    },
    obs_noises={
        "dof_pos": 0.01,
        "dof_vel": 0.2,
        "projected_gravity": 0.05,
        "base_ang_vel_local": 0.2,
    },
    actor_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_ang_vel_local",
        "diff_base_yaw",
        "diff_base_pos_local_yaw",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "projected_gravity",
        # Reference
        "motion_obs_history",
    ],
    critic_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "diff_base_yaw",
        "diff_base_pos_local_yaw",
        "projected_gravity",
        # Motion Difference
        "diff_dof_pos",
        "diff_dof_vel",
        "diff_base_rotation_6D",
        "diff_base_euler",
        "diff_base_ang_vel_local",
        "diff_base_lin_vel_local",
        "diff_tracking_link_pos_global_local_yaw",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "diff_tracking_link_lin_vel_local_yaw",
        "diff_tracking_link_ang_vel_local_yaw",
        # Reference
        "motion_obs_history",
        # Privileged
        "dr_obs",
        "base_lin_vel_local",
        "tracking_link_pos_local_yaw",
        "foot_contact_weighted",
        "ref_foot_contact_weighted",
    ],
    reset_yaw_range=(-0.15, 0.15),
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    tracking_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "torso_link",
        "pelvis",
    ],
    dof_weights={
        "hip_roll": 1.0,
        "hip_pitch": 0.6,
        "hip_yaw": 1.0,
        "knee": 0.6,
        "ankle": 0.3,
        "waist": 2.0,
        "shoulder": 1.0,
        "elbow": 0.6,
        "wrist": 0.3,
    },
    link_pos_global_weights={
        "ankle": 0.5,
        "wrist": 0.0,
        "torso": 1.0,
        "pelvis": 1.0,
    },
    link_pos_local_weights={
        "ankle": 1.0,
        "wrist": 2.0,
        "torso": 0.0,
        "pelvis": 0.0,
    },
    link_quat_weights={
        "ankle": 0.0,
        "wrist": 2.0,
        "torso": 5.0,
    },
    no_terminate_before_motion_time=1.0,
    no_terminate_after_reset_time=2.0,
    no_terminate_after_random_push_time=2.0,
    # [initial_threshold, [min_threshold, max_threshold]]
    terminate_after_error={
        "base_pos_error": [2.0, [0.1, 1.0]],
        "base_quat_error": [2.0, [0.1, 1.0]],
    },
    adaptive_termination_ratio=None,
    deviation_thresholds={
        "base_pos_error": 0.5,
        "base_quat_error": 1.0,
        "base_lin_vel_error": 2.0,
    },
    observed_steps={
        "base_pos": [
            1,
        ],
        "base_quat": [
            1,
        ],
        "base_lin_vel": [
            1,
        ],
        "base_ang_vel": [
            1,
        ],
        "link_pos_local": [
            1,
        ],
        "link_quat_local": [
            1,
        ],
        "foot_contact": [
            1,
        ],
    },
    motion_file="assets/motion/evaluate.pkl",
    relative_motion_obs=False,
    motion_obs_history_len=1,
)
