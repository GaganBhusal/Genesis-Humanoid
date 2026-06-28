import sys
import time
from pathlib import Path
import yaml
import fire
import torch

from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_from_angle_axis,
    quat_mul,
    quat_to_euler,
    quat_to_rotation_6D,
)
from gs_env.common.utils.motion_utils import MotionLib, build_motion_obs_from_dict, batched_global_to_local
from gs_env.sim.envs.config.schema import MotionEnvArgs

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from examples.utils import yaml_to_config  # type: ignore


def load_checkpoint_and_env_args(
    exp_name: str, num_ckpt: int | None = None, device: str = "cpu"
) -> tuple[Path, MotionEnvArgs]:
    """Resolve and load env_args and checkpoint path from deploy/logs or logs directory.

    Args:
        exp_name: Experiment name
        num_ckpt: Checkpoint number. If None, loads the latest checkpoint.

    Returns:
        Tuple of (ckpt_path, env_args)
    """
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    env_args_path = None
    ckpt_path = None

    # 1. Try JIT-traced folder
    if deploy_dir.exists():
        env_args_path = deploy_dir / "env_args.yaml"
        if num_ckpt is not None:
            ckpt_path = deploy_dir / f"checkpoint_{num_ckpt:04d}.pt"
            if not ckpt_path.exists():
                ckpt_path = None
        else:
            ckpts = list(deploy_dir.glob("checkpoint_*.pt"))
            if ckpts:
                ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))

    # 2. Try raw training logs folder
    if ckpt_path is None:
        import glob
        import os
        log_pattern = f"logs/{exp_name}/*"
        log_dirs = glob.glob(log_pattern)
        if log_dirs:
            log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            exp_dir = Path(log_dirs[0])
            env_args_path = exp_dir / "configs" / "env_args.yaml"
            if num_ckpt is not None:
                ckpt_path = exp_dir / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
            if ckpt_path is None or not ckpt_path.exists():
                ckpts = list((exp_dir / "checkpoints").glob("checkpoint_*.pt"))
                if ckpts:
                    ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))

    if ckpt_path is None or env_args_path is None or not env_args_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint or env_args for experiment: {exp_name}")

    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)

    return ckpt_path, env_args


def main(
    exp_name: str = "kick_01_student_resume_resume",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = True,
    sim: bool = True,
    action_scale: float = 0.0,  # only for real robot
    motion_file: str = "./assets/motion/optitrack/kick_01.pkl",
) -> None:
    """Run looping motion playback policy on either simulation or real robot for 23 DOF G1.

    Args:
        exp_name: Experiment name
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
    """
    device = "cpu" if not torch.cuda.is_available() else device
    device_t = torch.device(device)

    # Resolve and load checkpoint path + env args
    ckpt_path, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt, device)
    env_args = env_args.model_copy(update={"motion_file": motion_file})

    wrapped_env = None
    if sim:
        print("Running in SIMULATION mode")
        import gs_env.sim.envs as envs

        envclass = getattr(envs, env_args.env_name)
        env = envclass(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=device_t,
            eval_mode=True,
        )
        env.eval()
        env.reset()
        from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
        wrapped_env = GenesisEnvWrapper(env, device=device_t)

        # Define custom get_terminated to ignore motion_end timeout and allow endless looping
        original_get_terminated = env.get_terminated
        def custom_get_terminated():
            original_time = env.time_since_reset.clone()
            env.time_since_reset[0] = 0.0
            term = original_get_terminated()
            env.time_since_reset.copy_(original_time)
            return term
        env.get_terminated = custom_get_terminated

    else:
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args,
            action_scale=action_scale,
            interactive=True,
            device=device_t,
            xml_path="assets/robot/unitree_g1/g1_mocap_23dof.xml",
        )

        print("Press Start button to start the policy")
        while not env.robot.Start:
            time.sleep(0.1)

    print("=" * 80)
    print("Starting 23-DOF looping policy execution")
    print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
    print(f"Device: {device}")
    print(f"Loading checkpoint from: {ckpt_path}")
    print("=" * 80)

    # Load policy (either JIT or raw algorithm weights)
    try:
        policy = torch.jit.load(str(ckpt_path))
        policy.to(device_t)
        policy.eval()
        print("Successfully loaded JIT-traced policy.")
    except Exception:
        print("Failed to load as JIT policy. Attempting to load as raw algorithm checkpoint...")
        exp_dir = ckpt_path.parent.parent
        algo_cfg_path = exp_dir / "configs" / "algo_cfg.yaml"
        if not algo_cfg_path.exists():
            raise FileNotFoundError(f"algo_cfg.yaml not found at: {algo_cfg_path}")
        with open(algo_cfg_path, "r") as f:
            algo_data = yaml.safe_load(f)
        algo_type = algo_data.get("algorithm_type", "PPO")
        print(f"Detected algorithm type: {algo_type}")

        if wrapped_env is None:
            from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
            wrapped_env = GenesisEnvWrapper(env, device=device_t)

        if algo_type == "BC":
            from gs_agent.algos.config.schema import BCArgs
            from gs_agent.algos.bc import BC
            algo_cfg = yaml_to_config(algo_cfg_path, BCArgs)
            algo = BC(env=wrapped_env, cfg=algo_cfg, device=device_t)
        else:
            from gs_agent.algos.config.schema import PPOArgs
            from gs_agent.algos.ppo import PPO
            algo_cfg = yaml_to_config(algo_cfg_path, PPOArgs)
            algo = PPO(env=wrapped_env, cfg=algo_cfg, device=device_t)

        algo.load(ckpt_path, load_optimizer=False)
        raw_policy = algo.get_inference_policy()

        # Deterministic wrapper for raw Gaussian/BC policy
        class CallablePolicy(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, obs):
                if hasattr(self.policy, "dist_from_obs"):
                    action = self.policy(obs, deterministic=True)
                else:
                    action = self.policy(obs)
                if isinstance(action, tuple):
                    action = action[0]
                return action

        policy = CallablePolicy(raw_policy)
        print("Successfully loaded raw algorithm policy wrapper.")

    def deploy_loop() -> None:
        nonlocal env, motion_file, wrapped_env

        # Initialize tracking variables
        last_action_t = torch.zeros(1, env.action_dim, device=device_t)
        total_inference_time = 0
        step_id = 0
        action_scale = 0

        # Initialize motion library (direct file playback)
        motion_lib = MotionLib(motion_file=motion_file, device=device_t)
        motion_id_t = torch.tensor([0], dtype=torch.long, device=device_t)
        t_val = 0.0

        # Initialize motion observation parameters
        motion_obs_steps = motion_lib.get_observed_steps(env_args.observed_steps)
        tracking_link_names = env_args.tracking_link_names
        link_names = motion_lib.tracking_link_names
        tracking_link_idx_local = (
            [link_names.index(name) for name in tracking_link_names] if tracking_link_names else []
        )
        envs_idx = torch.tensor([0], dtype=torch.long, device=device_t)

        if sim:
            env.hard_reset_motion(envs_idx, 0)
            wrapped_env.get_observations()

        obs_history = None

        next_step_time = time.time() + 0.02
        start_step_time = time.time()
        while True:
            # Check termination condition (only for real robot)
            if not sim and hasattr(env, "is_emergency_stop") and env.is_emergency_stop:
                print("Emergency stop triggered!")
                break

            if step_id < 50:
                action_scale += 0.02
                action_scale = min(action_scale, 1.0)

            # Advance motion time and compute reference frame (looping)
            t_val += 0.02
            if t_val > motion_lib.get_motion_length(motion_id_t):
                t_val = 0.0
            motion_time_t = torch.tensor([t_val], dtype=torch.float32, device=device_t)
            (
                ref_base_pos,
                ref_base_quat,
                ref_base_lin_vel,
                ref_base_ang_vel,
                ref_base_ang_vel_local,
                ref_dof_pos,
                ref_dof_vel,
                ref_link_pos_global,
                ref_link_pos_local,
                ref_link_quat_global,
                ref_link_quat_local,
                ref_link_lin_vel,
                ref_link_lin_vel_local,
                ref_link_ang_vel,
                ref_link_ang_vel_local,
                _,
                ref_foot_contact,
                ref_foot_contact_weighted,
            ) = motion_lib.get_ref_motion_frame(motion_ids=motion_id_t, motion_times=motion_time_t)

            _ = ref_base_ang_vel_local
            _ = ref_link_pos_global
            _ = ref_link_pos_local
            _ = ref_link_quat_global
            _ = ref_link_quat_local
            _ = ref_link_lin_vel
            _ = ref_link_lin_vel_local
            _ = ref_link_ang_vel
            _ = ref_link_ang_vel_local
            _ = ref_foot_contact
            _ = ref_foot_contact_weighted
            ref_base_euler = quat_to_euler(ref_base_quat)

            if sim:
                obs_t = wrapped_env.obs
            else:
                # Construct observation (matching training observation structure)
                obs_components = []
                for key in env_args.actor_obs_terms:
                    if key == "last_action":
                        obs_gt = last_action_t
                    elif key in ["motion_obs", "motion_obs_history"]:
                        # Build motion observation from motion library
                        if len(motion_obs_steps) > 0:
                            curr_motion_obs_dict, future_motion_obs_dict = (
                                motion_lib.get_motion_future_obs(
                                    motion_id_t, motion_time_t, motion_obs_steps
                                )
                            )
                            base_quat = env.base_quat
                            obs_gt = build_motion_obs_from_dict(
                                curr_motion_obs_dict,
                                future_motion_obs_dict,
                                envs_idx,
                                tracking_link_idx_local=tracking_link_idx_local,
                                base_quat=base_quat,
                            )
                        else:
                            obs_gt = torch.zeros(1, 0, device=device_t)
                    elif key.startswith("ref_"):
                        if key == "ref_base_pos":
                            obs_gt = ref_base_pos
                        elif key == "ref_base_quat":
                            obs_gt = ref_base_quat
                        elif key == "ref_base_lin_vel":
                            obs_gt = ref_base_lin_vel
                        elif key == "ref_base_ang_vel":
                            obs_gt = ref_base_ang_vel
                        elif key == "ref_base_lin_vel_local":
                            obs_gt = ref_base_lin_vel
                        elif key == "ref_base_ang_vel_local":
                            obs_gt = ref_base_ang_vel
                        elif key == "ref_dof_pos":
                            obs_gt = ref_dof_pos
                        elif key == "ref_dof_vel":
                            obs_gt = ref_dof_vel
                        else:
                            # Fallback: try env if it exposes extra ref_* tensors
                            obs_gt = getattr(env, key)
                    elif key == "diff_base_yaw":
                        obs_gt = (ref_base_euler[0, 2] - env.base_euler[0, 2]).reshape(1, -1)
                    elif key == "diff_base_pos_local_yaw":
                        obs_gt = ref_base_lin_vel * 0.0
                    elif key == "diff_tracking_link_pos_local_yaw":
                        ref_tracking_link_pos = ref_link_pos_local[:, tracking_link_idx_local, :]
                        diff_pos = env.tracking_link_pos_local_yaw - ref_tracking_link_pos
                        obs_gt = diff_pos.reshape(1, -1)
                    elif key == "diff_tracking_link_rotation_6D":
                        ref_tracking_link_quat = ref_link_quat_local[:, tracking_link_idx_local, :]
                        diff_quat = quat_diff(
                            ref_tracking_link_quat,
                            env.tracking_link_quat_local_yaw,
                        )
                        obs_gt = quat_to_rotation_6D(diff_quat).reshape(1, -1)
                    else:
                        obs_gt = getattr(env, key)
                    obs_gt = obs_gt * env_args.obs_scales.get(key, 1.0)
                    obs_components.append(obs_gt)
                obs_t = torch.cat(obs_components, dim=-1)
                if obs_history is None:
                    obs_history = torch.zeros_like(obs_t.reshape(-1, 1)).repeat(
                        1, env_args.obs_history_len
                    )
                obs_history = torch.cat([obs_history[:, 1:], obs_t.reshape(-1, 1)], dim=1)
                obs_t = obs_history.clone().reshape(1, -1)

            # Get action from policy
            with torch.no_grad():
                start_time = time.time()
                action_t = policy(obs_t)
                end_time = time.time()
                total_inference_time += end_time - start_time

            env.apply_action(action_t * action_scale)

            if sim:
                env.time_since_reset[0] = t_val
                terminated = env.get_terminated()
                if terminated[0]:
                    wrapped_env.reset_idx(envs_idx)
                    env.hard_reset_motion(envs_idx, 0)
                    wrapped_env.get_observations()
                    t_val = 0.0
                else:
                    env.update_history()
                    wrapped_env.update_obs_history()

                ref_quat_yaw = quat_from_angle_axis(
                    ref_base_euler[0, 2],
                    torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
                )
                
                # Dynamic mapping for 23-DOF wrist links mapping
                link_name_to_idx = {}
                for link_name in env.scene.objects.keys():
                    if link_name in env.args.tracking_link_names:
                        link_name_to_idx[link_name] = env.args.tracking_link_names.index(link_name)
                    elif link_name == "left_wrist_yaw_link" and "left_wrist_roll_rubber_hand" in env.args.tracking_link_names:
                        link_name_to_idx[link_name] = env.args.tracking_link_names.index("left_wrist_roll_rubber_hand")
                    elif link_name == "right_wrist_yaw_link" and "right_wrist_roll_rubber_hand" in env.args.tracking_link_names:
                        link_name_to_idx[link_name] = env.args.tracking_link_names.index("right_wrist_roll_rubber_hand")

                env.scene.scene.clear_debug_objects()
                for link_name in env.scene.objects.keys():
                    if link_name in link_name_to_idx:
                        link_idx = link_name_to_idx[link_name]
                        if link_idx < ref_link_pos_local.shape[1]:
                            ref_link_pos = ref_link_pos_local[:, link_idx, :]
                            ref_link_quat = ref_link_quat_local[:, link_idx, :]
                            ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                            ref_link_pos += ref_base_pos
                            ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                            env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)
                        else:
                            continue
                        if link_name == "left_ankle_roll_link":
                            env.scene.scene.draw_debug_arrow(
                                ref_link_pos.cpu(),
                                (ref_foot_contact[0, 0]
                                * torch.tensor([0.0, 0.0, 0.5], device=env.device)).cpu(),
                                radius=0.01,
                                color=(0.0, 0.0, 1.0),
                            )
                        if link_name == "right_ankle_roll_link":
                            env.scene.scene.draw_debug_arrow(
                                ref_link_pos.cpu(),
                                (ref_foot_contact[0, 1]
                                * torch.tensor([0.0, 0.0, 0.5], device=env.device)).cpu(),
                                radius=0.01,
                                color=(0.0, 0.0, 1.0),
                            )
            last_action_t = action_t.clone()
            step_id += 1

            # Control loop timing (50 Hz)
            if time.time() < next_step_time:
                time.sleep(max(0, next_step_time - time.time()))
                next_step_time = next_step_time + 0.02
            else:
                next_step_time = time.time() + 0.02

            if step_id % 100 == 0 and step_id > 0:
                print(f"Step {step_id}: Average inference time: {total_inference_time / 100:.4f}s")
                print(f"Step {step_id}: FPS: {100 / (time.time() - start_step_time):.2f}")
                total_inference_time = 0
                start_step_time = time.time()

    try:
        deploy_loop()
    except KeyboardInterrupt:
        if not sim:
            env.emergency_stop()
        print("\nKeyboardInterrupt received, stopping...")
    finally:
        if not sim:
            print("Stopping robot handler...")
        else:
            print("Simulation stopped.")


if __name__ == "__main__":
    fire.Fire(main)
