#!/usr/bin/env python3
"""
Interactive simulation evaluation script for custom KickEnv.
Spawns the G1 robot at a random position/heading, smoothly interpolates
from standing to the kick ready posture, and triggers the kick on ENTER keypress.
"""

import glob
import os
import time
import random
import yaml
from pathlib import Path
import sys
import select

import fire
import torch
import numpy as np

import gs_env.sim.envs as gs_envs
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.utils.math_utils import quat_apply, quat_from_angle_axis, quat_mul, quat_to_euler
from gs_env.sim.envs.config.schema import MotionEnvArgs
from utils import yaml_to_config

def is_enter_pressed() -> bool:
    i, o, e = select.select([sys.stdin], [], [], 0.0)
    if i:
        sys.stdin.readline()
        return True
    return False


def check_if_fallen(env) -> bool:
    # 1. Check tilt (roll / pitch > 0.5 rad (~30 degrees))
    tilt_mask = torch.logical_or(
        torch.abs(env.base_euler[:, 0]) > 0.5,
        torch.abs(env.base_euler[:, 1]) > 0.5,
    )
    if tilt_mask[0].item():
        print(f"Fall detected: robot tilted too much (roll={env.base_euler[0, 0]:.2f}, pitch={env.base_euler[0, 1]:.2f})")
        return True

    # 2. Check height (base Z < 0.45m)
    if env.base_pos[0, 2].item() < 0.45:
        print(f"Fall detected: robot height too low ({env.base_pos[0, 2]:.2f}m)")
        return True
    
    # 3. Check termination contact forces on specified links
    contact_force_mask = torch.any(
        torch.norm(env.link_contact_forces[:, env._terminate_link_idx_local, :], dim=-1) > 1.0,
        dim=-1,
    )
    if contact_force_mask[0].item():
        print("Fall detected: contact force limit exceeded on termination links")
        return True
    
    # 4. Check floor collisions
    floor_collision_mask = env._check_floor_collision(
        env._get_ref_terminate_after_floor_collision_link_idx_global()
    )
    if floor_collision_mask[0].item():
        print("Fall detected: unauthorized floor collision detected")
        return True
        
    return False


def main(
    exp_name: str = "simple_kick_distill_run",
    num_ckpt: int | None = None,
    device: str = "cuda",
    motion_file: str = "assets/motion/optitrack/simple_kick_23dof.pkl",
) -> None:
    # 1. Find latest experiment path and load configs
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    exp_dir = log_dirs[0]
    print("=" * 80)
    print(f"Loading policy from experiment: {exp_dir}")

    env_args = yaml_to_config(Path(exp_dir) / "configs" / "env_args.yaml", MotionEnvArgs)
    
    # Configure arguments for simulation testing
    env_args = env_args.model_copy(update={
        "motion_file": motion_file,
        "show_viewer": True
    })

    # Load checkpoint
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
    print(f"Loading checkpoint file: {ckpt_path}")
    print("=" * 80)

    # 2. Initialize environment
    device_t = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    env_class = getattr(gs_envs, env_args.env_name)
    env = env_class(
        args=env_args,
        num_envs=1,
        show_viewer=True,
        device=device_t,
        eval_mode=True,
    )
    env.eval()

    # 3. Detect algorithm type and initialize policy wrapper
    wrapped_env = GenesisEnvWrapper(env, device=device_t)
    
    algo_cfg_path = Path(exp_dir) / "configs" / "algo_cfg.yaml"
    if not algo_cfg_path.exists():
        raise FileNotFoundError(f"algo_cfg.yaml not found in {exp_dir}/configs")
        
    with open(algo_cfg_path, "r") as f:
        algo_data = yaml.safe_load(f)
    
    algo_type = algo_data.get("algorithm_type", "PPO")
    print(f"Detected algorithm type: {algo_type}")

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
    policy = algo.get_inference_policy()

    # Joint index and scaling definitions
    eval_env_idx = torch.tensor([0], device=device_t, dtype=torch.long)
    standing_dof = env._robot.default_dof_pos.clone()
    action_scale = env.action_scale.clone()

    print("\n" + "=" * 80)
    print("Interactive Kick Evaluation Started!")
    print("- Spawns G1 at a random (X, Y) and randomized heading yaw.")
    print("- Smoothly interpolates from stand -> kick-ready posture.")
    print("- If kick completes successfully, continues from current location.")
    print("- If robot falls/terminates, resets and randomizes location.")
    print("- Prompts user for ENTER to trigger policy execution.")
    print("=" * 80 + "\n")

    needs_reset = True

    try:
        while True:
            # Retrieve starting motion frame reference
            (
                ref_base_pos,
                ref_base_quat,
                ref_base_lin_vel,
                ref_base_ang_vel,
                ref_dof_pos,
                ref_dof_vel,
            ) = env.motion_lib.get_motion_frame(
                torch.tensor([0], device=device_t, dtype=torch.long),
                torch.tensor([0.0], device=device_t, dtype=torch.float32)
            )

            if needs_reset:
                # A. Generate random position in [-2m, 2m] and random yaw heading
                random_x = random.uniform(-2.0, 2.0)
                random_y = random.uniform(-2.0, 2.0)
                random_yaw = random.uniform(0.0, 2.0 * np.pi)

                # Apply offsets to environment base alignment buffers
                env.base_pos_offset[0, 0] = random_x
                env.base_pos_offset[0, 1] = random_y
                env.base_pos_offset[0, 2] = 0.0
                env.base_yaw_offset[0] = random_yaw
                env.base_yaw_offset_quat[0] = quat_from_angle_axis(
                    torch.tensor(random_yaw, device=device_t),
                    torch.tensor([0.0, 0.0, 1.0], device=device_t, dtype=torch.float)
                )

                # B. Reset environment motion time
                env.time_since_reset[0] = 0.0
                env.hard_reset_motion(eval_env_idx, 0)

                # Calculate base position and quat with random spawn offsets
                world_base_pos = quat_apply(env.base_yaw_offset_quat[eval_env_idx], ref_base_pos) + env.base_pos_offset[eval_env_idx]
                world_base_pos[:, 2] += 0.03  # slight height offset to clear ground collision during reset

                # Teleport robot to standing pose at the random location with pure yaw orientation
                standing_base_pos = env._robot.default_pos.clone()
                standing_base_pos[:2] = 0.0  # relative position offset
                world_stand_base_pos = quat_apply(env.base_yaw_offset_quat[eval_env_idx], standing_base_pos.unsqueeze(0)) + env.base_pos_offset[eval_env_idx]
                world_stand_base_pos[:, 2] += 0.03

                env._robot.set_state(
                    pos=world_stand_base_pos,
                    quat=env.base_yaw_offset_quat[eval_env_idx],
                    dof_pos=standing_dof.unsqueeze(0),
                    envs_idx=eval_env_idx,
                    lin_vel=torch.zeros(1, 3, device=device_t),
                    ang_vel=torch.zeros(1, 3, device=device_t),
                    dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                )
                
                env.update_buffers()
                wrapped_env.update_obs_history()

                print(f"\nSpawned at: X={random_x:.2f}m, Y={random_y:.2f}m, Yaw={random_yaw * 180 / np.pi:.1f}°")
            else:
                # We continue from current actual position
                curr_pos = env.base_pos[0].clone()
                curr_quat = env.base_quat[0].clone()
                print(f"\nContinuing at current location: X={curr_pos[0]:.2f}m, Y={curr_pos[1]:.2f}m")

            # Reset environment time for standing
            env.time_since_reset[0] = 0.0
            env._motion_time_offsets[0] = 0.0
            env._update_ref_motion(envs_idx=eval_env_idx)
            env.update_buffers()
            wrapped_env.update_obs_history()

            print("G1 is in stable standing posture.")
            print(">>> Press [ENTER] in terminal to execute the KICK! <<<")

            # Consume any pre-existing ENTER keypresses
            while is_enter_pressed():
                pass

            # Active standing loop while waiting for ENTER
            failed = False
            while True:
                # Kinematically hold the stable standing pose to prevent open-loop gravity drift/falls
                env._robot.set_state(
                    pos=world_stand_base_pos,
                    quat=env.base_yaw_offset_quat[eval_env_idx],
                    dof_pos=standing_dof.unsqueeze(0),
                    envs_idx=eval_env_idx,
                    lin_vel=torch.zeros(1, 3, device=device_t),
                    ang_vel=torch.zeros(1, 3, device=device_t),
                    dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                )

                # Step visualizer to keep window interactive and update viewer frames
                env.scene.scene.step(refresh_visualizer=True, update_visualizer=True)
                
                env.time_since_reset[0] = 0.0
                env._motion_time_offsets[0] = 0.0

                if is_enter_pressed():
                    break

                time.sleep(0.02)

            # User pressed ENTER! Now interpolate from standing_dof to ref_dof_pos[0]
            print("Interpolating standing posture -> kick start frame...")
            curr_dof_pos = env.dof_pos[0].clone()
            curr_pos = env.base_pos[0].clone()
            curr_quat = env.base_quat[0].clone()

            # Align reference base offset to current location before kicking
            euler = quat_to_euler(curr_quat.unsqueeze(0))
            curr_yaw = euler[0, 2]
            env.base_yaw_offset[0] = curr_yaw
            env.base_yaw_offset_quat[0] = quat_from_angle_axis(
                curr_yaw,
                torch.tensor([0.0, 0.0, 1.0], device=device_t, dtype=torch.float)
            )
            ref_base_pos_rotated = quat_apply(env.base_yaw_offset_quat[eval_env_idx], ref_base_pos)
            env.base_pos_offset[0] = curr_pos - ref_base_pos_rotated[0]
            env.base_pos_offset[0, 2] = 0.0

            # Target orientation (includes reference motion base tilt)
            world_base_quat_target = quat_mul(env.base_yaw_offset_quat[eval_env_idx], ref_base_quat)[0]

            N_interp = 40
            for step in range(N_interp):
                alpha = (step + 1) / float(N_interp)
                target_joints = (1.0 - alpha) * curr_dof_pos + alpha * ref_dof_pos[0]
                interp_base_pos = (1.0 - alpha) * curr_pos + alpha * (ref_base_pos_rotated[0] + env.base_pos_offset[eval_env_idx][0])
                interp_base_pos[2] = (1.0 - alpha) * curr_pos[2] + alpha * (ref_base_pos_rotated[0, 2] + 0.03)

                interp_quat = (1.0 - alpha) * curr_quat + alpha * world_base_quat_target
                interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)

                env._robot.set_state(
                    pos=interp_base_pos.unsqueeze(0),
                    quat=interp_quat.unsqueeze(0),
                    dof_pos=target_joints.unsqueeze(0),
                    envs_idx=eval_env_idx,
                    lin_vel=torch.zeros(1, 3, device=device_t),
                    ang_vel=torch.zeros(1, 3, device=device_t),
                    dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                )
                env.scene.scene.step(refresh_visualizer=True, update_visualizer=True)
                time.sleep(0.01)

            # Re-sync buffers/observations right before policy starts
            env.time_since_reset[0] = 0.0
            env._motion_time_offsets[0] = 0.0
            env._update_ref_motion(envs_idx=eval_env_idx)
            env.update_buffers()
            wrapped_env.update_obs_history()
            obs = wrapped_env.obs

            # Execute kick motion using policy
            print("Executing policy...")
            motion_length = env.motion_lib.get_motion_length(torch.tensor([0], device=device_t, dtype=torch.long))

            while env.motion_times[0] < motion_length - 0.02:
                with torch.no_grad():
                    action = policy(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                env.apply_action(action)
                
                if check_if_fallen(env):
                    failed = True
                    break
                    
                env.update_history()
                wrapped_env.update_obs_history()
                obs = wrapped_env.obs
                
                time.sleep(0.02)

            if failed:
                needs_reset = True
                print("Fall detected. Will reset to a new random location.")
                print("Resetting in 1.5 seconds...")
                time.sleep(1.5)
            else:
                # Smoothly interpolate back to standing pose
                print("Kick completed successfully! Interpolating back to stable standing pose...")
                curr_dof_pos = env.dof_pos[0].clone()
                curr_pos = env.base_pos[0].clone()
                curr_quat = env.base_quat[0].clone()

                # Get pure yaw orientation at the final location
                euler = quat_to_euler(curr_quat.unsqueeze(0))
                curr_yaw = euler[0, 2]
                stand_quat_target = quat_from_angle_axis(
                    curr_yaw,
                    torch.tensor([0.0, 0.0, 1.0], device=device_t, dtype=torch.float)
                )

                # Smoothly morph joint angles and orientation back to standing
                N_interp = 40
                for step in range(N_interp):
                    alpha = (step + 1) / float(N_interp)
                    target_joints = (1.0 - alpha) * curr_dof_pos + alpha * standing_dof
                    
                    interp_quat = (1.0 - alpha) * curr_quat + alpha * stand_quat_target[0]
                    interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)
                    
                    env._robot.set_state(
                        pos=curr_pos.unsqueeze(0),
                        quat=interp_quat.unsqueeze(0),
                        dof_pos=target_joints.unsqueeze(0),
                        envs_idx=eval_env_idx,
                        lin_vel=torch.zeros(1, 3, device=device_t),
                        ang_vel=torch.zeros(1, 3, device=device_t),
                        dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                    )
                    env.scene.scene.step(refresh_visualizer=True, update_visualizer=True)
                    time.sleep(0.01)

                needs_reset = False

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")


if __name__ == "__main__":
    fire.Fire(main)
