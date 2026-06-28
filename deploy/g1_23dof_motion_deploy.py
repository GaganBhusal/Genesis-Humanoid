import sys
import time
import select
from pathlib import Path

import fire
import torch
import numpy as np
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_from_angle_axis,
    quat_mul,
    quat_to_euler,
    quat_to_rotation_6D,
)
from gs_env.common.utils.motion_utils import MotionLib, build_motion_obs_from_dict
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from examples.utils import yaml_to_config  # type: ignore

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

def load_checkpoint_and_env_args(
    exp_name: str, num_ckpt: int | None = None, device: str = "cpu"
) -> tuple[torch.jit.ScriptModule, MotionEnvArgs]:
    """Load JIT checkpoint and env_args from deploy/logs directory."""
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")

    # Load env_args from YAML
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")

    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)

    # Load checkpoint
    if num_ckpt is not None:
        ckpt_path = deploy_dir / f"checkpoint_{num_ckpt}.pt"
        if not ckpt_path.exists():
            ckpt_path = deploy_dir / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Find latest checkpoint
        ckpts = list(deploy_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {deploy_dir}")
        ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))

    print(f"Loading checkpoint from: {ckpt_path}")
    policy = torch.jit.load(str(ckpt_path))
    policy.to(device)
    policy.eval()

    return policy, env_args

def main(
    exp_name: str = "kick_01",
    num_ckpt: int | None = None,
    device: str = "cuda",
    show_viewer: bool = True,
    sim: bool = True,
    action_scale: float = 1.0,  # only for real robot
    motion_file: str = "assets/motion/optitrack/kick_01.pkl",
) -> None:
    """Run 23-DOF JIT-traced kick policy with stable stand-hold and interpolation.

    Args:
        exp_name: Experiment name (subdirectory in deploy/logs)
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
    """
    device = "cpu" if not torch.cuda.is_available() else device
    device_t = torch.device(device)

    # Load JIT policy and env_args
    policy, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt, device)
    env_args = env_args.model_copy(update={"motion_file": motion_file})

    if sim:
        print("Running in SIMULATION mode")
        # Disable domain randomization and observation noise for stable evaluation/viewing
        from gs_env.sim.robots.config.schema import DomainRandomizationArgs
        robot_args = env_args.robot_args.model_copy(
            update={
                "dr_args": DomainRandomizationArgs(
                    kp_range=(1.0, 1.0),
                    kd_range=(1.0, 1.0),
                    motor_strength_range=(1.0, 1.0),
                    motor_offset_range=(0.0, 0.0),
                    friction_range=(1.0, 1.0),
                    mass_range=(0.0, 0.0),
                    com_displacement_range=(0.0, 0.0),
                )
            }
        )
        env_args = env_args.model_copy(
            update={
                "robot_args": robot_args,
                "obs_noises": {},
            }
        )

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

        # Add wrappers for helper features
        wrapped_env = GenesisEnvWrapper(env, device=device_t)
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
    print("Starting 23-DOF JIT policy execution")
    print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
    print(f"Device: {device}")
    print("=" * 80)

    # Setup motion library
    motion_lib = MotionLib(motion_file=motion_file, device=device_t)
    motion_id_t = torch.tensor([0], dtype=torch.long, device=device_t)
    motion_obs_steps = motion_lib.get_observed_steps(env_args.observed_steps)
    tracking_link_names = env_args.tracking_link_names
    link_names = motion_lib.tracking_link_names
    tracking_link_idx_local = (
        [link_names.index(name) for name in tracking_link_names] if tracking_link_names else []
    )
    envs_idx = torch.tensor([0], dtype=torch.long, device=device_t)

    # Stable Standing joint angles definition
    standing_dof = env._robot.default_dof_pos.clone()

    def get_obs(last_action_t, t_val: float = 0.0):
        obs_components = []
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
        ref_base_euler = quat_to_euler(ref_base_quat)

        for key in env_args.actor_obs_terms:
            if key == "last_action":
                obs_gt = last_action_t
            elif key == "motion_obs":
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
                    obs_gt = getattr(env, key)
            elif key == "diff_base_yaw":
                if t_val > 0.0:
                    obs_gt = (ref_base_euler[0, 2] - env.base_euler[0, 2]).reshape(1, -1)
                else:
                    obs_gt = torch.zeros(1, 1, device=device_t)
            elif key == "diff_base_pos_local_yaw":
                obs_gt = torch.zeros(1, 3, device=device_t)
            elif key == "diff_tracking_link_pos_local_yaw":
                if t_val > 0.0:
                    ref_tracking_link_pos = ref_link_pos_local[:, tracking_link_idx_local, :]
                    diff_pos = env.tracking_link_pos_local_yaw - ref_tracking_link_pos
                    obs_gt = diff_pos.reshape(1, -1)
                else:
                    obs_gt = torch.zeros(1, len(tracking_link_idx_local)*3, device=device_t)
            elif key == "diff_tracking_link_rotation_6D":
                if t_val > 0.0:
                    ref_tracking_link_quat = ref_link_quat_local[:, tracking_link_idx_local, :]
                    diff_quat = quat_diff(
                        ref_tracking_link_quat,
                        env.tracking_link_quat_local_yaw,
                    )
                    obs_gt = quat_to_rotation_6D(diff_quat).reshape(1, -1)
                else:
                    obs_gt = torch.zeros(1, len(tracking_link_idx_local)*6, device=device_t)
            else:
                obs_gt = getattr(env, key)
            obs_gt = obs_gt * env_args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        return torch.cat(obs_components, dim=-1)

    try:
        if sim:
            # Initialize offsets to the initial spawn position
            env.base_pos_offset[0] = env.base_pos[0].clone()
            env.base_pos_offset[0, 2] = 0.0
            
            euler = quat_to_euler(env.base_quat[0].unsqueeze(0))
            curr_yaw = euler[0, 2]
            env.base_yaw_offset[0] = curr_yaw
            env.base_yaw_offset_quat[0] = quat_from_angle_axis(
                curr_yaw, torch.tensor([0.0, 0.0, 1.0], device=device_t, dtype=torch.float)
            )

        while True:
            # Check emergency stop
            if not sim and env.is_emergency_stop:
                print("Emergency stop triggered!")
                break

            # ----------------------------------------------------
            # 1. Standing wait loop
            # ----------------------------------------------------
            print("\nG1 is in stable standing posture.")
            if sim:
                print(">>> Press [ENTER] in terminal to execute the KICK! <<<")
                
                # Consume old enters
                while is_enter_pressed():
                    pass

                while True:
                    # Dynamically calculate the world position based on offsets
                    standing_base_pos = env._robot.default_pos.clone()
                    standing_base_pos[:2] = 0.0
                    world_stand_base_pos = quat_apply(env.base_yaw_offset_quat[0].unsqueeze(0), standing_base_pos.unsqueeze(0)) + env.base_pos_offset[0]
                    world_stand_base_pos[:, 2] += 0.03

                    # Kinematically hold the standing position in simulation
                    env._robot.set_state(
                        pos=world_stand_base_pos,
                        quat=env.base_yaw_offset_quat[0].unsqueeze(0),
                        dof_pos=standing_dof.unsqueeze(0),
                        envs_idx=envs_idx,
                        lin_vel=torch.zeros(1, 3, device=device_t),
                        ang_vel=torch.zeros(1, 3, device=device_t),
                        dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                    )
                    env.scene.scene.step(refresh_visualizer=True, update_visualizer=True)
                    
                    if is_enter_pressed():
                        break
                    time.sleep(0.02)
            else:
                print(">>> Press [ENTER] in terminal to start KICK! <<<")
                while not is_enter_pressed():
                    # Send zero action to stay in standard stand
                    env.apply_action(torch.zeros(1, env.action_dim, device=device_t))
                    time.sleep(0.02)

            # ----------------------------------------------------
            # 2. Interpolate from Standing -> Kick Ready Position
            # ----------------------------------------------------
            print("Interpolating standing posture -> kick start frame...")
            N_interp = 40

            # Get first frame reference
            (
                ref_base_pos, ref_base_quat, ref_base_lin_vel, ref_base_ang_vel,
                _, _, ref_dof_pos, ref_dof_vel, *rest
            ) = motion_lib.get_ref_motion_frame(motion_ids=motion_id_t, motion_times=torch.tensor([0.0], device=device_t))

            if sim:
                curr_dof_pos = env.dof_pos[0].clone()
                curr_pos = env.base_pos[0].clone()
                curr_quat = env.base_quat[0].clone()

                # Align base position/yaw offset
                euler = quat_to_euler(curr_quat.unsqueeze(0))
                curr_yaw = euler[0, 2]
                env.base_yaw_offset[0] = curr_yaw
                env.base_yaw_offset_quat[0] = quat_from_angle_axis(
                    curr_yaw, torch.tensor([0.0, 0.0, 1.0], device=device_t, dtype=torch.float)
                )
                ref_base_pos_rotated = quat_apply(env.base_yaw_offset_quat[0], ref_base_pos)
                env.base_pos_offset[0] = curr_pos - ref_base_pos_rotated[0]
                env.base_pos_offset[0, 2] = 0.0

                world_base_quat_target = quat_mul(env.base_yaw_offset_quat[0].unsqueeze(0), ref_base_quat)[0]

                for step in range(N_interp):
                    alpha = (step + 1) / float(N_interp)
                    target_joints = (1.0 - alpha) * curr_dof_pos + alpha * ref_dof_pos[0]
                    interp_base_pos = (1.0 - alpha) * curr_pos + alpha * (ref_base_pos_rotated[0] + env.base_pos_offset[0])
                    interp_base_pos[2] = (1.0 - alpha) * curr_pos[2] + alpha * (ref_base_pos_rotated[0, 2] + 0.03)

                    interp_quat = (1.0 - alpha) * curr_quat + alpha * world_base_quat_target
                    interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)

                    env._robot.set_state(
                        pos=interp_base_pos.unsqueeze(0),
                        quat=interp_quat.unsqueeze(0),
                        dof_pos=target_joints.unsqueeze(0),
                        envs_idx=envs_idx,
                        lin_vel=torch.zeros(1, 3, device=device_t),
                        ang_vel=torch.zeros(1, 3, device=device_t),
                        dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                    )
                    env.scene.scene.step(refresh_visualizer=True, update_visualizer=True)
                    time.sleep(0.01)

                # Reset env time/obs history
                env.time_since_reset[0] = 0.0
                env._motion_time_offsets[0] = 0.0
                env._update_ref_motion(envs_idx=envs_idx)
                env.update_buffers()
                # Clear observation history in the wrapper
                wrapped_env._obs_history[envs_idx] = 0.0
                wrapped_env._updated[envs_idx] = 0.0
                wrapped_env.update_obs_history()
                obs_t = wrapped_env.obs
            else:
                # Real robot joint/action space interpolation
                obs_t = get_obs(torch.zeros(1, env.action_dim, device=device_t))
                with torch.no_grad():
                    action_0 = policy(obs_t)
                    if isinstance(action_0, tuple):
                        action_0 = action_0[0]

                # Interpolate action command from 0 (standing) to action_0 (ready pose)
                for step in range(N_interp):
                    alpha = (step + 1) / float(N_interp)
                    interp_action = alpha * action_0
                    env.apply_action(interp_action)
                    time.sleep(0.02)

                # Initialize history
                obs_history = None

            # ----------------------------------------------------
            # 3. Policy Execution (Perform the kick)
            # ----------------------------------------------------
            print("Executing kick policy...")
            motion_length = motion_lib.get_motion_length(motion_id_t)[0].item()
            t_val = 0.0
            last_action_t = torch.zeros(1, env.action_dim, device=device_t)
            failed = False
            next_step_time = time.time() + 0.02

            while t_val < motion_length - 0.02:
                # Get current observation
                if sim:
                    obs_t = wrapped_env.obs
                else:
                    # Construct observation manually on real robot
                    obs_step = get_obs(last_action_t, t_val=t_val)
                    if obs_history is None:
                        obs_history = torch.zeros_like(obs_step.reshape(-1, 1)).repeat(
                            1, env_args.obs_history_len
                        )
                    obs_history = torch.cat([obs_history[:, 1:], obs_step.reshape(-1, 1)], dim=1)
                    obs_t = obs_history.clone().reshape(1, -1)

                with torch.no_grad():
                    action = policy(obs_t)
                    if isinstance(action, tuple):
                        action = action[0]

                env.apply_action(action)
                last_action_t = action.clone()

                if sim:
                    if check_if_fallen(env):
                        failed = True
                        break
                    env.update_history()
                    wrapped_env.update_obs_history()
                else:
                    if env.is_emergency_stop:
                        failed = True
                        break

                t_val += 0.02
                if time.time() < next_step_time:
                    time.sleep(max(0.0, next_step_time - time.time()))
                    next_step_time = next_step_time + 0.02
                else:
                    next_step_time = time.time() + 0.02

            if failed:
                print("Kick failed (fall or emergency stop). Resetting standing wait loop...")
                time.sleep(1.0)
                if sim:
                    env.reset_idx(envs_idx)
                continue

            # ----------------------------------------------------
            # 4. Interpolate from Kick position -> Stable Standing
            # ----------------------------------------------------
            print("Kick completed successfully! Interpolating back to stable standing posture...")
            if sim:
                curr_dof_pos = env.dof_pos[0].clone()
                curr_pos = env.base_pos[0].clone()
                curr_quat = env.base_quat[0].clone()

                # Standing base target at current position
                standing_base_pos_final = curr_pos.clone()
                standing_base_pos_final[2] = env._robot.default_pos[2] + 0.03

                # Stand orientation (yaw-only matching current heading)
                euler = quat_to_euler(curr_quat.unsqueeze(0))
                curr_yaw = euler[0, 2]
                stand_quat_final = quat_from_angle_axis(
                    curr_yaw, torch.tensor([0.0, 0.0, 1.0], device=device_t, dtype=torch.float)
                )[0]

                for step in range(N_interp):
                    alpha = (step + 1) / float(N_interp)
                    target_joints = (1.0 - alpha) * curr_dof_pos + alpha * standing_dof
                    interp_base_pos = (1.0 - alpha) * curr_pos + alpha * standing_base_pos_final
                    interp_quat = (1.0 - alpha) * curr_quat + alpha * stand_quat_final
                    interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)

                    env._robot.set_state(
                        pos=interp_base_pos.unsqueeze(0),
                        quat=interp_quat.unsqueeze(0),
                        dof_pos=target_joints.unsqueeze(0),
                        envs_idx=envs_idx,
                        lin_vel=torch.zeros(1, 3, device=device_t),
                        ang_vel=torch.zeros(1, 3, device=device_t),
                        dof_vel=torch.zeros(1, env._robot.dof_dim, device=device_t),
                    )
                    env.scene.scene.step(refresh_visualizer=True, update_visualizer=True)
                    time.sleep(0.01)

                # Set position offset to current base location to align standing wait loops
                env.base_pos_offset[0] = env.base_pos[0].clone()
                env.base_pos_offset[0, 2] = 0.0
                env.base_yaw_offset[0] = curr_yaw
                env.base_yaw_offset_quat[0] = stand_quat_final.clone()
            else:
                # Real robot joint/action space recovery interpolation
                curr_action = last_action_t.clone()
                for step in range(N_interp):
                    alpha = (step + 1) / float(N_interp)
                    interp_action = (1.0 - alpha) * curr_action
                    env.apply_action(interp_action)
                    time.sleep(0.02)

    except KeyboardInterrupt:
        if not sim:
            env.emergency_stop()
        print("\nKeyboardInterrupt received, stopping...")
    finally:
        if sim:
            print("Simulation stopped.")
        else:
            print("Stopping robot handler...")

if __name__ == "__main__":
    fire.Fire(main)
