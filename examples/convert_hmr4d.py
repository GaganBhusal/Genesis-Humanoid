import os
import pickle
import time
from pathlib import Path
from typing import Any

import gs_env.sim.envs as gs_envs
import numpy as np
import smplx
import torch
import yaml
from gs_env.common.utils.motion_utils import (
    GeneralMotionRetargeting,
    load_smplx_data_frames,
)
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from scipy.spatial.transform import Rotation as R
import fire

HUMAN_TO_ROBOT_TRACKING_DICT = {
    "pelvis": "pelvis",
    "spine3": "torso_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}

def retarget_smplx(
    smplx_data: list[dict[str, Any]],
    fps: int,
    actual_human_height: float,
    env: gs_envs.MotionEnv,
    show_viewer: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Retarget SMPLX data to the robot motion data
    """
    # Determine if active robot is 23-DOF
    is_23dof = (len(env.dof_names) == 23)
    tracking_dict = HUMAN_TO_ROBOT_TRACKING_DICT.copy()
    if is_23dof:
        tracking_dict["left_wrist"] = "left_wrist_roll_rubber_hand"
        tracking_dict["right_wrist"] = "right_wrist_roll_rubber_hand"

    robot_xml_file = (
        "assets/robot/unitree_g1/g1_mocap_23dof.xml"
        if is_23dof
        else "assets/robot/unitree_g1/g1_mocap_29dof.xml"
    )
    ik_config_file = (
        "assets/robot/unitree_g1/smplx_to_g1_23dof.json"
        if is_23dof
        else "assets/robot/unitree_g1/smplx_to_g1.json"
    )

    # Initialize the retargeting system
    retargeter = GeneralMotionRetargeting(
        robot_xml_file=robot_xml_file,
        ik_config_file=ik_config_file,
        actual_human_height=actual_human_height,
        aligned_fps=fps,
    )

    raw_tracking_link_names = [robot_name for _, robot_name in tracking_dict.items()]
    raw_tracking_link_pos_global = []
    raw_tracking_link_quat_global = []
    raw_pos_list = []
    raw_quat_list = []
    foot_links_idx = (
        raw_tracking_link_names.index("left_ankle_roll_link"),
        raw_tracking_link_names.index("right_ankle_roll_link"),
    )
    raw_motion_data = {
        "fps": fps,
        "link_names": raw_tracking_link_names,
        "dof_names": env.dof_names,
    }
    base_idx = raw_tracking_link_names.index("pelvis")
    retargeted_tracking_link_names = [link.name for link in env.robot.robot.links]
    retargeted_tracking_link_pos_global = []
    retargeted_tracking_link_quat_global = []
    tracking_link_pos = torch.zeros_like(env.tracking_link_pos_global)[0]
    tracking_link_quat = torch.zeros_like(env.tracking_link_quat_local_yaw)[0]
    pos_list = []
    quat_list = []
    dof_pos_list = []
    foot_contact_list = []
    foot_last_pos = None
    foot_contact = torch.ones(2, dtype=torch.float32)
    retargeted_motion_data = {
        "fps": fps,
        "link_names": retargeted_tracking_link_names,
        "dof_names": env.dof_names,
    }

    frame_counter = 0
    retarget_start_time = time.time()
    speed_measurement_interval = 2.0

    for frame_idx in range(len(smplx_data)):
        if show_viewer:
            # FPS measurements
            frame_counter += 1
            current_time = time.time()
            if current_time - retarget_start_time >= speed_measurement_interval:
                actual_fps = frame_counter / (current_time - retarget_start_time)
                print(f"Actual retargeting FPS: {actual_fps:.2f}")
                frame_counter = 0
                retarget_start_time = current_time

        # Current SMPLX frame
        smplx_frame = smplx_data[frame_idx]

        # Retarget
        scaled_human_data = retargeter.process_human_data(smplx_frame)
        qpos = retargeter.retarget(scaled_human_data)
        qpos_t = torch.tensor(qpos, device=env.device, dtype=torch.float32)

        for j, (human_name, robot_name) in enumerate(tracking_dict.items()):
            if human_name in scaled_human_data.keys():
                pos, quat = scaled_human_data[human_name]
                pos_t = torch.tensor(pos, device=env.device, dtype=torch.float32)
                quat_t = torch.tensor(quat, device=env.device, dtype=torch.float32)
                if "ankle" in robot_name:
                    offset = torch.tensor([-0.1, 0, 0.02], device=env.device, dtype=torch.float32)
                    pos_t += R.from_quat(quat_t, scalar_first=True).apply(offset)
                if "torso" in robot_name:
                    offset = np.array([-0.0039635, 0.0, 0.044], dtype=float)
                    pos_t = torch.tensor(scaled_human_data["pelvis"][0]) + R.from_quat(
                        quat_t, scalar_first=True
                    ).apply(offset)
                tracking_link_pos[j] = pos_t
                tracking_link_quat[j] = quat_t

        foot_pos = tracking_link_pos[foot_links_idx, :]
        if foot_last_pos is not None:
            foot_vel = torch.clamp(
                (torch.norm((foot_pos[..., :2] - foot_last_pos[..., :2]) * fps, dim=-1) - 0.2)
                / 0.2,
                0.0,
                1.0,
            )
            foot_lift = torch.clamp((foot_pos[:, 2] - 0.2) / 0.2, 0.0, 1.0)
            foot_not_contact = (foot_lift + foot_vel).clamp(0.0, 1.0)
            foot_contact = 1 - foot_not_contact
        foot_last_pos = foot_pos.clone()
        foot_contact_list.append(foot_contact.clone())

        raw_tracking_link_pos_global.append(tracking_link_pos.clone())
        raw_tracking_link_quat_global.append(tracking_link_quat.clone())
        raw_pos_list.append(tracking_link_pos[base_idx].clone())
        raw_quat_list.append(tracking_link_quat[base_idx].clone())

        env.robot.set_state(
            pos=qpos_t[:3],
            quat=qpos_t[3:7],
            dof_pos=qpos_t[7:],
        )
        env.update_buffers()

        pos_list.append(qpos_t[:3].clone())
        quat_list.append(qpos_t[3:7].clone())
        dof_pos_list.append(qpos_t[7:].clone())
        retargeted_tracking_link_pos_global.append(env.link_positions[0].clone())
        retargeted_tracking_link_quat_global.append(env.link_quaternions[0].clone())

        if show_viewer:
            env.scene.scene.clear_debug_objects()
            for j, link_name in enumerate(raw_tracking_link_names):
                pos = tracking_link_pos[j]
                quat = tracking_link_quat[j]
                vis_name = link_name
                if link_name == "left_wrist_roll_rubber_hand":
                    vis_name = "left_wrist_yaw_link"
                elif link_name == "right_wrist_roll_rubber_hand":
                    vis_name = "right_wrist_yaw_link"
                env.scene.set_obj_pose(vis_name, pos=pos[None, :], quat=quat[None, :])
            for i in range(len(foot_links_idx)):
                env.scene.scene.draw_debug_arrow(
                    foot_pos[i],
                    foot_contact[i] * torch.tensor([0.0, 0.0, 0.5]),
                    radius=0.01,
                    color=(0.0, 0.0, 1.0),
                )
            env.scene.scene.step()

    raw_motion_data["pos"] = torch.stack(raw_pos_list).numpy()
    raw_motion_data["quat"] = torch.stack(raw_quat_list).numpy()
    raw_motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
    raw_motion_data["link_pos"] = torch.stack(raw_tracking_link_pos_global).numpy()
    raw_motion_data["link_quat"] = torch.stack(raw_tracking_link_quat_global).numpy()
    raw_motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

    retargeted_motion_data["pos"] = torch.stack(pos_list).numpy()
    retargeted_motion_data["quat"] = torch.stack(quat_list).numpy()
    retargeted_motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
    retargeted_motion_data["link_pos"] = torch.stack(retargeted_tracking_link_pos_global).numpy()
    retargeted_motion_data["link_quat"] = torch.stack(retargeted_tracking_link_quat_global).numpy()
    retargeted_motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

    return raw_motion_data, retargeted_motion_data

def load_hmr4d_data(
    pt_file: str, body_models: Any, fps: int = 30
) -> tuple[dict[str, Any], Any, Any, float]:
    """Load SMPL parameters from HMR4D results and pass through SMPLX body model."""
    data = torch.load(pt_file, map_location="cpu")
    global_params = data["smpl_params_global"]

    num_frames = global_params["body_pose"].shape[0]

    # Convert tensors to float
    body_pose = global_params["body_pose"].float()  # (N, 63)
    global_orient = global_params["global_orient"].float()  # (N, 3)
    transl = global_params["transl"].float()  # (N, 3)
    betas_tensor = global_params["betas"].float()  # (N, 10)

    # Use neutral gender body model
    body_model = body_models["neutral"]

    # Pad betas if size is less than num_betas (typically 16)
    num_pad = body_model.num_betas - betas_tensor.shape[-1]
    if num_pad > 0:
        betas_input = torch.cat([betas_tensor, torch.zeros(num_frames, num_pad)], dim=-1)
    else:
        betas_input = betas_tensor

    expression = torch.zeros(num_frames, 10).float()

    smplx_output = body_model(
        betas=betas_input,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=expression,
        return_full_pose=True,
    )

    first_beta = betas_tensor[0, 0].item() if betas_tensor.ndim == 2 else betas_tensor[0].item()
    human_height = 1.66 + 0.1 * first_beta

    smplx_data = {
        "pose_body": body_pose.cpu().numpy(),
        "betas": betas_input[0].cpu().numpy(),
        "root_orient": global_orient.cpu().numpy(),
        "trans": transl.cpu().numpy(),
        "mocap_frame_rate": np.array(fps),
        "gender": np.array("neutral"),
    }

    return smplx_data, body_model, smplx_output, human_height

def hmr4d_to_motion_data(
    env: gs_envs.MotionEnv, body_models: Any, pt_path: Path, fps: int = 30, show_viewer: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    smplx_data, body_model, smplx_output, actual_human_height = load_hmr4d_data(
        str(pt_path), body_models, fps=fps
    )

    # Align FPS to target (e.g. 50 FPS for env simulation steps)
    smplx_data, fps_aligned = load_smplx_data_frames(smplx_data, body_model, smplx_output, tgt_fps=50)

    # Apply 90 degrees rotation about X-axis to convert HMR's Y-up to Genesis's Z-up frame
    r_rot = R.from_euler('x', 90, degrees=True)
    rotated_smplx_data = []
    for frame in smplx_data:
        rotated_frame = {}
        for joint_name, (pos, quat) in frame.items():
            # Rotate 3D Cartesian position
            pos_rot = r_rot.apply(pos)
            # Rotate orientation quaternion
            r_quat = R.from_quat(quat, scalar_first=True)
            r_new = r_rot * r_quat
            quat_rot = r_new.as_quat(scalar_first=True)
            
            rotated_frame[joint_name] = (pos_rot, quat_rot)
        rotated_smplx_data.append(rotated_frame)
    smplx_data = rotated_smplx_data

    raw_motion_data, retargeted_motion_data = retarget_smplx(
        smplx_data, fps_aligned, actual_human_height, env, show_viewer
    )

    return raw_motion_data, retargeted_motion_data

def main(
    pt_file: str,
    output_pkl: str = "assets/motion/optitrack/hmr4d_retargeted.pkl",
    fps: int = 30,
    show_viewer: bool = False,
    env_args_name: str = "g1_motion",
) -> None:
    SMPLX_FOLDER = "assets/body_models"

    print("Loading SMPLX body models...")
    body_models = {
        "neutral": smplx.create(
            SMPLX_FOLDER,
            "smplx",
            gender="neutral",
            use_pca=False,
        ),
    }

    print("Initializing motion environment...")
    env_args = EnvArgsRegistry[env_args_name].model_copy(update={"motion_file": None})
    envclass = getattr(gs_envs, env_args.env_name)
    env = envclass(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    print(f"Retargeting {pt_file} to G1 format (with Y-up to Z-up rotation)...")
    raw_motion_data, retargeted_motion_data = hmr4d_to_motion_data(
        env, body_models, Path(pt_file), fps=fps, show_viewer=show_viewer
    )

    # Save retargeted file
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(retargeted_motion_data, f)
    print(f"Successfully retargeted motion and saved to: {output_path}")

    if show_viewer:
        print("\nPlayback starting... Press Ctrl+C to exit or close the viewer.")
        num_frames = len(retargeted_motion_data["pos"])
        raw_tracking_link_names = raw_motion_data["link_names"]
        
        try:
            while True:
                for f in range(num_frames):
                    pos = torch.tensor(retargeted_motion_data["pos"][f], device=env.device, dtype=torch.float32)
                    quat = torch.tensor(retargeted_motion_data["quat"][f], device=env.device, dtype=torch.float32)
                    dof_pos = torch.tensor(retargeted_motion_data["dof_pos"][f], device=env.device, dtype=torch.float32)
                    env.robot.set_state(
                        pos=pos,
                        quat=quat,
                        dof_pos=dof_pos,
                    )
                    env.scene.scene.clear_debug_objects()
                    for j, link_name in enumerate(raw_tracking_link_names):
                        raw_pos = torch.tensor(raw_motion_data["link_pos"][f, j], device=env.device, dtype=torch.float32)
                        raw_quat = torch.tensor(raw_motion_data["link_quat"][f, j], device=env.device, dtype=torch.float32)
                        vis_name = link_name
                        if link_name == "left_wrist_roll_rubber_hand":
                            vis_name = "left_wrist_yaw_link"
                        elif link_name == "right_wrist_roll_rubber_hand":
                            vis_name = "right_wrist_yaw_link"
                        env.scene.set_obj_pose(vis_name, pos=raw_pos[None, :], quat=raw_quat[None, :])
                    
                    env.scene.scene.step()
                    time.sleep(1.0 / fps)
        except KeyboardInterrupt:
            print("Playback stopped.")

if __name__ == "__main__":
    fire.Fire(main)
