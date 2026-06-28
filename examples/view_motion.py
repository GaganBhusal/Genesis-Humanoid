from __future__ import annotations

from typing import Any

import fire
import gs_env.sim.envs as gs_envs
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry


def view_motion(
    env_args: Any,
    show_viewer: bool = True,
    save_video: bool = False,
    video_path: str = "motion_playback.mp4",
    save_gif: bool = False,
    gif_path: str = "motion_playback.gif",
) -> None:
    """View reference motion playback (no policy)."""
    # Create environment for evaluation
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    import time

    link_name_to_idx: dict[str, int] = {}
    for link_name in env.scene.objects.keys():
        if link_name in env.motion_lib.tracking_link_names:
            link_name_to_idx[link_name] = env.motion_lib.tracking_link_names.index(link_name)
        elif link_name == "left_wrist_yaw_link" and "left_wrist_roll_rubber_hand" in env.motion_lib.tracking_link_names:
            link_name_to_idx[link_name] = env.motion_lib.tracking_link_names.index("left_wrist_roll_rubber_hand")
        elif link_name == "right_wrist_yaw_link" and "right_wrist_roll_rubber_hand" in env.motion_lib.tracking_link_names:
            link_name_to_idx[link_name] = env.motion_lib.tracking_link_names.index("right_wrist_roll_rubber_hand")

    def run() -> None:
        nonlocal env
        motion_id = 0
        env_idx = torch.tensor([0], device=env.device, dtype=torch.long)

        if save_video or save_gif:
            env.start_rendering(
                save_gif=save_gif,
                gif_path=gif_path,
                save_video=save_video,
                video_path=video_path,
            )

        while True:
            env.time_since_reset[0] = 0.0
            motion_id_tensor = torch.tensor([motion_id], device=env.device, dtype=torch.long)
            env.hard_reset_motion(env_idx, motion_id)
            env.hard_sync_motion(env_idx)
            last_update_time = time.time()
            while env.motion_times[0] + 0.02 < env.motion_lib.get_motion_length(motion_id_tensor):
                env.scene.scene.step()
                env.time_since_reset[0] += 0.02
                env.hard_sync_motion(env_idx)
                env.update_buffers()
                for link_name in env.scene.objects.keys():
                    link_pos = env.ref_tracking_link_pos_global[:, link_name_to_idx[link_name]]
                    link_quat = env.ref_tracking_link_quat_global[:, link_name_to_idx[link_name]]
                    env.scene.set_obj_pose(link_name, pos=link_pos, quat=link_quat)
                env.scene.scene.clear_debug_objects()
                for i in range(len(env.robot.foot_links_idx)):
                    env.scene.scene.draw_debug_arrow(
                        env.link_positions[0, env.robot.foot_links_idx[i]].cpu(),
                        (
                            env.ref_foot_contact_weighted[0, i]
                            * torch.tensor([0.0, 0.0, 0.5], device=env.device)
                        ).cpu(),
                        radius=0.01,
                        color=(0.0, 0.0, 1.0),
                    )

                if save_video or save_gif:
                    env._render_headless()

                if show_viewer:
                    while time.time() - last_update_time < 0.02:
                        time.sleep(0.01)
                    last_update_time = time.time()

            if not show_viewer:
                break

            env.time_since_reset[0] = 0.0
            while True:
                # action = input(
                #     "Enter n to play next motion, q to quit, r to replay current motion, p to play previous motion, id to play specific motion\n"
                # )
                action = 'r'
                if action == "n":
                    motion_id = (motion_id + 1) % env.motion_lib.num_motions
                    break
                elif action == "q":
                    return
                elif action == "r":
                    break
                elif action == "p":
                    motion_id = (motion_id - 1) % env.motion_lib.num_motions
                    break
                elif action.isdigit():
                    motion_id = int(action)
                    break
                else:
                    print("Invalid action")
                    return

        if save_video or save_gif:
            env.stop_rendering()

    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if "Viewer closed" in str(e):
            print("Viewer closed successfully.")
        else:
            raise e


def main(
    motion_file: str = "assets/motion/evaluate.pkl",
    env_args_name: str = "g1_motion",
    show_viewer: bool = True,
    save_video: bool = False,
    video_path: str = "motion_playback.mp4",
    save_gif: bool = False,
    gif_path: str = "motion_playback.gif",
) -> None:
    env_args = EnvArgsRegistry[env_args_name]
    env_args = env_args.model_copy(update={"motion_file": motion_file})
    view_motion(
        env_args,
        show_viewer=show_viewer,
        save_video=save_video,
        video_path=video_path,
        save_gif=save_gif,
        gif_path=gif_path,
    )


if __name__ == "__main__":
    fire.Fire(main)
