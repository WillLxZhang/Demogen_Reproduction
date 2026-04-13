import os

import numpy as np
import zarr
from termcolor import cprint
from tqdm import tqdm

from demo_generation.demogen_lift_phase_copy import LiftPhaseCopyDemoGen


class LiftPhaseCopyExecConsistentDemoGen(LiftPhaseCopyDemoGen):
    """
    Minimal rollback-friendly fork for Lift.

    Differences from LiftPhaseCopyDemoGen:
    - do not use translation_correction_scale_xyz hacks
    - if motion_exec_pulse_scale_xyz is omitted in config, auto-load the
      recommended pulse scale from source zarr meta attrs
    - update generated state / point cloud using the extra motion that the
      encoded executable action actually delivers, instead of the requested
      translation increment
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.phase_copy_state_update_semantics = "delivered_exec_extra"
        self.translation_correction_scale_xyz = np.ones(3, dtype=np.float32)

        if self.motion_exec_mode == "osc_pose_pulse" and self.motion_exec_pulse_scale_xyz is None:
            source_zarr = os.path.join(
                self.data_root,
                "datasets",
                "source",
                f"{self.source_name}.zarr",
            )
            zarr_root = zarr.open(source_zarr, mode="r")
            recommended = zarr_root["meta"].attrs.get("exec_motion_xyz_scale_recommended", None)
            if recommended is None:
                raise ValueError(
                    "motion_exec_pulse_scale_xyz is unset and source zarr meta "
                    "does not contain exec_motion_xyz_scale_recommended."
                )
            self.motion_exec_pulse_scale_xyz = np.asarray(recommended, dtype=np.float32)
            if self.motion_exec_pulse_scale_xyz.shape != (3,):
                raise ValueError(
                    "exec_motion_xyz_scale_recommended in source zarr meta must have shape (3,)"
                )

        cprint(
            (
                "Lift exec-consistent fork: "
                "translation_correction_scale_xyz is disabled, "
                "and stored state follows delivered extra exec motion."
            ),
            "cyan",
        )

    def _estimate_delivered_extra_step(self, source_exec_action, final_exec_action):
        delta_exec_xyz = (
            np.asarray(final_exec_action[:3], dtype=np.float32)
            - np.asarray(source_exec_action[:3], dtype=np.float32)
        )
        if self.motion_exec_mode == "direct":
            return delta_exec_xyz.astype(np.float32)
        if self.motion_exec_mode == "osc_pose_delta":
            if self.motion_exec_output_scale_xyz is None:
                raise ValueError(
                    "motion_exec_output_scale_xyz must be set when motion_exec_mode=osc_pose_delta"
                )
            return (delta_exec_xyz * self.motion_exec_output_scale_xyz).astype(np.float32)
        if self.motion_exec_mode == "osc_pose_pulse":
            if self.motion_exec_pulse_scale_xyz is None:
                raise ValueError(
                    "motion_exec_pulse_scale_xyz must be set when motion_exec_mode=osc_pose_pulse"
                )
            return (delta_exec_xyz * self.motion_exec_pulse_scale_xyz).astype(np.float32)
        raise ValueError(f"Unsupported motion_exec_mode: {self.motion_exec_mode}")

    def one_stage_augment(self, n_demos, render_video=False, gen_mode="random"):
        if gen_mode == "random":
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="random")
        elif gen_mode == "grid":
            def check_squared_number(arr):
                return np.isclose(np.sqrt(arr), np.round(np.sqrt(arr)))

            assert check_squared_number(n_demos), "n_demos must be a squared number"
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="grid")
        else:
            raise NotImplementedError

        generated_episodes = []

        cprint(
            (
                "Lift exec-consistent schedule: "
                f"compose_mode={self.translation_exec_compose_mode}, "
                f"mode={self.translation_schedule_mode}, "
                f"smooth_window={self.translation_schedule_smooth_window}, "
                f"floor_ratio={self.translation_schedule_floor_ratio}, "
                f"active_ratio={self.translation_schedule_active_ratio}, "
                f"hold_final_frame={self.translation_schedule_hold_final_frame}, "
                f"pre_descend_margin={self.translation_schedule_pre_descend_margin_frames}, "
                f"pulse_scale={np.round(self.motion_exec_pulse_scale_xyz, 6).tolist()}"
            ),
            "cyan",
        )

        for i in tqdm(range(self.n_source_episodes)):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]

            if self.use_manual_parsing_frames:
                skill_1_frame = self.resolve_parsing_frame("skill-1", i)
            else:
                ee_poses = source_demo["state"][:, :3]
                skill_1_frame = self.parse_frames_one_stage(pcds, i, ee_poses)
            print(f"Skill-1: {skill_1_frame}")
            if skill_1_frame <= 0:
                raise ValueError(
                    f"Parsed skill_1_frame={skill_1_frame} for source demo {i}. "
                    "This makes the motion segment empty, so augmentation collapses "
                    "to a copy of the source trajectory. Use manual parsing frames "
                    "or tighten the automatic parser."
                )

            pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
            obj_bbox = self.pcd_bbox(pcd_obj)

            for obj_trans_vec in tqdm(trans_vectors):
                current_frame = 0
                traj_states = []
                traj_actions = []
                traj_pcds = []

                translation_increments = self._build_translation_increments(
                    source_demo=source_demo,
                    skill_1_frame=skill_1_frame,
                    object_translation=obj_trans_vec,
                )
                trans_sofar = np.zeros(3, dtype=np.float32)
                motion_exec_state = self._init_motion_exec_state()

                for _ in range(skill_1_frame):
                    source_exec_action = np.asarray(source_demo["action"][current_frame], dtype=np.float32)
                    source_motion_action = np.asarray(
                        self._get_motion_action(source_demo, current_frame), dtype=np.float32
                    )
                    extra_step = np.asarray(translation_increments[current_frame], dtype=np.float32)
                    step_action = np.asarray(source_motion_action[:3] + extra_step, dtype=np.float32)

                    state = np.asarray(source_demo["state"][current_frame], dtype=np.float32).copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)

                    source_pcd = np.asarray(source_demo["point_cloud"][current_frame], dtype=np.float32).copy()
                    pcd_obj, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox])
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj], axis=0))

                    action = self._build_phase_copy_motion_exec_action(
                        step_action,
                        source_exec_action,
                        motion_exec_state,
                        source_motion_action=source_motion_action,
                    )
                    traj_actions.append(action)

                    delivered_extra_step = self._estimate_delivered_extra_step(
                        source_exec_action=source_exec_action,
                        final_exec_action=action,
                    )
                    trans_sofar = trans_sofar + delivered_extra_step
                    current_frame += 1

                num_frames = source_demo["state"].shape[0]
                while current_frame < num_frames:
                    action = np.asarray(source_demo["action"][current_frame], dtype=np.float32).copy()
                    traj_actions.append(action)

                    state = np.asarray(source_demo["state"][current_frame], dtype=np.float32).copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)

                    pcd_obj_robot = np.asarray(source_demo["point_cloud"][current_frame], dtype=np.float32).copy()
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(pcd_obj_robot)

                    current_frame += 1

                generated_episode = {
                    "state": traj_states,
                    "action": traj_actions,
                    "point_cloud": traj_pcds,
                    "_source_episode_idx": int(i),
                    "_object_translation": np.asarray(obj_trans_vec, dtype=np.float32),
                    "_motion_frame_count": int(skill_1_frame),
                    "_realized_object_translation_estimate": np.asarray(trans_sofar, dtype=np.float32),
                }
                generated_episodes.append(generated_episode)

                if render_video:
                    video_name = (
                        f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}].mp4"
                    )
                    video_path = (
                        f"{self.data_root}/videos/{self.generated_name}/{self.gen_name}/{video_name}"
                    )
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)

        save_path = (
            f"{self.data_root}/datasets/generated/{self.generated_name}_{self.gen_name}_{n_demos}.zarr"
        )
        self.save_episodes(generated_episodes, save_path)
