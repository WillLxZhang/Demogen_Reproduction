import json
import os
from pathlib import Path

import h5py
import numpy as np
import zarr
from termcolor import cprint
from tqdm import tqdm

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from demo_generation.demogen_lift_phase_copy_exec_consistent import (
    LiftPhaseCopyExecConsistentDemoGen,
)


TASK_OBJECT_STATE_INDICES = {
    "Lift": {
        "object": np.array([10, 11, 12], dtype=np.int64),
        "target": None,
    },
    "Stack": {
        "object": np.array([10, 11, 12], dtype=np.int64),
        "target": np.array([17, 18, 19], dtype=np.int64),
    },
}


class StackPhaseCopyReplayConsistentDemoGen(LiftPhaseCopyExecConsistentDemoGen):
    """
    Minimal two-phase replay-consistent fork.

    Goals:
    - keep the validated replay_h1 / pulse-scale semantics
    - keep old DemoGen untouched
    - support two translated objects and two motion segments
    - avoid task-specific correction scales
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.phase_copy_state_update_semantics = "robosuite_replay_observation_two_stage"
        self.translation_correction_scale_xyz = np.ones(3, dtype=np.float32)

        source_demo_hdf5 = getattr(cfg, "source_demo_hdf5", None)
        if source_demo_hdf5 is None:
            raise ValueError(
                "StackPhaseCopyReplayConsistentDemoGen requires cfg.source_demo_hdf5"
            )
        source_demo_hdf5 = Path(source_demo_hdf5).expanduser()
        if not source_demo_hdf5.is_absolute():
            source_demo_hdf5 = Path.cwd() / source_demo_hdf5
        self.source_demo_hdf5 = source_demo_hdf5.resolve()
        if not self.source_demo_hdf5.exists():
            raise FileNotFoundError(f"source_demo_hdf5 not found: {self.source_demo_hdf5}")

        self.source_control_steps = int(getattr(cfg, "source_control_steps", 1))
        if self.source_control_steps <= 0:
            raise ValueError("source_control_steps must be positive")

        env_name = self._load_env_name(self.source_demo_hdf5)
        default_indices = TASK_OBJECT_STATE_INDICES.get(env_name, {})

        replay_object_state_indices = getattr(cfg, "replay_object_state_indices", None)
        replay_target_state_indices = getattr(cfg, "replay_target_state_indices", None)

        self.replay_object_state_indices = self._resolve_index_array(
            replay_object_state_indices,
            default_indices.get("object"),
            "replay_object_state_indices",
            env_name,
        )
        self.replay_target_state_indices = self._resolve_optional_index_array(
            replay_target_state_indices,
            default_indices.get("target"),
        )

        self.source_demo_keys = self._list_demo_keys(self.source_demo_hdf5)

        cprint(
            (
                "Stack replay-consistent fork: "
                f"source_demo_hdf5={self.source_demo_hdf5}, "
                f"source_control_steps={self.source_control_steps}, "
                f"object_state_indices={None if self.replay_object_state_indices is None else self.replay_object_state_indices.tolist()}, "
                f"target_state_indices={None if self.replay_target_state_indices is None else self.replay_target_state_indices.tolist()}"
            ),
            "cyan",
        )

    @staticmethod
    def _resolve_index_array(raw, default, key_name: str, env_name: str):
        if raw is not None:
            arr = np.asarray(raw, dtype=np.int64)
        else:
            arr = None if default is None else np.asarray(default, dtype=np.int64)
        if arr is None:
            raise ValueError(
                f"No default {key_name} configured for env={env_name}. Set it in config."
            )
        return arr

    @staticmethod
    def _resolve_optional_index_array(raw, default):
        if raw is None:
            return None if default is None else np.asarray(default, dtype=np.int64)
        if raw in [None, "null"]:
            return None
        return np.asarray(raw, dtype=np.int64)

    @staticmethod
    def _list_demo_keys(source_demo_hdf5: Path):
        with h5py.File(source_demo_hdf5, "r") as f:
            return list(f["data"].keys())

    @staticmethod
    def _load_env_name(source_demo_hdf5: Path) -> str:
        with h5py.File(source_demo_hdf5, "r") as f:
            env_args = json.loads(f["data"].attrs["env_args"])
        return env_args["env_name"].split("_")[0]

    def _load_reset_state(self, source_episode_idx: int):
        with h5py.File(self.source_demo_hdf5, "r") as f:
            if source_episode_idx < 0 or source_episode_idx >= len(self.source_demo_keys):
                raise IndexError(
                    f"source episode index {source_episode_idx} out of range for demos={self.source_demo_keys}"
                )
            ep = self.source_demo_keys[source_episode_idx]
            group = f[f"data/{ep}"]
            state = {
                "states": group["states"][0],
                "model": group.attrs["model_file"],
            }
            if "ep_meta" in group.attrs:
                state["ep_meta"] = group.attrs["ep_meta"]
            return state

    def _build_replay_reset_state(
        self,
        source_episode_idx: int,
        object_translation: np.ndarray,
        target_translation: np.ndarray,
    ):
        reset_state = self._load_reset_state(source_episode_idx)
        reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
        if object_translation is not None and self.replay_object_state_indices is not None:
            reset_state["states"][self.replay_object_state_indices] += np.asarray(
                object_translation[: len(self.replay_object_state_indices)],
                dtype=np.float64,
            )
        if target_translation is not None and self.replay_target_state_indices is not None:
            reset_state["states"][self.replay_target_state_indices] += np.asarray(
                target_translation[: len(self.replay_target_state_indices)],
                dtype=np.float64,
            )
        return reset_state

    def _build_segment_axiswise_schedule_signal(
        self,
        source_demo,
        start_frame: int,
        end_frame: int,
    ) -> np.ndarray:
        seg_len = max(0, int(end_frame) - int(start_frame))
        if seg_len <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        if self.translation_schedule_mode in {"uniform", "uniform_pre_descend_xy"}:
            return np.ones((seg_len, 3), dtype=np.float32)

        if self.translation_schedule_mode in {
            "source_motion_axiswise",
            "source_motion_axiswise_pre_descend_xy",
        }:
            signal = np.abs(
                np.asarray(
                    source_demo[self.motion_action_key][start_frame:end_frame, :3],
                    dtype=np.float32,
                )
            )
        elif self.translation_schedule_mode in {
            "state_delta_axiswise",
            "state_delta_axiswise_pre_descend_xy",
        }:
            cur_xyz = np.asarray(source_demo["state"][start_frame:end_frame, :3], dtype=np.float32)
            next_xyz = np.asarray(
                source_demo["state"][start_frame + 1:min(end_frame + 1, source_demo["state"].shape[0]), :3],
                dtype=np.float32,
            )
            if len(next_xyz) < len(cur_xyz):
                pad = np.repeat(cur_xyz[-1:], len(cur_xyz) - len(next_xyz), axis=0)
                next_xyz = np.concatenate([next_xyz, pad], axis=0)
            signal = np.abs(next_xyz - cur_xyz)
        else:
            raise ValueError(f"Unsupported translation_schedule_mode: {self.translation_schedule_mode}")

        signal = self._smooth_signal(signal, self.translation_schedule_smooth_window)
        floor = self.translation_schedule_floor_ratio * np.maximum(
            signal.mean(axis=0, keepdims=True),
            1e-8,
        )
        signal = signal + floor
        return signal.astype(np.float32)

    @staticmethod
    def _infer_pre_descend_xy_end_segment(motion_xyz: np.ndarray) -> int:
        z_active = np.where(np.abs(motion_xyz[:, 2]) > 1e-6)[0]
        if len(z_active) == 0:
            return int(len(motion_xyz))
        return int(max(1, z_active[0]))

    def _build_segment_translation_increments(
        self,
        source_demo,
        start_frame: int,
        end_frame: int,
        translation: np.ndarray,
    ) -> np.ndarray:
        translation = np.asarray(translation, dtype=np.float32)
        seg_len = max(0, int(end_frame) - int(start_frame))
        increments = np.zeros((seg_len, 3), dtype=np.float32)
        if seg_len <= 0:
            return increments

        active_len = max(1, int(np.floor(seg_len * self.translation_schedule_active_ratio)))
        if self.translation_schedule_hold_final_frame and active_len >= seg_len and seg_len > 1:
            active_len = seg_len - 1

        signal = self._build_segment_axiswise_schedule_signal(
            source_demo=source_demo,
            start_frame=start_frame,
            end_frame=end_frame,
        )[:active_len]

        if self.translation_schedule_mode.endswith("_pre_descend_xy"):
            motion_xyz = np.asarray(
                source_demo[self.motion_action_key][start_frame:end_frame, :3],
                dtype=np.float32,
            )[:active_len]
            pre_descend_xy_end = min(
                active_len,
                self._infer_pre_descend_xy_end_segment(motion_xyz),
            )
            pre_descend_xy_end = max(
                1,
                pre_descend_xy_end - self.translation_schedule_pre_descend_margin_frames,
            )
            signal[pre_descend_xy_end:, :2] = 0.0

        for axis in range(3):
            total = float(signal[:, axis].sum())
            if np.isclose(translation[axis], 0.0):
                continue
            if total <= 1e-8:
                increments[:active_len, axis] = translation[axis] / float(active_len)
            else:
                increments[:active_len, axis] = translation[axis] * (signal[:, axis] / total)

        return increments.astype(np.float32)

    def _render_stage_point_cloud(
        self,
        source_pcd: np.ndarray,
        obj_bbox: np.ndarray,
        tar_bbox: np.ndarray,
        obj_trans_vec: np.ndarray,
        tar_trans_vec: np.ndarray,
        robot_trans_vec: np.ndarray,
        stage: str,
    ) -> np.ndarray:
        if stage == "motion1":
            pcd_obj, pcd_tar, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
            pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
            pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
            pcd_robot = self.pcd_translate(pcd_robot, robot_trans_vec)
            return np.concatenate([pcd_robot, pcd_obj, pcd_tar], axis=0)

        if stage in {"skill1", "motion2"}:
            pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
            pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
            pcd_obj_robot = self.pcd_translate(pcd_obj_robot, robot_trans_vec)
            return np.concatenate([pcd_obj_robot, pcd_tar], axis=0)

        if stage == "skill2":
            return self.pcd_translate(source_pcd, tar_trans_vec)

        raise ValueError(f"Unsupported stage={stage}")

    def save_episodes(self, generated_episodes, save_dir):
        super().save_episodes(generated_episodes, save_dir)
        if not generated_episodes:
            return

        zarr_root = zarr.open(save_dir, mode="a")
        zarr_meta = zarr_root["meta"]
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

        if all("_motion_2_frame" in ep for ep in generated_episodes):
            motion_2_frame = np.asarray(
                [ep["_motion_2_frame"] for ep in generated_episodes],
                dtype=np.int64,
            )
            zarr_meta.create_dataset(
                "motion_2_frame",
                data=motion_2_frame,
                chunks=(min(100, len(motion_2_frame)),),
                dtype="int64",
                overwrite=True,
                compressor=compressor,
            )
        if all("_skill_2_frame" in ep for ep in generated_episodes):
            skill_2_frame = np.asarray(
                [ep["_skill_2_frame"] for ep in generated_episodes],
                dtype=np.int64,
            )
            zarr_meta.create_dataset(
                "skill_2_frame",
                data=skill_2_frame,
                chunks=(min(100, len(skill_2_frame)),),
                dtype="int64",
                overwrite=True,
                compressor=compressor,
            )
        zarr_meta.attrs["task_stage_semantics"] = "two_phase_motion1_skill1_motion2_skill2"

    def two_stage_augment(self, n_demos, render_video=False, gen_mode="random"):
        trans_vectors = []
        if gen_mode == "random":
            for _ in range(n_demos):
                obj_xyz = self.generate_trans_vectors(self.object_trans_range, 1, mode="random")[0]
                targ_xyz = self.generate_trans_vectors(self.target_trans_range, 1, mode="random")[0]
                trans_vectors.append(np.concatenate([obj_xyz, targ_xyz], axis=0))
        elif gen_mode == "grid":
            def check_fourth_power(arr):
                fourth_roots = np.power(arr, 1 / 4)
                return np.isclose(fourth_roots, np.round(fourth_roots))

            assert check_fourth_power(n_demos), "n_demos must be a fourth power"
            sqrt_n_demos = int(np.sqrt(n_demos))
            obj_xyz = self.generate_trans_vectors(self.object_trans_range, sqrt_n_demos, mode="grid")
            targ_xyz = self.generate_trans_vectors(self.target_trans_range, sqrt_n_demos, mode="grid")
            for o_xyz in obj_xyz:
                for t_xyz in targ_xyz:
                    trans_vectors.append(np.concatenate([o_xyz, t_xyz], axis=0))
        else:
            raise NotImplementedError

        generated_episodes = []

        cprint(
            (
                "Stack replay-consistent schedule: "
                f"compose_mode={self.translation_exec_compose_mode}, "
                f"mode={self.translation_schedule_mode}, "
                f"smooth_window={self.translation_schedule_smooth_window}, "
                f"floor_ratio={self.translation_schedule_floor_ratio}, "
                f"active_ratio={self.translation_schedule_active_ratio}, "
                f"hold_final_frame={self.translation_schedule_hold_final_frame}, "
                f"pre_descend_margin={self.translation_schedule_pre_descend_margin_frames}, "
                f"pulse_scale={np.round(self.motion_exec_pulse_scale_xyz, 6).tolist()}, "
                f"control_steps={self.source_control_steps}"
            ),
            "cyan",
        )

        robosuite_wrapper.N_CONTROL_STEPS = self.source_control_steps
        env = Robosuite3DEnv(str(self.source_demo_hdf5), render=False)

        try:
            for i in range(self.n_source_episodes):
                cprint(f"Generating demos for source demo {i}", "blue")
                source_demo = self.replay_buffer.get_episode(i)
                pcds = source_demo["point_cloud"]

                if self.use_manual_parsing_frames:
                    skill_1_frame = int(self.parsing_frames["skill-1"])
                    motion_2_frame = int(self.parsing_frames["motion-2"])
                    skill_2_frame = int(self.parsing_frames["skill-2"])
                else:
                    ee_poses = source_demo["state"][:, :3]
                    skill_1_frame, motion_2_frame, skill_2_frame = self.parse_frames_two_stage(
                        pcds, i, ee_poses
                    )

                if not (0 < skill_1_frame <= motion_2_frame <= skill_2_frame <= len(source_demo["state"])):
                    raise ValueError(
                        "Invalid two-stage parsing frames: "
                        f"skill_1_frame={skill_1_frame}, motion_2_frame={motion_2_frame}, "
                        f"skill_2_frame={skill_2_frame}, episode_len={len(source_demo['state'])}"
                    )

                pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
                pcd_tar = self.get_objects_pcd_from_sam_mask(pcds[0], i, "target")
                obj_bbox = self.pcd_bbox(pcd_obj)
                tar_bbox = self.pcd_bbox(pcd_tar)

                for trans_vec in tqdm(trans_vectors):
                    obj_trans_vec = np.asarray(trans_vec[:3], dtype=np.float32)
                    tar_trans_vec = np.asarray(trans_vec[3:6], dtype=np.float32)

                    motion1_increments = self._build_segment_translation_increments(
                        source_demo=source_demo,
                        start_frame=0,
                        end_frame=skill_1_frame,
                        translation=obj_trans_vec,
                    )
                    motion2_increments = self._build_segment_translation_increments(
                        source_demo=source_demo,
                        start_frame=motion_2_frame,
                        end_frame=skill_2_frame,
                        translation=tar_trans_vec - obj_trans_vec,
                    )

                    current_frame = 0
                    traj_states = []
                    traj_actions = []
                    traj_pcds = []
                    motion_exec_state = self._init_motion_exec_state()
                    reset_state = self._build_replay_reset_state(
                        source_episode_idx=i,
                        object_translation=obj_trans_vec,
                        target_translation=tar_trans_vec,
                    )
                    obs = env.reset_to(reset_state)
                    current_agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).copy()

                    num_frames = source_demo["state"].shape[0]
                    while current_frame < num_frames:
                        if current_frame < skill_1_frame:
                            stage = "motion1"
                        elif current_frame < motion_2_frame:
                            stage = "skill1"
                        elif current_frame < skill_2_frame:
                            stage = "motion2"
                        else:
                            stage = "skill2"

                        traj_states.append(current_agent_pos.copy())

                        source_state = np.asarray(source_demo["state"][current_frame], dtype=np.float32)
                        robot_trans_vec = np.asarray(current_agent_pos[:3], dtype=np.float32) - source_state[:3]
                        source_pcd = np.asarray(source_demo["point_cloud"][current_frame], dtype=np.float32).copy()
                        traj_pcds.append(
                            self._render_stage_point_cloud(
                                source_pcd=source_pcd,
                                obj_bbox=obj_bbox,
                                tar_bbox=tar_bbox,
                                obj_trans_vec=obj_trans_vec,
                                tar_trans_vec=tar_trans_vec,
                                robot_trans_vec=robot_trans_vec,
                                stage=stage,
                            )
                        )

                        source_exec_action = np.asarray(source_demo["action"][current_frame], dtype=np.float32)
                        if stage == "motion1":
                            source_motion_action = np.asarray(
                                self._get_motion_action(source_demo, current_frame), dtype=np.float32
                            )
                            extra_step = np.asarray(motion1_increments[current_frame], dtype=np.float32)
                            step_action = np.asarray(source_motion_action[:3] + extra_step, dtype=np.float32)
                            action = self._build_phase_copy_motion_exec_action(
                                step_action,
                                source_exec_action,
                                motion_exec_state,
                                source_motion_action=source_motion_action,
                            )
                        elif stage == "motion2":
                            source_motion_action = np.asarray(
                                self._get_motion_action(source_demo, current_frame), dtype=np.float32
                            )
                            extra_step = np.asarray(
                                motion2_increments[current_frame - motion_2_frame],
                                dtype=np.float32,
                            )
                            step_action = np.asarray(source_motion_action[:3] + extra_step, dtype=np.float32)
                            action = self._build_phase_copy_motion_exec_action(
                                step_action,
                                source_exec_action,
                                motion_exec_state,
                                source_motion_action=source_motion_action,
                            )
                        else:
                            action = source_exec_action.copy()

                        traj_actions.append(action)
                        obs, _, _, _ = env.step(action)
                        current_agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
                        current_frame += 1

                    generated_episode = {
                        "state": traj_states,
                        "action": traj_actions,
                        "point_cloud": traj_pcds,
                        "_source_episode_idx": int(i),
                        "_object_translation": np.asarray(
                            np.concatenate([obj_trans_vec, tar_trans_vec], axis=0),
                            dtype=np.float32,
                        ),
                        "_motion_frame_count": int(skill_1_frame),
                        "_motion_2_frame": int(motion_2_frame),
                        "_skill_2_frame": int(skill_2_frame),
                    }
                    generated_episodes.append(generated_episode)

                    if render_video:
                        video_name = (
                            f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]"
                            f"_tar[{np.round(tar_trans_vec[0], 3)},{np.round(tar_trans_vec[1], 3)}].mp4"
                        )
                        video_path = (
                            f"{self.data_root}/videos/{self.generated_name}/{self.gen_name}/{video_name}"
                        )
                        self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
        finally:
            env.close()

        save_path = (
            f"{self.data_root}/datasets/generated/{self.generated_name}_{self.gen_name}_{n_demos}.zarr"
        )
        self.save_episodes(generated_episodes, save_path)
