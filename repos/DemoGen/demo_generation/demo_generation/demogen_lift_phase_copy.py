import numpy as np
from termcolor import cprint
from tqdm import tqdm

from demo_generation.demogen import DemoGen


class LiftPhaseCopyDemoGen(DemoGen):
    """
    Copied Lift-focused generator that keeps the original DemoGen class untouched.

    Key differences from the upstream one-stage generator:
    - store generated state / point cloud with pre-action semantics so
      agent_pos[t] lines up with replay observation before action[t]
    - preserve the source motion schedule and only encode extra translation
      increments on top of the source motion interface
    - distribute translation with a configurable phase-aware schedule instead of
      constant per-frame linear interpolation
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.generated_name = str(getattr(cfg, "generated_name", self.source_name))
        self.translation_exec_compose_mode = str(
            getattr(cfg, "translation_exec_compose_mode", "inherit")
        )
        self.translation_schedule_mode = str(
            getattr(cfg, "translation_schedule_mode", "source_motion_axiswise")
        )
        self.translation_schedule_smooth_window = int(
            getattr(cfg, "translation_schedule_smooth_window", 5)
        )
        self.translation_schedule_floor_ratio = float(
            getattr(cfg, "translation_schedule_floor_ratio", 0.1)
        )
        self.translation_schedule_hold_final_frame = bool(
            getattr(cfg, "translation_schedule_hold_final_frame", True)
        )
        self.translation_schedule_active_ratio = float(
            getattr(cfg, "translation_schedule_active_ratio", 1.0)
        )
        self.translation_schedule_pre_descend_margin_frames = int(
            getattr(cfg, "translation_schedule_pre_descend_margin_frames", 0)
        )
        translation_correction_scale_xyz = getattr(
            cfg,
            "translation_correction_scale_xyz",
            [1.0, 1.0, 1.0],
        )
        self.translation_correction_scale_xyz = np.asarray(
            translation_correction_scale_xyz,
            dtype=np.float32,
        )

        valid_modes = {
            "uniform",
            "source_motion_axiswise",
            "state_delta_axiswise",
            "uniform_pre_descend_xy",
            "source_motion_axiswise_pre_descend_xy",
            "state_delta_axiswise_pre_descend_xy",
        }
        if self.translation_schedule_mode not in valid_modes:
            raise ValueError(
                f"Unsupported translation_schedule_mode={self.translation_schedule_mode}. "
                f"Expected one of {sorted(valid_modes)}"
            )
        if self.translation_schedule_smooth_window <= 0:
            raise ValueError("translation_schedule_smooth_window must be positive")
        if self.translation_schedule_floor_ratio < 0:
            raise ValueError("translation_schedule_floor_ratio must be non-negative")
        if not (0.0 < self.translation_schedule_active_ratio <= 1.0):
            raise ValueError("translation_schedule_active_ratio must be in (0, 1]")
        if self.translation_schedule_pre_descend_margin_frames < 0:
            raise ValueError("translation_schedule_pre_descend_margin_frames must be non-negative")
        if self.translation_correction_scale_xyz.shape != (3,):
            raise ValueError(
                "translation_correction_scale_xyz must have shape (3,)"
            )
        valid_compose_modes = {
            "inherit",
            "defer_extra_on_source_conflict",
            "override_source_on_opposite_conflict",
            "replace_source_xy_with_step",
        }
        if self.translation_exec_compose_mode not in valid_compose_modes:
            raise ValueError(
                f"Unsupported translation_exec_compose_mode={self.translation_exec_compose_mode}. "
                f"Expected one of {sorted(valid_compose_modes)}"
            )

    @staticmethod
    def _init_motion_exec_state():
        return {
            "residual_xyz": np.zeros(3, dtype=np.float32),
            "extra_translation_residual_xyz": np.zeros(3, dtype=np.float32),
            "replace_step_residual_xyz": np.zeros(3, dtype=np.float32),
        }

    def _encode_with_residual_key(
        self,
        step_xyz,
        motion_exec_state,
        residual_key: str,
        axis_mask=None,
    ):
        if motion_exec_state is None:
            raise ValueError("motion_exec_state must be provided")
        if self.motion_exec_pulse_scale_xyz is None:
            raise ValueError("motion_exec_pulse_scale_xyz must be set for pulse encoding")

        step_xyz = np.asarray(step_xyz, dtype=np.float32)
        if axis_mask is None:
            axis_mask = np.ones(3, dtype=bool)
        else:
            axis_mask = np.asarray(axis_mask, dtype=bool)

        residual_xyz = motion_exec_state[residual_key]
        residual_xyz += step_xyz
        exec_xyz = np.zeros(3, dtype=np.float32)
        threshold_xyz = self.motion_exec_pulse_scale_xyz * self.motion_exec_pulse_threshold_ratio

        for axis in range(3):
            if not axis_mask[axis]:
                continue

            scale = self.motion_exec_pulse_scale_xyz[axis]
            threshold = threshold_xyz[axis]
            if scale <= 0 or threshold <= 0:
                raise ValueError("motion_exec_pulse_scale_xyz and threshold must be positive")

            if residual_xyz[axis] >= threshold:
                exec_xyz[axis] = min(1.0, np.floor(residual_xyz[axis] / scale))
            elif residual_xyz[axis] <= -threshold:
                exec_xyz[axis] = max(-1.0, -np.floor((-residual_xyz[axis]) / scale))

            residual_xyz[axis] -= exec_xyz[axis] * scale

        motion_exec_state[residual_key] = residual_xyz
        return exec_xyz.astype(np.float32)

    def _build_phase_copy_motion_exec_action(
        self,
        step_action,
        source_exec_action,
        motion_exec_state=None,
        source_motion_action=None,
    ):
        if self.translation_exec_compose_mode not in {
            "inherit",
            "defer_extra_on_source_conflict",
            "override_source_on_opposite_conflict",
            "replace_source_xy_with_step",
        }:
            raise ValueError(
                f"Unsupported translation_exec_compose_mode: {self.translation_exec_compose_mode}"
            )
        if motion_exec_state is None:
            raise ValueError("motion_exec_state must be provided for phase-copy composition")
        if source_motion_action is None:
            raise ValueError("source_motion_action must be provided for phase-copy composition")

        extra_step = (
            np.asarray(step_action, dtype=np.float32)
            - np.asarray(source_motion_action[:3], dtype=np.float32)
        )
        scaled_extra_step = extra_step * self.translation_correction_scale_xyz
        source_exec_xyz = np.asarray(source_exec_action[:3], dtype=np.float32)

        if self.translation_exec_compose_mode == "inherit":
            effective_step_action = (
                np.asarray(source_motion_action[:3], dtype=np.float32) + scaled_extra_step
            )
            return super()._build_motion_exec_action(
                effective_step_action,
                source_exec_action,
                motion_exec_state=motion_exec_state,
                source_motion_action=source_motion_action,
            )

        if self.translation_exec_compose_mode == "replace_source_xy_with_step":
            encoded_step_xy = self._encode_with_residual_key(
                step_xyz=np.asarray(
                    [scaled_extra_step[0] + source_motion_action[0], scaled_extra_step[1] + source_motion_action[1], 0.0],
                    dtype=np.float32,
                ),
                motion_exec_state=motion_exec_state,
                residual_key="replace_step_residual_xyz",
                axis_mask=np.asarray([True, True, False], dtype=bool),
            )
            exec_xyz = source_exec_xyz.copy()
            exec_xyz[:2] = encoded_step_xy[:2]
            return np.concatenate([exec_xyz, source_exec_action[3:]], axis=0)

        exec_xyz = source_exec_xyz.copy()

        residual_xyz = motion_exec_state["extra_translation_residual_xyz"]
        residual_xyz += scaled_extra_step
        threshold_xyz = self.motion_exec_pulse_scale_xyz * self.motion_exec_pulse_threshold_ratio

        for axis in range(3):
            scale = self.motion_exec_pulse_scale_xyz[axis]
            threshold = threshold_xyz[axis]
            if scale <= 0 or threshold <= 0:
                raise ValueError("motion_exec_pulse_scale_xyz and threshold must be positive")

            extra_pulse = 0.0
            if residual_xyz[axis] >= threshold:
                extra_pulse = min(1.0, np.floor(residual_xyz[axis] / scale))
            elif residual_xyz[axis] <= -threshold:
                extra_pulse = max(-1.0, -np.floor((-residual_xyz[axis]) / scale))

            source_pulse = source_exec_xyz[axis]
            if np.abs(source_pulse) > 1e-6:
                if self.translation_exec_compose_mode == "defer_extra_on_source_conflict":
                    continue
                if extra_pulse != 0.0 and np.sign(extra_pulse) != np.sign(source_pulse):
                    exec_xyz[axis] = extra_pulse
                    residual_xyz[axis] -= extra_pulse * scale
                continue

            exec_xyz[axis] = np.clip(exec_xyz[axis] + extra_pulse, -1.0, 1.0)
            residual_xyz[axis] -= extra_pulse * scale

        motion_exec_state["extra_translation_residual_xyz"] = residual_xyz
        return np.concatenate([exec_xyz, source_exec_action[3:]], axis=0)

    @staticmethod
    def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if window <= 1 or signal.shape[0] <= 1:
            return signal.astype(np.float32)

        kernel = np.ones(window, dtype=np.float32) / float(window)
        padded = np.pad(signal, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
        smoothed = np.zeros_like(signal, dtype=np.float32)
        for axis in range(signal.shape[1]):
            smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
        return smoothed.astype(np.float32)

    def _build_axiswise_schedule_signal(self, source_demo, skill_1_frame: int) -> np.ndarray:
        if self.translation_schedule_mode in {"uniform", "uniform_pre_descend_xy"}:
            return np.ones((skill_1_frame, 3), dtype=np.float32)

        if self.translation_schedule_mode in {
            "source_motion_axiswise",
            "source_motion_axiswise_pre_descend_xy",
        }:
            signal = np.abs(
                np.asarray(source_demo[self.motion_action_key][:skill_1_frame, :3], dtype=np.float32)
            )
        elif self.translation_schedule_mode in {
            "state_delta_axiswise",
            "state_delta_axiswise_pre_descend_xy",
        }:
            next_xyz = np.asarray(source_demo["state"][1 : skill_1_frame + 1, :3], dtype=np.float32)
            cur_xyz = np.asarray(source_demo["state"][:skill_1_frame, :3], dtype=np.float32)
            signal = np.abs(next_xyz - cur_xyz)
        else:
            raise ValueError(f"Unsupported translation_schedule_mode: {self.translation_schedule_mode}")

        signal = self._smooth_signal(signal, self.translation_schedule_smooth_window)
        floor = self.translation_schedule_floor_ratio * np.maximum(signal.mean(axis=0, keepdims=True), 1e-8)
        signal = signal + floor
        return signal.astype(np.float32)

    def _infer_pre_descend_xy_end(self, source_demo, skill_1_frame: int) -> int:
        motion_xyz = np.asarray(
            source_demo[self.motion_action_key][:skill_1_frame, :3],
            dtype=np.float32,
        )
        z_active = np.where(np.abs(motion_xyz[:, 2]) > 1e-6)[0]
        if len(z_active) == 0:
            return int(skill_1_frame)
        return int(max(1, z_active[0]))

    def _build_translation_increments(
        self,
        source_demo,
        skill_1_frame: int,
        object_translation: np.ndarray,
    ) -> np.ndarray:
        object_translation = np.asarray(object_translation, dtype=np.float32)
        increments = np.zeros((skill_1_frame, 3), dtype=np.float32)
        if skill_1_frame <= 0:
            return increments

        active_len = max(1, int(np.floor(skill_1_frame * self.translation_schedule_active_ratio)))
        if self.translation_schedule_hold_final_frame and active_len >= skill_1_frame and skill_1_frame > 1:
            active_len = skill_1_frame - 1

        signal = self._build_axiswise_schedule_signal(source_demo, skill_1_frame)[:active_len]
        if self.translation_schedule_mode.endswith("_pre_descend_xy"):
            pre_descend_xy_end = min(
                active_len,
                self._infer_pre_descend_xy_end(source_demo, skill_1_frame),
            )
            pre_descend_xy_end = max(
                1,
                pre_descend_xy_end - self.translation_schedule_pre_descend_margin_frames,
            )
            signal[pre_descend_xy_end:, :2] = 0.0
        for axis in range(3):
            total = float(signal[:, axis].sum())
            if np.isclose(object_translation[axis], 0.0):
                continue
            if total <= 1e-8:
                increments[:active_len, axis] = object_translation[axis] / float(active_len)
            else:
                increments[:active_len, axis] = object_translation[axis] * (signal[:, axis] / total)
        return increments.astype(np.float32)

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
                "Lift phase-copy schedule: "
                f"compose_mode={self.translation_exec_compose_mode}, "
                f"mode={self.translation_schedule_mode}, "
                f"smooth_window={self.translation_schedule_smooth_window}, "
                f"floor_ratio={self.translation_schedule_floor_ratio}, "
                f"active_ratio={self.translation_schedule_active_ratio}, "
                f"hold_final_frame={self.translation_schedule_hold_final_frame}, "
                f"pre_descend_margin={self.translation_schedule_pre_descend_margin_frames}, "
                f"correction_scale={self.translation_correction_scale_xyz.tolist()}"
            ),
            "cyan",
        )

        for i in tqdm(range(self.n_source_episodes)):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]

            if self.use_manual_parsing_frames:
                skill_1_frame = self.parsing_frames["skill-1"]
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

                for j in range(skill_1_frame):
                    source_exec_action = np.asarray(source_demo["action"][current_frame], dtype=np.float32)
                    source_motion_action = np.asarray(
                        self._get_motion_action(source_demo, current_frame), dtype=np.float32
                    )
                    extra_step = translation_increments[j]
                    step_action = np.asarray(source_motion_action[:3] + extra_step, dtype=np.float32)

                    state = np.asarray(source_demo["state"][current_frame], dtype=np.float32).copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)

                    source_pcd = np.asarray(source_demo["point_cloud"][current_frame], dtype=np.float32).copy()
                    pcd_obj, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox])
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj], axis=0))

                    traj_actions.append(
                        self._build_phase_copy_motion_exec_action(
                            step_action,
                            source_exec_action,
                            motion_exec_state,
                            source_motion_action=source_motion_action,
                        )
                    )

                    trans_sofar = trans_sofar + extra_step
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
