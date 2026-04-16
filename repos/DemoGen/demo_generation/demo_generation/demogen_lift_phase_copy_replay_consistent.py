import json
import os
from pathlib import Path

import h5py
import numpy as np
from termcolor import cprint
from tqdm import tqdm

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv
from diffusion_policies.env.robosuite.dataset_meta import load_env_name_from_dataset

from demo_generation.demogen_lift_phase_copy_exec_consistent import (
    LiftPhaseCopyExecConsistentDemoGen,
)


TASK_OBJECT_STATE_INDICES = {
    "Lift": np.array([10, 11, 12], dtype=np.int64),
}


class LiftPhaseCopyReplayConsistentDemoGen(LiftPhaseCopyExecConsistentDemoGen):
    """
    Replay-in-the-loop Lift generator fork.

    Compared with exec-consistent:
    - motion-phase state is written from actual robosuite replay observations
    - motion-phase robot translation for point cloud follows actual replayed eef pose
    - action synthesis still reuses the existing phase-copy policy logic

    This keeps old generators untouched while making gate evaluation target
    the same semantics that generated zarr stores.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.phase_copy_state_update_semantics = "robosuite_replay_observation"

        source_demo_hdf5 = getattr(cfg, "source_demo_hdf5", None)
        if source_demo_hdf5 is None:
            raise ValueError(
                "LiftPhaseCopyReplayConsistentDemoGen requires cfg.source_demo_hdf5"
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

        replay_object_state_indices = getattr(cfg, "replay_object_state_indices", None)
        if replay_object_state_indices is not None:
            self.replay_object_state_indices = np.asarray(
                replay_object_state_indices, dtype=np.int64
            )
        else:
            env_name = self._load_env_name(self.source_demo_hdf5)
            self.replay_object_state_indices = TASK_OBJECT_STATE_INDICES.get(env_name, None)
            if self.replay_object_state_indices is None:
                raise ValueError(
                    f"No default object state indices configured for env={env_name}. "
                    "Set replay_object_state_indices in config."
                )

        self.source_demo_keys = self._list_demo_keys(self.source_demo_hdf5)

        cprint(
            (
                "Lift replay-consistent fork: "
                f"source_demo_hdf5={self.source_demo_hdf5}, "
                f"source_control_steps={self.source_control_steps}"
            ),
            "cyan",
        )

    @staticmethod
    def _list_demo_keys(source_demo_hdf5: Path):
        with h5py.File(source_demo_hdf5, "r") as f:
            return list(f["data"].keys())

    @staticmethod
    def _load_env_name(source_demo_hdf5: Path) -> str:
        return load_env_name_from_dataset(source_demo_hdf5)

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

    def _build_replay_reset_state(self, source_episode_idx: int, object_translation: np.ndarray):
        reset_state = self._load_reset_state(source_episode_idx)
        reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
        if object_translation is not None:
            idx = self.replay_object_state_indices
            reset_state["states"][idx] += np.asarray(object_translation[: len(idx)], dtype=np.float64)
        return reset_state

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
                "Lift replay-consistent schedule: "
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
                    motion_exec_state = self._init_motion_exec_state()
                    reset_state = self._build_replay_reset_state(
                        source_episode_idx=i,
                        object_translation=np.asarray(obj_trans_vec, dtype=np.float32),
                    )
                    obs = env.reset_to(reset_state)
                    current_agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
                    trans_sofar = np.zeros(3, dtype=np.float32)

                    for _ in range(skill_1_frame):
                        source_exec_action = np.asarray(source_demo["action"][current_frame], dtype=np.float32)
                        source_motion_action = np.asarray(
                            self._get_motion_action(source_demo, current_frame), dtype=np.float32
                        )
                        extra_step = np.asarray(translation_increments[current_frame], dtype=np.float32)
                        step_action = np.asarray(source_motion_action[:3] + extra_step, dtype=np.float32)

                        state = current_agent_pos.copy()
                        traj_states.append(state)

                        trans_sofar = (
                            np.asarray(current_agent_pos[:3], dtype=np.float32)
                            - np.asarray(source_demo["state"][current_frame][:3], dtype=np.float32)
                        )
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

                        obs, _, _, _ = env.step(action)
                        current_agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
                        current_frame += 1

                    num_frames = source_demo["state"].shape[0]
                    if current_frame < num_frames:
                        trans_sofar = (
                            np.asarray(current_agent_pos[:3], dtype=np.float32)
                            - np.asarray(source_demo["state"][current_frame][:3], dtype=np.float32)
                        )

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
        finally:
            env.close()

        save_path = (
            f"{self.data_root}/datasets/generated/{self.generated_name}_{self.gen_name}_{n_demos}.zarr"
        )
        self.save_episodes(generated_episodes, save_path)
