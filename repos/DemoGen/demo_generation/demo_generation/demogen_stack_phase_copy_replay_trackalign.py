import numpy as np
from termcolor import cprint
from tqdm import tqdm

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from demo_generation.demogen_stack_phase_copy_replay_consistent import (
    StackPhaseCopyReplayConsistentDemoGen,
)


class StackPhaseCopyReplayTrackAlignDemoGen(StackPhaseCopyReplayConsistentDemoGen):
    """
    Stack-specific replay-in-the-loop fork that tracks the shifted source state
    directly during motion phases.

    Compared with StackPhaseCopyReplayConsistentDemoGen:
    - keep the old line untouched
    - still reset object / target with pure translation augmentation
    - during motion-1 / motion-2, synthesize executable xyz from the desired
      next shifted source state, instead of relying on deferred correction
      pulses on top of the source motion interface
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.track_align_use_next_state = bool(
            getattr(cfg, "track_align_use_next_state", True)
        )
        cprint(
            (
                "Stack replay-track-align fork: "
                f"use_next_state={self.track_align_use_next_state}, "
                "motion xyz is encoded from replay error to shifted source state."
            ),
            "cyan",
        )

    def _build_tracking_step_xyz(
        self,
        source_demo,
        current_frame: int,
        current_agent_pos: np.ndarray,
        translation_offset: np.ndarray,
    ) -> np.ndarray:
        source_states = np.asarray(source_demo["state"], dtype=np.float32)
        current_frame = int(current_frame)
        if self.track_align_use_next_state:
            target_frame = min(current_frame + 1, source_states.shape[0] - 1)
        else:
            target_frame = current_frame
        desired_agent_pos = (
            np.asarray(source_states[target_frame, :3], dtype=np.float32)
            + np.asarray(translation_offset, dtype=np.float32)
        )
        return (desired_agent_pos - np.asarray(current_agent_pos[:3], dtype=np.float32)).astype(
            np.float32
        )

    def _build_tracking_motion_exec_action(
        self,
        step_xyz: np.ndarray,
        source_exec_action: np.ndarray,
        motion_exec_state,
    ) -> np.ndarray:
        exec_xyz = self._encode_motion_exec_xyz(
            np.asarray(step_xyz, dtype=np.float32),
            motion_exec_state=motion_exec_state,
        )
        return np.concatenate(
            [exec_xyz.astype(np.float32), np.asarray(source_exec_action[3:], dtype=np.float32)],
            axis=0,
        )

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
                "Stack replay-track-align schedule: "
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
                            step_xyz = self._build_tracking_step_xyz(
                                source_demo=source_demo,
                                current_frame=current_frame,
                                current_agent_pos=current_agent_pos,
                                translation_offset=obj_trans_vec,
                            )
                            action = self._build_tracking_motion_exec_action(
                                step_xyz=step_xyz,
                                source_exec_action=source_exec_action,
                                motion_exec_state=motion_exec_state,
                            )
                        elif stage == "motion2":
                            step_xyz = self._build_tracking_step_xyz(
                                source_demo=source_demo,
                                current_frame=current_frame,
                                current_agent_pos=current_agent_pos,
                                translation_offset=tar_trans_vec,
                            )
                            action = self._build_tracking_motion_exec_action(
                                step_xyz=step_xyz,
                                source_exec_action=source_exec_action,
                                motion_exec_state=motion_exec_state,
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
