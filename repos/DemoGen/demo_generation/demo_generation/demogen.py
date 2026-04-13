# from re import T
# from turtle import st
from diffusion_policies.common.replay_buffer import ReplayBuffer
# from regex import I
# import pcd_visualizer
import numpy as np
import copy
import os
import json
import shutil
os.environ.setdefault("MPLBACKEND", "Agg")

import zarr
import matplotlib
matplotlib.use("Agg")
from termcolor import cprint
from demo_generation.mask_util import restore_and_filter_pcd, filter_points_by_world_mask
import imageio
from scipy.spatial import cKDTree
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import logging
from omegaconf import DictConfig, ListConfig


class DemoGen:
    def __init__(self, cfg):
        self.data_root = cfg.data_root
        self.source_name = cfg.source_name
        
        self.task_n_object = cfg.task_n_object
        self.use_linear_interpolation = cfg.use_linear_interpolation
        self.interpolate_step_size = cfg.interpolate_step_size

        self.use_manual_parsing_frames = cfg.use_manual_parsing_frames
        self.parsing_frames = cfg.parsing_frames
        self.mask_names = cfg.mask_names
        if (not self.use_manual_parsing_frames) and (self.parsing_frames is not None):
            configured_skill_1 = self.parsing_frames.get("skill-1", None)
            if configured_skill_1 not in [None, 0]:
                cprint(
                    "[WARN] parsing_frames.skill-1 is set but use_manual_parsing_frames=False, "
                    "so the manual frame will be ignored.",
                    "yellow",
                )
        self.motion_retarget_mode = getattr(cfg, "motion_retarget_mode", "legacy")
        self.motion_action_key = getattr(cfg, "motion_action_key", "action")
        self.motion_exec_mode = getattr(cfg, "motion_exec_mode", "direct")
        self.motion_exec_xyz_strategy = getattr(cfg, "motion_exec_xyz_strategy", "replace")
        motion_exec_output_scale_xyz = getattr(cfg, "motion_exec_output_scale_xyz", None)
        self.motion_exec_output_scale_xyz = None
        if motion_exec_output_scale_xyz is not None:
            self.motion_exec_output_scale_xyz = np.asarray(motion_exec_output_scale_xyz, dtype=np.float32)
            if self.motion_exec_output_scale_xyz.shape != (3,):
                raise ValueError(
                    f"motion_exec_output_scale_xyz must have shape (3,), got {self.motion_exec_output_scale_xyz.shape}"
                )
        motion_exec_pulse_scale_xyz = getattr(cfg, "motion_exec_pulse_scale_xyz", None)
        self.motion_exec_pulse_scale_xyz = None
        if motion_exec_pulse_scale_xyz is not None:
            self.motion_exec_pulse_scale_xyz = np.asarray(motion_exec_pulse_scale_xyz, dtype=np.float32)
            if self.motion_exec_pulse_scale_xyz.shape != (3,):
                raise ValueError(
                    f"motion_exec_pulse_scale_xyz must have shape (3,), got {self.motion_exec_pulse_scale_xyz.shape}"
                )
        self.motion_exec_pulse_threshold_ratio = float(getattr(cfg, "motion_exec_pulse_threshold_ratio", 1.0))

        self.gen_name = cfg.generation.range_name
        self.object_trans_range = cfg.trans_range[self.gen_name]["object"]
        self.target_trans_range = cfg.trans_range[self.gen_name]["target"]

        self.n_gen_per_source = cfg.generation.n_gen_per_source
        self.render_video = cfg.generation.render_video
        if self.render_video:
            cprint("[NOTE] Rendering video is enabled. It takes ~10s to render a single generated trajectory.", "yellow")
        self.gen_mode = cfg.generation.mode

        source_zarr = os.path.join(self.data_root, "datasets", "source", self.source_name + ".zarr")
        self._load_from_zarr(source_zarr)

    def _load_from_zarr(self, zarr_path):
        cprint(f"Loading data from {zarr_path}", "blue")
        zarr_root = zarr.open(zarr_path, mode="r")
        available_keys = set(zarr_root["data"].keys())
        if self.motion_action_key not in available_keys:
            if self.motion_action_key != "action":
                cprint(
                    f"[WARN] motion_action_key={self.motion_action_key} not found in source zarr. Falling back to action.",
                    "yellow",
                )
            self.motion_action_key = "action"

        replay_keys = ["state", "action", "point_cloud"]
        if self.motion_action_key not in replay_keys:
            replay_keys.append(self.motion_action_key)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=replay_keys)
        self.n_source_episodes = self.replay_buffer.n_episodes
        self.demo_name = zarr_path.split("/")[-1].split(".")[0]
        cprint(f"Motion retarget action key: {self.motion_action_key}", "cyan")
        cprint(f"Motion exec mode: {self.motion_exec_mode}", "cyan")
        cprint(f"Motion exec xyz strategy: {self.motion_exec_xyz_strategy}", "cyan")

    def resolve_parsing_frame(self, frame_name: str, episode_idx: int):
        if self.parsing_frames is None:
            raise ValueError(
                f"parsing_frames is required when resolving manual frame '{frame_name}'"
            )
        if frame_name not in self.parsing_frames:
            raise KeyError(f"Missing parsing_frames['{frame_name}']")

        value = self.parsing_frames[frame_name]
        if value is None:
            return None

        if isinstance(value, (list, tuple, ListConfig)):
            if episode_idx >= len(value):
                raise IndexError(
                    f"parsing_frames['{frame_name}'] only has {len(value)} entries, "
                    f"cannot index source episode {episode_idx}"
                )
            value = value[episode_idx]
        elif isinstance(value, (dict, DictConfig)):
            for key in (str(episode_idx), episode_idx, f"demo_{episode_idx + 1}"):
                if key in value:
                    value = value[key]
                    break
            else:
                raise KeyError(
                    f"parsing_frames['{frame_name}'] has no entry for source episode {episode_idx}"
                )

        if value is None:
            return None
        return int(value)

    @staticmethod
    def _init_motion_exec_state():
        return {
            "residual_xyz": np.zeros(3, dtype=np.float32),
        }
    
    def generate_trans_vectors(self, trans_range, n_demos, mode="random"):
        """
        Argument: trans_range: (2, 3)
            [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        Return: A list of translation vectors. (n_demos, 3)
        """
        x_min, x_max, y_min, y_max = trans_range[0][0], trans_range[1][0], trans_range[0][1], trans_range[1][1]
        if mode == "grid":
            n_side = int(np.sqrt(n_demos))
            # print(f"n_side: {n_side}, n_demos: {n_demos}")
            if n_side ** 2 != n_demos or n_demos == 1:
                raise ValueError("In grid mode, n_demos must be a squared number larger than 1")
            x_values = [x_min + i / (n_side - 1) * (x_max - x_min) for i in range(n_side)]
            y_values = [y_min + i / (n_side - 1) * (y_max - y_min) for i in range(n_side)]
            xyz = list(set([(x, y, 0) for x in x_values for y in y_values]))
            # assert len(xyz) == n_demos
            return np.array(xyz)
        elif mode == "random":
            xyz = []
            for _ in range(n_demos):
                x = np.random.random() * (x_max - x_min) + x_min
                y = np.random.random() * (y_max - y_min) + y_min
                xyz.append([x, y, 0])
            return np.array(xyz)
        else:
            raise NotImplementedError
        
    def generate_offset_trans_vectors(self, offsets, trans_range, n_demos, mode="grid"):
        """
        For each point (translation vector) generate n_demos in trans_range.
        points_pos: (2, n_points)
            [[x1, x2, ..., xn], [y1, y2, ..., yn]]
        # NOTE: Small-range offsets are used in the experiments in our paper. However, we later found it is 
            in many times unnecessary, if we add random jitter augmentations to the point clouds when training the policy.
        """
        trans_vectors = []

        for x_offset, y_offset in zip(offsets[0], offsets[1]):
            trans_range_offset = copy.deepcopy(trans_range)
            trans_range_offset[0][0] += x_offset
            trans_range_offset[1][0] += x_offset
            trans_range_offset[0][1] += y_offset
            trans_range_offset[1][1] += y_offset
            trans_vector = self.generate_trans_vectors(trans_range_offset, n_demos, mode)
            trans_vectors.append(trans_vector)

        return np.concatenate(trans_vectors, axis=0)
    
    def get_objects_pcd_from_sam_mask(self, pcd, demo_idx, object_or_target="object"):
        assert object_or_target in ["object", "target"]
        mask_root = os.path.join(self.data_root, f"sam_mask/{self.source_name}/{demo_idx}")
        mask = imageio.imread(os.path.join(mask_root, f"{self.mask_names[object_or_target]}.jpg"))
        mask = mask > 128
        camera_info_path = os.path.join(mask_root, "camera_info.json")
        if os.path.exists(camera_info_path):
            with open(camera_info_path, "r", encoding="utf-8") as f:
                camera_info = json.load(f)
            intrinsic_matrix = np.asarray(camera_info["intrinsics"], dtype=np.float32)
            extrinsic_matrix = np.asarray(camera_info["extrinsics"], dtype=np.float32)
            image_size = tuple(camera_info.get("image_size", [mask.shape[1], mask.shape[0]]))
            filtered_pcd = filter_points_by_world_mask(
                pcd,
                mask,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
                image_size=image_size,
            )
        else:
            filtered_pcd = restore_and_filter_pcd(pcd, mask)
        return filtered_pcd
    
    def generate_demo(self):
        if self.task_n_object == 1:
            self.one_stage_augment(self.n_gen_per_source, self.render_video, self.gen_mode)
        elif self.task_n_object == 2:
            self.two_stage_augment(self.n_gen_per_source, self.render_video, self.gen_mode)
        else:
            raise NotImplementedError

    def _get_motion_action(self, source_demo, frame_idx):
        return source_demo[self.motion_action_key][frame_idx]

    def _compute_motion_start_pos(self, source_demo, start_frame=0):
        if self.motion_retarget_mode == "legacy":
            return source_demo["state"][start_frame][:3] - self._get_motion_action(source_demo, start_frame)[:3]
        if self.motion_retarget_mode == "state_based":
            return source_demo["state"][start_frame][:3].copy()
        raise ValueError(f"Unsupported motion_retarget_mode: {self.motion_retarget_mode}")

    def _encode_motion_exec_xyz(self, step_action, motion_exec_state=None):
        encoded_xyz = np.asarray(step_action, dtype=np.float32).copy()
        if self.motion_exec_mode == "direct":
            return encoded_xyz
        if self.motion_exec_mode == "osc_pose_delta":
            if self.motion_exec_output_scale_xyz is None:
                raise ValueError("motion_exec_output_scale_xyz must be set when motion_exec_mode=osc_pose_delta")
            encoded_xyz = encoded_xyz / self.motion_exec_output_scale_xyz
            encoded_xyz = np.clip(encoded_xyz, -1.0, 1.0)
            return encoded_xyz.astype(np.float32)
        if self.motion_exec_mode == "osc_pose_pulse":
            if self.motion_exec_pulse_scale_xyz is None:
                raise ValueError("motion_exec_pulse_scale_xyz must be set when motion_exec_mode=osc_pose_pulse")
            if motion_exec_state is None:
                raise ValueError("motion_exec_state must be provided when motion_exec_mode=osc_pose_pulse")

            residual_xyz = motion_exec_state["residual_xyz"]
            residual_xyz += encoded_xyz
            exec_xyz = np.zeros(3, dtype=np.float32)
            threshold_xyz = self.motion_exec_pulse_scale_xyz * self.motion_exec_pulse_threshold_ratio

            for axis in range(3):
                scale = self.motion_exec_pulse_scale_xyz[axis]
                threshold = threshold_xyz[axis]
                if scale <= 0 or threshold <= 0:
                    raise ValueError("motion_exec_pulse_scale_xyz and threshold must be positive")

                if residual_xyz[axis] >= threshold:
                    exec_xyz[axis] = min(1.0, np.floor(residual_xyz[axis] / scale))
                elif residual_xyz[axis] <= -threshold:
                    exec_xyz[axis] = max(-1.0, -np.floor((-residual_xyz[axis]) / scale))

                residual_xyz[axis] -= exec_xyz[axis] * scale

            motion_exec_state["residual_xyz"] = residual_xyz
            return exec_xyz.astype(np.float32)
        raise ValueError(f"Unsupported motion_exec_mode: {self.motion_exec_mode}")

    def _build_motion_exec_action(
        self,
        step_action,
        source_exec_action,
        motion_exec_state=None,
        source_motion_action=None,
    ):
        if self.motion_exec_xyz_strategy == "replace":
            exec_xyz = self._encode_motion_exec_xyz(step_action, motion_exec_state=motion_exec_state)
            return np.concatenate([exec_xyz, source_exec_action[3:]], axis=0)
        if self.motion_exec_xyz_strategy == "source_plus_correction":
            if source_motion_action is None:
                raise ValueError(
                    "source_motion_action must be provided when motion_exec_xyz_strategy=source_plus_correction"
                )
            # Preserve the original executable controller action and only encode
            # the extra correction needed to retarget the source trajectory.
            correction_step = np.asarray(step_action, dtype=np.float32) - np.asarray(source_motion_action[:3], dtype=np.float32)
            correction_xyz = self._encode_motion_exec_xyz(correction_step, motion_exec_state=motion_exec_state)
            exec_xyz = np.clip(np.asarray(source_exec_action[:3], dtype=np.float32) + correction_xyz, -1.0, 1.0)
            return np.concatenate([exec_xyz, source_exec_action[3:]], axis=0)
        raise ValueError(f"Unsupported motion_exec_xyz_strategy: {self.motion_exec_xyz_strategy}")
        
    def parse_frames_two_stage(self, pcds, demo_idx, ee_poses, distance_mode="ee2pcd", threshold_1=0.15, threshold_2=0.235, threshold_3=0.275,):
        """
        There are two ways to parse the frames of whole trajectory into object-centric segments: (1) Either by comparing the distance between 
            the end-effector and the object point cloud, (2) Or by manually specifying the frames when `self.use_manual_parsing_frames = True`.
        This function implements the first way. While it is an automatic process, you need to tune the distance thresholds to achieve a clean parse.
        Since DemoGen requires very few source demos, it is also feasible (actually recommended) to manually specify the frames for parsing.
        To manually decide the parsing frames, you can set the translation vectors to zero, run the DemoGen code, render the videos, and check the
            frame_idx on the left top of the video.
        """
        assert distance_mode in ["ee2pcd", "pcd2pcd"]
        stage = 1
        for i in range(pcds.shape[0]):
            object_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "object")
            target_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "target")
            if stage == 1:
                if distance_mode == "ee2pcd":            
                    if self.average_distance_to_point_cloud(ee_poses[i], object_pcd) <= threshold_1:
                        # visualizer.visualize_pointcloud(pcds[i])
                        stage = 2
                        skill_1_frame = i
                        object_origin_pcd = object_pcd
                elif distance_mode == "pcd2pcd":
                    obj_bbox = self.pcd_bbox(object_pcd)
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, _, ee_pcd = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
                    if self.chamfer_distance(object_pcd, ee_pcd) <= threshold_1:
                        stage = 2
                        skill_1_frame = i
                        object_origin_pcd = object_pcd
                        
            elif stage == 2:
                if distance_mode == "ee2pcd":
                    if self.average_distance_to_point_cloud(ee_poses[i], object_origin_pcd) >= threshold_2:
                        stage = 3
                        motion_2_frame = i
                        # visualizer.visualize_pointcloud(pcds[i])
                elif distance_mode == "pcd2pcd":
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, ee_obj_pcd = self.pcd_divide(source_pcd, [tar_bbox])
                    if self.chamfer_distance(ee_obj_pcd, object_origin_pcd) >= threshold_2:
                        stage = 3
                        motion_2_frame = i
                        
            elif stage == 3:
                if distance_mode == "ee2pcd":
                    if self.average_distance_to_point_cloud(ee_poses[i], target_pcd) <= threshold_3:
                        skill_2_frame = i
                        # visualizer.visualize_pointcloud(pcds[i])
                        break
                elif distance_mode == "pcd2pcd":
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, ee_obj_pcd = self.pcd_divide(source_pcd, [tar_bbox])
                    if self.chamfer_distance(ee_obj_pcd, target_pcd) <= threshold_3:
                        skill_2_frame = i
                        break
                
        print(f"Stage 1: {skill_1_frame}, Pre-2: {motion_2_frame}, Stage 2: {skill_2_frame}")
        return skill_1_frame, motion_2_frame, skill_2_frame

    def two_stage_augment(self, n_demos, render_video=False, gen_mode='random'):
        """
        An implementation of the DemoGen augmentation process for manipulation tasks involving two objects. More specifically, the task contains
            4 sub-stages: (1) Motion-1, (2) Skill-1, (3) Motion-2, (4) Skill-2.
        TODO: Refactor the code to support tasks involving any number of objects and manipulation stages.
        """
        # Prepare translation vectors
        trans_vectors = []      # [n_demos, 6 (obj_xyz + targ_xyz)]
        if gen_mode == 'random':
            for _ in range(n_demos):
                obj_xyz = self.generate_trans_vectors(self.object_trans_range, 1, mode="random")[0]
                targ_xyz = self.generate_trans_vectors(self.target_trans_range, 1, mode="random")[0]
                trans_vectors.append(np.concatenate([obj_xyz, targ_xyz], axis=0))
        elif gen_mode == 'grid':
            def check_fourth_power(arr):
                fourth_roots = np.power(arr, 1/4)
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
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            
            if self.use_manual_parsing_frames:
                skill_1_frame = self.resolve_parsing_frame("skill-1", i)
                motion_2_frame = self.resolve_parsing_frame("motion-2", i)
                skill_2_frame = self.resolve_parsing_frame("skill-2", i)
            else:
                ee_poses = source_demo["state"][:, :3]
                skill_1_frame, motion_2_frame, skill_2_frame = self.parse_frames_two_stage(pcds, i, ee_poses)
            # print(f"Skill-1: {skill_1_frame}, Motion-2: {motion_2_frame}, Skill-2: {skill_2_frame}")
            
            pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
            pcd_tar = self.get_objects_pcd_from_sam_mask(pcds[0], i, "target")
            obj_bbox = self.pcd_bbox(pcd_obj)
            tar_bbox = self.pcd_bbox(pcd_tar)

            # Generate demos according to translation vectors
            for trans_vec in tqdm(trans_vectors):
                obj_trans_vec = trans_vec[:3]
                tar_trans_vec = trans_vec[3:6]
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)

                ############# stage {motion-1} starts #############
                trans_togo = obj_trans_vec.copy()
                source_demo = self.replay_buffer.get_episode(i)
                start_pos = self._compute_motion_start_pos(source_demo)
                end_pos = source_demo["state"][skill_1_frame-1][:3] + trans_togo
                
                if self.use_linear_interpolation:
                    step_action = (end_pos - start_pos) / skill_1_frame
                else:
                    xy_stage_frame = skill_1_frame
                    step_actions = []
                    z_action = end_pos[2] - start_pos[2]
                    xy_action = end_pos[:2] - start_pos[:2]
                    
                    if z_action != 0:
                        z_action = np.sign(z_action) * round(np.abs(z_action), 3)
                        z_step_num = int(np.abs(z_action) / 0.015)
                        for _ in range(z_step_num):
                            step_actions.append(np.array([0, 0, np.sign(z_action) * 0.015]))
                            xy_stage_frame -= 1
                    
                    if xy_stage_frame > 0:
                        action = xy_action / xy_stage_frame
                        for _ in range(xy_stage_frame):
                            step_actions.append(np.array([*action, 0]))
                            
                    # inverse the step_actions
                    step_actions = step_actions[::-1]
                motion_exec_state = self._init_motion_exec_state()
                
                for j in range(skill_1_frame):
                    if not self.use_linear_interpolation:
                        step_action = step_actions[j]

                    source_exec_action = source_demo["action"][current_frame]
                    source_motion_action = self._get_motion_action(source_demo, current_frame)
                    traj_actions.append(
                        self._build_motion_exec_action(
                            step_action,
                            source_exec_action,
                            motion_exec_state,
                            source_motion_action=source_motion_action,
                        )
                    )
                    trans_this_frame = step_action - source_motion_action[:3]
                    
                    trans_sofar[:2] += trans_this_frame[:2] # for x y only

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj, pcd_tar, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
                    # visualizer.visualize_pointcloud(pcd_robot)
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj, pcd_tar], axis=0))

                    current_frame += 1
                ############## stage {motion-1} ends #############
                
                ############# stage {skill-1} starts #############
                is_stage_motion2 = current_frame >= motion_2_frame
                while not is_stage_motion2:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_obj_robot, pcd_tar], axis=0))

                    current_frame += 1
                    is_stage_motion2 = current_frame >= motion_2_frame
                ############## stage {skill-1} ends #############

                ############# stage {motion-2} starts #############
                trans_togo = tar_trans_vec - obj_trans_vec
                start_pos = self._compute_motion_start_pos(source_demo, motion_2_frame)
                end_pos = source_demo["state"][skill_2_frame-1][:3] + trans_togo
                
                if self.use_linear_interpolation:
                    step_action = (end_pos - start_pos) / (skill_2_frame - motion_2_frame)
                else:
                    xy_stage_frame = skill_2_frame - motion_2_frame
                    step_actions = []
                    z_action = end_pos[2] - start_pos[2]
                    xy_action = end_pos[:2] - start_pos[:2]
                    
                    if z_action != 0:
                        z_action = np.sign(z_action) * round(np.abs(z_action), 3)
                        z_step_num = int(np.abs(z_action) / 0.015)
                        for _ in range(z_step_num):
                            step_actions.append(np.array([0, 0, np.sign(z_action) * 0.015]))
                            xy_stage_frame -= 1
                    
                    if xy_stage_frame > 0:
                        action = xy_action / xy_stage_frame
                        for _ in range(xy_stage_frame):
                            step_actions.append(np.array([*action, 0]))
                motion_exec_state = self._init_motion_exec_state()

                for k in range(skill_2_frame - motion_2_frame):
                    if not self.use_linear_interpolation:
                        step_action = step_actions[k]
                    source_exec_action = source_demo["action"][current_frame]
                    source_motion_action = self._get_motion_action(source_demo, current_frame)
                    traj_actions.append(
                        self._build_motion_exec_action(
                            step_action,
                            source_exec_action,
                            motion_exec_state,
                            source_motion_action=source_motion_action,
                        )
                    )
                    trans_this_frame = step_action - source_motion_action[:3]
                    
                    trans_sofar[:2] += trans_this_frame[:2] # for x y only

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)

                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_obj_robot, pcd_tar], axis=0))

                    current_frame += 1
                ############## stage {motion-2} ends #############
                    
                ############# stage {skill-2} starts #############
                later_frames = self.translate_all_frames(source_demo, tar_trans_vec, current_frame)
                ############# stage {skill-2} ends #############

                generated_episode = {
                    "state": np.concatenate([traj_states, later_frames["state"]], axis=0) if len(traj_states) > 0 else later_frames["state"],
                    "action": np.concatenate([traj_actions, later_frames["action"]], axis=0) if len(traj_actions) > 0 else later_frames["action"],
                    "point_cloud": np.concatenate([traj_pcds, later_frames["point_cloud"]], axis=0) if len(traj_pcds) > 0 else later_frames["point_cloud"]
                }
                generated_episodes.append(generated_episode)

                if render_video:
                    vfunc = np.vectorize("{:.3f}".format)
                    video_name = f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_tar[{np.round(tar_trans_vec[0], 3)},{np.round(tar_trans_vec[1], 3)}].mp4"
                    video_path = os.path.join(self.data_root, "videos", self.source_name, self.gen_name, video_name)
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
                # self._examine_episode(generated_episode, aug_setting, i, obj_trans_vec, tar_trans_vec)
                ############## end of generating one episode ##############

        # save the generated episodes
        save_path = os.path.join(self.data_root, "datasets", "generated", f"{self.source_name}_{self.gen_name}_{n_demos}.zarr")
        self.save_episodes(generated_episodes, save_path)
        
    def parse_frames_one_stage(self, pcds, demo_idx, ee_poses, distance_mode="pcd2pcd", threshold_1=0.23):
        assert distance_mode in ["ee2pcd", "pcd2pcd"]
        start_frame = None
        for i in range(pcds.shape[0]):
            object_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "object")
            if distance_mode == "pcd2pcd":
                obj_bbox = self.pcd_bbox(object_pcd)
                source_pcd = pcds[i].copy()
                _, pcd_ee = self.pcd_divide(source_pcd, [obj_bbox])
                if self.chamfer_distance(pcd_ee, object_pcd) <= threshold_1:
                    print(f"Stage starts at frame {i}")
                    start_frame = i
                    break
            elif distance_mode == "ee2pcd":
                if self.average_distance_to_point_cloud(ee_poses[i], object_pcd) <= threshold_1:
                    print(f"Stage starts at frame {i}")
                    start_frame = i
                    break
        if start_frame is None:
            raise RuntimeError(
                f"Failed to parse one-stage motion segment for demo_idx={demo_idx}. "
                f"distance_mode={distance_mode}, threshold_1={threshold_1}"
            )
        return start_frame

    def one_stage_augment(self, n_demos, render_video=False, gen_mode='random'):
        # Prepare translation vectors
        trans_vectors = []      # [n_demos, 6 (obj_xyz + targ_xyz)]
        if gen_mode == 'random':
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="random")
        elif gen_mode == 'grid':
            def check_squared_number(arr):
                return np.isclose(np.sqrt(arr), np.round(np.sqrt(arr)))
            assert check_squared_number(n_demos), "n_demos must be a squared number"
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="grid")


        generated_episodes = []

        for i in tqdm(range(self.n_source_episodes)):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            # visualizer.visualize_pointcloud(pcds[0])
            
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
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)

                ############# stage {motion-1} starts #############
                trans_togo = obj_trans_vec.copy()
                source_demo = self.replay_buffer.get_episode(i)
                start_pos = self._compute_motion_start_pos(source_demo)
                end_pos = source_demo["state"][skill_1_frame-1][:3] + trans_togo
                desired_pos = start_pos.copy()
                
                if self.use_linear_interpolation:
                    step_action = (end_pos - start_pos) / skill_1_frame
                else:
                    xy_stage_frame = skill_1_frame
                    step_actions = []
                    z_action = end_pos[2] - start_pos[2]
                    xy_action = end_pos[:2] - start_pos[:2]
                    
                    if z_action != 0:
                        z_action = np.sign(z_action) * round(np.abs(z_action), 3)
                        z_step_num = int(np.abs(z_action) / 0.015)
                        for _ in range(z_step_num):
                            step_actions.append(np.array([0, 0, np.sign(z_action) * 0.015]))
                            xy_stage_frame -= 1
                    
                    if xy_stage_frame > 0:
                        action = xy_action / xy_stage_frame
                        for _ in range(xy_stage_frame):
                            step_actions.append(np.array([*action, 0]))
                            
                    # inverse the step_actions
                    step_actions = step_actions[::-1]
                motion_exec_state = self._init_motion_exec_state()
                
                for j in range(skill_1_frame):
                    if not self.use_linear_interpolation:
                        step_action = step_actions[j]

                    source_exec_action = source_demo["action"][current_frame]
                    source_motion_action = self._get_motion_action(source_demo, current_frame)
                    traj_actions.append(
                        self._build_motion_exec_action(
                            step_action,
                            source_exec_action,
                            motion_exec_state,
                            source_motion_action=source_motion_action,
                        )
                    )

                    if self.motion_retarget_mode == "legacy":
                        trans_this_frame = step_action - source_motion_action[:3]
                        trans_sofar[:2] += trans_this_frame[:2] # for x y only
                    else:
                        desired_pos += step_action
                        source_pos = source_demo["state"][current_frame][:3]
                        trans_sofar = desired_pos - source_pos

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox])
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj], axis=0))

                    current_frame += 1
                ############## stage {motion-1} ends #############
                num_frames = source_demo["state"].shape[0]
                ############# stage {skill-1} starts #############
                while current_frame < num_frames:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    pcd_obj_robot = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(pcd_obj_robot)

                    current_frame += 1
                    ############## stage {skill-1} ends #############

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
                    vfunc = np.vectorize("{:.3f}".format)
                    video_name = f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}].mp4"
                    video_path = os.path.join(self.data_root, "videos", self.source_name, self.gen_name, video_name)
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
                ############## end of generating one episode ##############
        
        # save the generated episodes
        save_path = os.path.join(self.data_root, "datasets", "generated", f"{self.source_name}_{self.gen_name}_{n_demos}.zarr")
        self.save_episodes(generated_episodes, save_path)

    def _examine_episode(self, episode, aug_setting, episode_id, obj_trans, tar_trans):
        """
        Examine the episode to see if the point cloud is correct
        """
        cprint(f"Examine episode {episode_id}", "green")
        
        debug_save_dir = f"/home/zhengrong/data/dp3/debug/demo_generation/{self.demo_name}/{aug_setting}/{episode_id}-{vfunc(obj_trans)}-{vfunc(tar_trans)}"
        os.makedirs(debug_save_dir, exist_ok=True)
        cprint(f"Saving episode examination to {debug_save_dir}", "yellow")
        vis = visualizer.Visualizer()
        ep_len = episode["state"].shape[0]
        pcds = []
        for i in range(0, 200, 1):
            # print(f"Frame {i}")
            cprint(f"action {i}: {vfunc(episode['action'][i])}", "blue")
            cprint(f"state {i}: {vfunc(episode['state'][i][:3])}", "blue")
            pcd = episode["point_cloud"][i]
            pcds.append(pcd)
            vis.save_visualization_to_file(pointcloud=pcd, file_path=os.path.join(debug_save_dir, f"frame_{i}.html"))

        # vis.preview_in_open3d(pcds)

    def _examine_actions(self, demo_trajectory):
        vfunc = np.vectorize("{:.2e}".format)
        actions = demo_trajectory["action"][:30]
        ee_actions = actions[:, :3]
        intensity = np.linalg.norm(ee_actions, axis=1)
        print(f"ee_actions: {vfunc(ee_actions)}")
        print(f"inensity: {vfunc(intensity)}")

    def save_episodes(self, generated_episodes, save_dir):
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        elif os.path.isfile(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        cprint(f"Saving data to {save_dir}", "green")
        # self.replay_buffer.save_to_store(save_dir)

        zarr_root = zarr.group(save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        state_arrays = np.concatenate([ep["state"] for ep in generated_episodes], axis=0)
        point_cloud_arrays = np.concatenate([ep["point_cloud"] for ep in generated_episodes], axis=0)
        action_arrays = np.concatenate([ep["action"] for ep in generated_episodes], axis=0)
        episode_ends = []
        count = 0
        for ep in generated_episodes:
            count += len(ep["state"])
            episode_ends.append(count)
        episode_ends_arrays = np.array(episode_ends)
        print(episode_ends_arrays)

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        action_chunk_size = (100, action_arrays.shape[1])
        zarr_data.create_dataset('agent_pos', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
        if all("_source_episode_idx" in ep for ep in generated_episodes):
            source_episode_idx_arrays = np.asarray(
                [ep["_source_episode_idx"] for ep in generated_episodes],
                dtype=np.int64,
            )
            zarr_meta.create_dataset(
                'source_episode_idx',
                data=source_episode_idx_arrays,
                chunks=(min(100, len(source_episode_idx_arrays)),),
                dtype='int64',
                overwrite=True,
                compressor=compressor,
            )
        if all("_object_translation" in ep for ep in generated_episodes):
            object_translation_arrays = np.stack(
                [np.asarray(ep["_object_translation"], dtype=np.float32) for ep in generated_episodes],
                axis=0,
            )
            zarr_meta.create_dataset(
                'object_translation',
                data=object_translation_arrays,
                chunks=(min(100, len(object_translation_arrays)), object_translation_arrays.shape[1]),
                dtype='float32',
                overwrite=True,
                compressor=compressor,
            )
        if all("_motion_frame_count" in ep for ep in generated_episodes):
            motion_frame_count_arrays = np.asarray(
                [ep["_motion_frame_count"] for ep in generated_episodes],
                dtype=np.int64,
            )
            zarr_meta.create_dataset(
                'motion_frame_count',
                data=motion_frame_count_arrays,
                chunks=(min(100, len(motion_frame_count_arrays)),),
                dtype='int64',
                overwrite=True,
                compressor=compressor,
            )

        cprint(f'-'*50, 'cyan')
        # print shape
        cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
        cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
        cprint(f'Saved zarr file to {save_dir}', 'green')

        # save to hdf5
        import h5py
        save_dir = save_dir.replace('.zarr', '.hdf5')
        if os.path.exists(save_dir):
            os.remove(save_dir)
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('agent_pos', data=state_arrays, compression='gzip')
            f.create_dataset('point_cloud', data=point_cloud_arrays, compression='gzip')
            f.create_dataset('action', data=action_arrays, compression='gzip')
            f.create_dataset('episode_ends', data=episode_ends_arrays, compression='gzip')
        cprint(f'Saved hdf5 file to {save_dir}', 'green')

    @staticmethod
    def point_cloud_to_video(point_clouds, output_file, fps=20, elev=30, azim=45):
        """
        Converts a sequence of point cloud frames into a video.

        Args:
            point_clouds (list): A list of (N, 6) numpy arrays representing the point clouds.
            output_file (str): The path to the output video file.
            fps (int, optional): The frames per second of the output video. Defaults to 15.
            elev (float, optional): The elevation angle (in degrees) for the 3D view. Defaults to 30.
            azim (float, optional): The azimuth angle (in degrees) for the 3D view. Defaults to 45.
        """
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        all_points = np.concatenate(point_clouds, axis=0)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        xyz_min = min_vals[:3]
        xyz_max = max_vals[:3]
        xyz_span = np.maximum(xyz_max - xyz_min, 1e-3)
        xyz_pad = np.maximum(xyz_span * 0.08, 0.02)
        plot_min = xyz_min - xyz_pad
        plot_max = xyz_max + xyz_pad

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = imageio.get_writer(output_file, fps=fps)
        logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

        for frame, points in enumerate(point_clouds):
            ax.clear()
            color = points[:, 3:] / 255
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')

            ax.set_box_aspect((plot_max - plot_min).tolist())
            ax.set_xlim(plot_min[0], plot_max[0])
            ax.set_ylim(plot_min[1], plot_max[1])
            ax.set_zlim(plot_min[2], plot_max[2])

            ax.tick_params(axis='both', which='major', labelsize=8)
            
            formatter = FormatStrFormatter('%.1f')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.zaxis.set_major_formatter(formatter)

            ax.view_init(elev=elev, azim=azim)
            ax.text2D(0.05, 0.95, f'Frame: {frame}', transform=ax.transAxes, fontsize=14, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)

        writer.close()
        plt.close(fig)

    @staticmethod
    def pcd_divide(pcd, bbox_list):
        """
        pcd: (n, 6)
        bbox_list: list of (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        :return: list of pcds
        """
        masks = []
        selected_pcds = []
        for bbox in bbox_list:
            assert np.array(bbox).shape == (2, 3)
            masks.append(np.all(pcd[:, :3] > bbox[0], axis=1) & np.all(pcd[:, :3] < bbox[1], axis=1))
        # add the rest of the points to the last mask
        masks.append(np.logical_not(np.any(masks, axis=0)))
        for mask in masks:
            selected_pcds.append(pcd[mask])
        assert np.sum([len(p) for p in selected_pcds]) == pcd.shape[0]
        return selected_pcds

    @staticmethod
    def pcd_translate(pcd, trans_vec):
        """
        Translate the points with trans_vec
        pcd: (n, 6)
        trans_vec (3,)
        """
        pcd_ = pcd.copy()
        pcd_[:, :3] += trans_vec
        return pcd_

    @staticmethod
    def translate_all_frames(source_demo, trans_vec, start_frame=0):
        """
        Translate all frames in the source demo by trans_vec from start_frame
        """
        source_states = source_demo["state"]
        source_actions = source_demo["action"]
        source_pcds = source_demo["point_cloud"]
        # states = source_states[start_frame:] + np.array([*trans_vec, *trans_vec, *trans_vec])
        states = source_states[start_frame:].copy()
        states[:, :3] += trans_vec
        actions = source_actions[start_frame:]
        pcds = source_pcds[start_frame:].copy()   # [T, N_points, 6]
        pcds[:, :, :3] += trans_vec
        assert states.shape[0] == actions.shape[0] == pcds.shape[0] == source_states.shape[0] - start_frame
        return {
            "state": states,
            "action": actions,
            "point_cloud": pcds
        }

    @staticmethod
    def chamfer_distance(pcd1, pcd2):
        tree1 = cKDTree(pcd1[:, :3])
        tree2 = cKDTree(pcd2[:, :3])

        distances1 = [tree2.query(point[:3], k=1)[0] for point in pcd1]
        distances2 = [tree1.query(point[:3], k=1)[0] for point in pcd2]

        chamfer_dist = (np.mean(distances1) + np.mean(distances2)) / 2
        return chamfer_dist
    
    @staticmethod
    def average_distance_to_point_cloud(target_point, point_cloud):
        target_point = np.array(target_point)
        point_cloud = np.array(point_cloud)
        
        if point_cloud.shape[1] != target_point.shape[0]:
            point_cloud_coords = point_cloud[:, :3]
        else:
            point_cloud_coords = point_cloud
        
        distances = np.linalg.norm(point_cloud_coords - target_point, axis=1)
        average_distance = np.mean(distances)
        
        return average_distance
    
    @staticmethod
    def pcd_bbox(pcd, relax=True):
        min_vals = np.min(pcd[:, :3], axis=0)
        max_vals = np.max(pcd[:, :3], axis=0)
        if relax:
            min_vals -= 0.01
            max_vals += 0.01
            min_vals[2] = 0.0
        return np.array([min_vals, max_vals])

    
