[TOC]

# 原文内容
* Pick-Cube、Handle-Press 是单物体空间随机化任务；Stack-Cube 是双物体任务。
* 原文设置里，单物体任务从 1 条 source demonstration 生成 100 条空间增广 demonstrations；双物体任务因为初始配置组合更多，会生成 200 条 demonstrations。

按照 3 *Seed / 20 rollout 的最大值/平均值成功率统计
* Pick-Cube DemoGen 的 76/73，优于 10 Source 的 29/29，接近 25 Source 的 82/74；
* Handle-Press 从 17/16 提升到 100/100，与 10 Source 的 100/99 基本持平，并达到 25 Source 的 100/100。
* Stack-Cube 从 0/0 提升到 79/77，优于 10 Source 的 44/38，并接近 25 Source 的 95/93。

* 遗留问题：单视角观测下的 visual mismatch：随着物体在三维空间中移动，其可见外观和透视关系会发生变化，但合成点云仍保留 source demonstration 中的固定视角外观，因此 synthetic data 与真实观测之间存在偏差。

* 论文在 Pick-Cube 上进一步发现，当增广数据的空间覆盖或密度继续增加时，性能提升会逐渐饱和。
# Pipeline One-Stage New
## One-Stage



### 数据语义

```
selected4 demo / low_dim / depth
-> replayh1 light source zarr
-> schedule source zarr
-> v37 replayconsistent generate(diagfix)
-> relalign solve
-> consistency / success gate
-> replayobs low_dim.hdf5
-> robomimic train
```

#### source zarr

- 给 DemoGen 提供 source 轨迹。保留 action ，新增字段 replay_h1_delta ：从回放里读取的逐帧 state 差

#### schedule source zarr

- 按先 xy 后 z 的规则把 generate 要参考的 `motion_action` 整理出来

#### generated zarr

- 做空间增广，产出 template，写出来的 `state/action/point_cloud` 在replay语义下自洽，但不保证任务的成功

#### solved zarr

- 优化问题，以上一步的state / eef相对cube的位姿 / 与原 action 的距离三项约束，把模板轨迹修到任务成功，再把真实 replay 结果写回 action，过程较慢

#### replayobs low_dim.hdf5
- 在用solved zarr replay 情况下读取并导出 lowdim

---

### 1. 输入
- source demo：
- source low_dim：
- source depth

---

### 2. source

新线的 source 分两步：

1. `raw hdf5 -> replayh1 light source zarr`
2. `light source zarr -> schedule source zarr`

#### 2.1 replayh1 light source

脚本：

- `repos/DemoGen/real_world/convert_robomimic_hdf5_to_zarr_exec_replay_h1_light.py`

内容：

- 从 `demo.hdf5 / low_dim.hdf5 / depth.hdf5` 读取原始轨迹和观测。
- 保留原始 controller action，写到 zarr 里的 `data/action`。
- 用 robosuite 重放每一帧 `set_state -> step(action)`，得到前缀区间里真实执行出来的位移“如果从第 t 帧的状态出发，执行第 t 帧这条动作，末端实际会移动的位移”。
- 把这份 replay 标定结果写成 `replay_h1_delta`。
- 同时生成 `state / agent_pos / point_cloud`。

source zarr 里字段：
- `data/action`
  - controller 执行的 pulse action。
- `data/replay_h1_delta`
  - 用 replay 标定过的逐帧几何位移。
- `data/state`
  - 轨迹状态。
- `data/point_cloud`
  - 点云观测。

---

#### 2.2 schedule source

脚本：

- `repos/DemoGen/real_world/convert_source_zarr_original_schedule_replay_h1.py`

内容：

- 读取上一步的 source zarr。用 `replay_h1_delta` 在 `skill1` 前累积位移，重建 zero-translation 的原始单阶段 schedule。
```
1. 先求总位移
  total_xyz = replay_h1_delta[:skill1, :3].sum(axis=0)

2.再按 old one-stage 规则重分配
    z 单独拿出来
    按 z_step_size=0.015 切成很多个固定小步。
    比如总共要下探 -0.09，就拆成 6 帧，每帧 [0, 0, -0.015]
    剩下的帧数全给 xy，xy 平均分。
    比如总共 x=0.06, y=-0.03，剩 184 帧，就每帧给 [0.06/184, -0.03/184, 0]
    最后把顺序反过来：先走 xy；后面几帧再走 z
```
- 把这份 schedule 写回 `data/motion_action`。
- `data/action` 仍然保持 source demo 的action。

---

### 3. generate
配置文件：

- `repos/DemoGen/demo_generation/demo_generation/config/lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_selected4_d2467_diagfix.yaml`

命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/demo_generation

conda run -n demogen bash gen_demo.sh \
  lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_selected4_d2467_diagfix \
  test grid 25 False
```
内容：
- 以 `motion_action` 作为几何参考轨迹。
- 把目标平移 `object_translation` 分配到skill1-轨迹里。
  - 先按规则合成这一帧 action
  - 再把这条 action 真正在 robosuite 里执行一次
  - 执行后拿环境返回的 agent_pos
  - 把这个返回的 agent_pos 写进 zarr 作为这一帧 / 下一步轨迹的一部分
  - trans_sofar 也不再是累加 extra_step，而是 current_agent_pos[:3] - source_state[:3]
- 再编码回 controller pulse。
- 保存生成后的 `action / agent_pos / point_cloud`。
- 每条样本都带 `source_episode_idx`、`object_translation`、`motion_frame_count`。

对比：
- 旧线路 LiftPhaseCopy：phase copy + scale + 估计式 state 写回
- phase copy + replay_h1 source

template zarr：
- agent_pos / state 对 replay 对齐，因为是在转换为source阶段replay读取的，所以一定正确

欠缺：

- step_action 还要再 encode 回 pulse action，中间有量化、threshold、residual 误差。
- 后段误差：template zarr 里 replay 对齐的是 motion prefix。prefix 后把剩余段落（tail action，接近物体的阶段）直接接回去，并用 trans_sofar 去平移写 state / pcd。
point_cloud 也不是真重新渲染出来的，只是 source pcd 按 robot / object 的平移去改。
- 但 action 只是按当前编码规则合成出来，并经过 replay 写回后的动作，任务不一定成功。


---

### 4. solve

脚本：
- `scripts/export_lift_solved_from_template_zarr_relalign.py`

命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python scripts/export_lift_solved_from_template_zarr_relalign.py \
  --config repos/DemoGen/demo_generation/demo_generation/config/lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_selected4_d2467_diagfix.yaml \
  --template-zarr /media/willzhang/KINGSTON/zlxtemp/demogen_selected4_diagfix/data/datasets/generated/lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_selected4_d2467_diagfix_test_25.zarr \
  --source-demo /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo_selected4_d2467.hdf5 \
  --episodes all \
  --control-steps 1 \
  --action-deviation-weight 1e-4 \
  --relative-tail-steps 40 \
  --relative-cost-weight 4.0 \
  --output-zarr /media/willzhang/KINGSTON/zlxtemp/demogen_selected4_diagfix/outputs/generated/lift_0_v37_selected4_d2467_relalign_all_diagfix.zarr \
  --output-json /media/willzhang/KINGSTON/zlxtemp/demogen_selected4_diagfix/outputs/analysis/lift_0_v37_selected4_d2467_relalign_all_diagfix.json
```
内容：

把 prefix action 重新当成一个离散控制问题去重解
- 逐条读取 template zarr 里的：
  - `source_episode_idx`
  - `object_translation`
  - `motion_frame_count`

- 每一步会在 27 个候选 xyz pulse 里选一个最好的，候选就是 [-1, 0, 1]
- 优化问题：三项总代价：
  - 下一步末端世界坐标要贴近期望的平移后轨迹(agent_pos)
  - 在 prefix 末尾段，eef 相对 cube 的关系贴近 source demo
  - 不要离 source action 太远

- template 提供目标：
  - 前缀每一帧希望到达的末端轨迹
  - 抓取尾窗里希望满足的 eef 相对 cube 位置
    - 抓取尾窗：tail_start = solve_steps - relative_tail_steps（起点 = solve_steps - 40）
    当 step_idx < tail_start 时，相对位置约束权重是 0； 从 step_idx >= tail_start 开始加 eef 相对 cube 的约束。权重线性增加，到前缀最后一步最大
- solve 在 robosuite 里做前向模拟，先给候选 prefix action。再把整条 episode 在 robosuite 里 replay 一遍，把 replay 出来的真实：
  - `state`
  - `action`
  - `point_cloud`
  重写回 zarr。

 
---

### 5. gate


1. consistency
2. success

#### consistency
脚本：
- `scripts/validate_generated_zarr_consistency.py`
  - 用 zarr 里的 action 在 robosuite 里重放。
  - 对比 replay 出来的轨迹和 zarr 里保存的轨迹。
  - 检查这份 solved zarr 的 `state/action` 是否自洽。

#### success
脚本：
- `scripts/eval_generated_zarr_success_rate.py`
  - 直接在 robosuite 里评估 solved zarr 的任务成功率。

---
### 6. exportobs

脚本：
- `repos/DemoGen/real_world/export_demogen_zarr_to_robomimic_lowdim_replayobs.py`

命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python repos/DemoGen/real_world/export_demogen_zarr_to_robomimic_lowdim_replayobs.py \
  --generated-zarr /media/willzhang/KINGSTON/zlxtemp/demogen_selected4_diagfix/outputs/generated/lift_0_v37_selected4_d2467_relalign_all_diagfix.zarr \
  --source-low-dim-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/low_dim_selected4_d2467.hdf5 \
  --output-hdf5 /media/willzhang/KINGSTON/zlxtemp/demogen_selected4_diagfix/outputs/robomimic/lift_0_v37_selected4_d2467_relalign_all_diagfix_replayobs_lowdim.hdf5 \
  --include-source-demos \
  --control-steps 1 \
  --overwrite
```
内容：

以前export回去导出来的是近似平移出来的 object obs。
- 先拿 source lowdim 里的 object 观测。把其中 object_pos 整体加上一个常数 translation。object_quat 直接沿用 source
- 再和 generated 的 eef_pos 拼成新的 object obs


新版导出的 generated obs 是 replay 观测。
- 读取 solved zarr 的每条 episode。
- 用 `source episode + object_translation` 构造 reset state。
- 在 robosuite 里重新 replay 整条 generated action。
- 每一帧读取 low-dim 观测：
  - `robot0_eef_pos`
  - `robot0_eef_quat`
  - `robot0_gripper_qpos`
  - cube 的 `pos / quat`
- 再拼成 robomimic 的 `object`：
  - `object = [object_pos, object_quat, object_pos - eef_pos]`


最后得到 robomimic 标准训练格式：

- `data/demo_i/actions`
- `data/demo_i/obs/robot0_eef_pos`
- `data/demo_i/obs/robot0_eef_quat`
- `data/demo_i/obs/robot0_gripper_qpos`
- `data/demo_i/obs/object`

有趣的是：老pipeline + 新export同样适用， 8 rollout， Success_Rate = 1.0，显著提升

---
## Two-Stage 

* 双物体支线：
  * 两段 motion：`motion1`、`motion2`
  * 两段 skill：`skill1`、`skill2`
  * 两类相对约束：`eef-object`、`object-target`

---

### 1. 输入

- source demo：
- source low_dim：
- source depth
- parsing frames：
  - `skill-1`
  - `motion-2`
  - `skill-2`

配置：

```yaml
parsing_frames:
  motion-1: 0
  skill-1: [400, 550, 340, 415]
  motion-2: [500, 700, 470, 480]
  skill-2: [860, 960, 800, 860]
```

- `motion1 = [0, skill1)`
- `skill1 = [skill1, motion2)`
- `motion2 = [motion2, skill2)`
- `skill2 = [skill2, end)`

---

### 2. source

1. `raw hdf5 -> replayh1 light source zarr`
2. `light source zarr -> twostage schedule source zarr`

#### 2.1 replayh1 light source

脚本：

- `repos/DemoGen/real_world/convert_robomimic_hdf5_to_zarr_exec_replay_h1_light.py`

这一步和单物体线一致，source zarr 里字段：

- `data/action`
- `data/replay_h1_delta`
- `data/state`
- `data/point_cloud`

---

#### 2.2 twostage schedule source

脚本：

- `repos/DemoGen/real_world/convert_source_zarr_twophase_schedule_replay_h1.py`

内容：

- 读取上一步的 source zarr。
- 对每条 source episode，按 `skill1 / motion2 / skill2` 切出两段 motion：
  - `motion1 = [0, skill1)`
  - `motion2 = [motion2, skill2)`
- 分别对这两段重建 schedule。

```text
motion1 total_xyz = replay_h1_delta[0:skill1, :3].sum(axis=0)
motion2 total_xyz = replay_h1_delta[motion2:skill2, :3].sum(axis=0)
```

每一段都按 old one-stage 规则重分配：
- `z` 单独按 `z_step_size` 切固定小步。
- 剩下帧数给 `xy` 平均分。
- 最后顺序反过来：
  - 前面先走 `xy`
  - 后面几帧再走 `z`

写回：

- `motion1` 段重写 `data/motion_action[:skill1]`
- `motion2` 段重写 `data/motion_action[motion2:skill2]`
- `skill1 / skill2` 段保持 parent source zarr 的 `motion_action`
- `data/action` 仍然保持 source demo 的 pulse action


---

### 3. generate

配置文件：

- `repos/DemoGen/demo_generation/demo_generation/config/nutassemblyround_0_v1_replayh1_twostage_selected4.yaml`

source 命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python scripts/run_twostage_raw_hdf5_pipeline.py \
  --config repos/DemoGen/demo_generation/demo_generation/config/nutassemblyround_0_v1_replayh1_twostage_selected4.yaml \
  --raw-demo-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/demo.hdf5 \
  --selected-episodes 0,1,2,3 \
  --data-root /home/willzhang/Science/Reproduction/Reproduction/data \
  --selected-demo-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/demo_selected4.hdf5 \
  --low-dim-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/low_dim_selected4.hdf5 \
  --depth-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/depth_selected4.hdf5 \
  --overwrite
```

generate 命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/demo_generation

conda run -n demogen python -W ignore gen_demo.py \
  --config-name=nutassemblyround_0_v1_replayh1_twostage_selected4.yaml \
  data_root=/home/willzhang/Science/Reproduction/Reproduction/data \
  source_name=nutassemblyround_0_v1_replayh1_twostage_source_selected4 \
  source_demo_hdf5=/home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/demo_selected4.hdf5 \
  generated_name=nutassemblyround_0_v1_replayh1_twostage_selected4 \
  generation.range_name=test \
  generation.mode=grid \
  generation.n_gen_per_source=25 \
  generation.render_video=False
```

内容：

- 生成器： `StackPhaseCopyReplayConsistentDemoGen`。
- 沿用 `stack` fork 名字，是通用的双物体 twostage fork。
- 每条 generated sample 都先构造 reset state，再在 robosuite 里一步一步执行。

双物体线里的字段名：

- `object_translation`
  - 在 twostage template 里仍然沿用这个字段名。
  - 但它存的其实是双物体平移信息，语义是 6 维：`[object_xyz, target_xyz]`。
  -  `nutassemblyround` 配置里，后 3 维固定是 `0`。
- `motion_frame_count`
  - 在 twostage 里沿用老名字。
  - 但它实际存的是 `skill1_frame`，也就是第一段 motion 的结束帧。
  - 第二段边界单独写在 `motion_2_frame` 和 `skill_2_frame`。

generate 时的四段处理是：

#### motion1

- reset 时先按任务定义把 object / target 平移到新位置。
- 对 `motion1 = [0, skill1)`：
  - 从 source 里读 `motion_action`
  - 给它叠加一份 `object_translation` 的逐帧增量
  - 再编码回 executable pulse action
- point cloud 渲染时：
  - object 平移 `obj_trans_vec`
  - target 平移 `tar_trans_vec`
  - robot 按 replay 出来的 `robot_trans_vec` 平移

#### skill1

- `skill1 = [skill1, motion2)` 直接 copy source executable action。
- 这一段不再额外加 motion schedule。
- point cloud 语义变成：
  - target 单独按 `tar_trans_vec` 平移
  - object 和 robot 跟随当前 replay 的 `robot_trans_vec`

#### motion2

- `motion2 = [motion2, skill2)` 再做一次 motion 增广。
- 这次加：`target_translation - object_translation`，也就是补“从当前 object 偏移，走到 target 偏移”这段相对位移。
- point cloud 语义和 `skill1` 一样：
  - target 单独按 `tar_trans_vec`
  - object 和 robot 跟随 replay 的 `robot_trans_vec`

#### skill2

- `skill2 = [skill2, end)` 继续 copy source executable action。
- point cloud 直接整体按 `tar_trans_vec` 平移。

template zarr 里写回这些 meta：

- `source_episode_idx`
- `object_translation`
- `motion_frame_count`
- `motion_2_frame`
- `skill_2_frame`

template ：

- `state/action/point_cloud` 都是按 robosuite replay 过程写出来的。
- 但两段 motion 的 action 仍然只是 template action，不保证任务最终成功。

---

### 4. solve

脚本：

- `scripts/export_stack_solved_from_template_zarr_relalign_twostage.py`

命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python scripts/export_stack_solved_from_template_zarr_relalign_twostage.py \
  --config repos/DemoGen/demo_generation/demo_generation/config/nutassemblyround_0_v1_replayh1_twostage_selected4.yaml \
  --data-root /home/willzhang/Science/Reproduction/Reproduction/data \
  --source-demo /home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/demo_selected4.hdf5 \
  --template-zarr /home/willzhang/Science/Reproduction/Reproduction/data/datasets/generated/nutassemblyround_0_v1_replayh1_twostage_selected4_test_25.zarr \
  --episodes all \
  --control-steps 1 \
  --action-deviation-weight 1e-4 \
  --motion1-relative-tail-steps 40 \
  --motion1-relative-cost-weight 4.0 \
  --motion2-relative-tail-steps 40 \
  --motion2-relative-cost-weight 4.0 \
  --output-zarr /home/willzhang/Science/Reproduction/Reproduction/outputs/generated/nutassemblyround_0_v1_replayh1_twostage_selected4_relalign_twostage_all.zarr \
  --output-json /home/willzhang/Science/Reproduction/Reproduction/outputs/analysis/nutassemblyround_0_v1_replayh1_twostage_selected4_relalign_twostage_all.json
```

内容：
- 逐条读取 template zarr 里的：
  - `source_episode_idx`
  - `object_translation`
  - `motion_frame_count`
  - `motion_2_frame`
  - `skill_2_frame`
- 然后把 `object_translation` 先拆开：
  - `object_translation`
  - `target_translation`

双物体 2 次 solve 完：

#### 4.1 solve motion1

- `motion1` 的目标轨迹来自：
  - source `state[:3]`
  - 再加上 `object_translation` 在 `[0, skill1)` 上的累积增量
- 候选 action 还是在 robosuite 里逐步前向模拟选出来。
- cost 里有三项：
  - `eef xyz` 对 template 目标的误差
  - `eef-object relative xyz` 对 source reference 的误差
  - 偏离 template action 的代价
- 其中 `eef-object` 约束只在尾窗里逐渐加权：
  - `tail_start = solve_steps - motion1_relative_tail_steps`
  - 越靠近第一段 motion 末尾，relative cost 权重越大

即：
- 前面主要先把末端几何轨迹拉回去
- 到临近抓取时，再把 `eef` 相对 object 的关系收紧

#### 4.2 solve motion2

- 先把 `motion1` solve 好的 action 当作 prefix replay 到 `motion2` 起点。
- `motion2` 的目标轨迹来自：
  - source 在 `[motion2, skill2)` 上的 `state[:3]`
  - 再加上当前 object offset
- 这里的 `current_object_offset` 不是常数，而是：

```text
object_translation
+ cumulative(target_translation - object_translation)
```

- 第二段 cost 里的 relative 约束换成：
  - `object-target relative xyz`
- 同样只在尾窗里逐渐加权：
  - `tail_start = solve_steps - motion2_relative_tail_steps`

这里的意思是：

- 第一段解决“抓到 object”
- 第二段解决“把 object 带到 target 上”

#### 4.3 replay 写回

- 两段 solve 完之后，把整条 full action 再在 robosuite 里 replay 一遍。
- 最后把 replay 出来的真实：
  - `state`
  - `action`
  - `point_cloud`
  重写回 zarr。

---

### 5. gate

---

### 6. exportobs

脚本：

- `repos/DemoGen/real_world/export_demogen_zarr_to_robomimic_lowdim_replayobs_twophase.py`

命令：

```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python repos/DemoGen/real_world/export_demogen_zarr_to_robomimic_lowdim_replayobs_twophase.py \
  --generated-zarr /home/willzhang/Science/Reproduction/Reproduction/outputs/generated/nutassemblyround_0_v1_replayh1_twostage_selected4_relalign_twostage_all.zarr \
  --source-low-dim-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/nutassemblyround_0/1776068238_0811434/low_dim_selected4.hdf5 \
  --output-hdf5 /home/willzhang/Science/Reproduction/Reproduction/outputs/robomimic/nutassemblyround_0_v1_replayh1_twostage_selected4_relalign_twostage_replayobs_lowdim.hdf5 \
  --include-source-demos \
  --control-steps 1 \
  --overwrite
```

内容：

- 读取 solved zarr 的每条 episode。
- 先把 `object_translation` 拆成：
  - `object_translation`
  - `target_translation`
- 用 `source episode + translation` 构造 reset state。
- 在 robosuite 里重新 replay 整条 generated action。
- 每一帧读取 low-dim 观测：
  - `robot0_eef_pos`
  - `robot0_eef_quat`
  - `robot0_gripper_qpos`
  - `object`

这里双物体线和单物体线的区别是：

- reset 不再只是平移一个 object。
- replayobs exporter 会按任务定义把 object / target translation 都应用到 reset state。
- 导出的 `obs/object` 是双物体 reset 下 replay 出来的真值。

最后得到 robomimic 标准训练格式：

- `data/demo_i/actions`
- `data/demo_i/obs/robot0_eef_pos`
- `data/demo_i/obs/robot0_eef_quat`
- `data/demo_i/obs/robot0_gripper_qpos`
- `data/demo_i/obs/object`

---

## Eval
结合仓库默认配置与论文原文，eval为3*seed 随机roll 每seed20，这里使用相同配置

项目页写： “DemoGen 的有效性已在 8 个改进的 MetaWorld 任务上得到验证，这些任务扩大了对象随机化范围。我们报告了在DemoGen生成的数据集上训练的视觉运动策略在 3 个种子点上的最大/平均成功率，每个任务仅使用一个 源演示。结果表明，DemoGen能够在数据收集所需人力减少 20 倍以上的情况下，保持策略的性能。”泛化结论来自未见配置上的测试，不仅限于 source demo 的初始场景。


default-reset：
- 评估时走 `env.reset()`，回到环境自己的 canonical 初始场景 / 初始布局。和训练数据里的 source-demo 初始分布不一致。

新增 custom-reset，用来检测模型有没有学到训练集里的特征：
- 评估时用 `source demo` 的orientation，并且在训练集的总范围内采样，重建初始场景，有平移时再叠加对应 translation。


### One-Stage -- Lift / Press Handle
#### Lift Cube
随机3 Seed / 20 rollout 测试中
- 4-100 相比原文 100 取得 65% / 成功率；单seed roll取得最高 80% 成功率
- 9-153取得 95% / 91.6% 成功率

#### Presss Handle
- 4-100 训中70pth roll*10  成功率100% 
- 3*seed 20roll 成功率100/100，和原文相符


### TwoStage -- Stack Cube Task / NutAssembly Task
本质上是双物体的Pick & Place
- stack任务采取1-81的配置（移动object&target的位置）
- nut任务采取4-100的配置（仅移动object位置）

Stack & Nut 在default上均成功率很低，只有0.1-0.2：
- Default Reset上仅能实现0.1左右成功率
- Custom Reset 上实现0.9成功率，说明对训练集内涵盖、中间情况等内容分布较好

结论：说明lift和press实现了一定程度的泛化 而 stack/nut 在目前配置下位置泛化能力存在，一旦改变位姿朝向，姿态泛化就几乎为0，正在排查原因
- 直到export回robomimic的hdf5回放也是正常的，并且能够取得分部内的成功，说明数据导出不是最主要原因

### 疑惑
条件铺垫：
- Demogen的数据增广仅涵盖平移，未涵盖xy内位姿的变化
- 在做One-StageLift任务时，数据增广量从4-100跃升至9-153时，成功率从65%升至95%。
- Press这类简单任务（Door的姿态不变化，仅位置变化），4-100足以cover100%成功率
- 在录制Demo/default-reset的配置下均为随机初始化
这说明原始情况下位姿的变化能够显著帮助模型训练效果的提升

疑惑：如果说DP对分布外的场景的泛化能力来自于插值（待验证），那么为什么原文的效果就是比我们的好？

### 分析误差来源

1. 原文 source demo 是 scripted，本repo是键盘录的 demo。
- 原文是 scripted policies，准备 1 条 source demo，再生成 100 / 200 条 synthetic demos；10/25 source 的 human-collected 对照也是拿 scripted policy 当参考采的。

- 本地是人手 demo，再手工切 motion/skill。scripted policy 的 demo 更平滑、分段清楚，没有犹豫、抖动、补救动作。

2. 原文双物体 200 不是 grid
two-object generator，如果是 grid 里动，n_gen_per_source 必须是四次方。论文：双物体任务生成 200 条 demos，不是四次方，推测为 random sampling 生成器。理论上可以覆盖更加多的位置分布。（待求证）

3. convert*2-generate-solve-export 的累积误差

4. 原文数据量更大
 
5. 把 4 条坏 solve 删掉，效果反而更差，说明主因是上述全局的因素


---

# Pipeline Old

- `robosuite`
  - 仿真与采集
  - 建环境、控制机器人、采集 demo、回放 demo、导出原始 `demo.hdf5`。
- `robomimic`
  - 数据集转换、训练与评测
- `DemoGen`：轨迹和观测数据的数据增广

- `robomimic`
  - 训练 policy、做 rollout 、评测。

---

## Setup

工作区搭建

```
/home/willzhang/Science/Reproduction/Reproduction
├── repos
│   ├── DemoGen
│   ├── robosuite
│   └── robomimic
├── data
│   ├── raw
│   ├── processed
│   └── generated
├── scripts    分析、验证、绘图、后处理等辅助脚本
│     
├── outputs    可视化与训练权重
│     
└── configs    本地训练配置，给 robomimic 使用
        
```

---

### 采 9 条 robosuite demo

```bash
conda activate robosuite
 # Mujoco键位
    # ↑ / ↓ 前后移末端
    # ← / → 左右移末端
    # .     向下
    # ;     向上
    # o / p yaw
    # y / h pitch
    # e / r roll
    # Space 开合夹爪，按一次切换一次

# 9 Episode
# 20 fps
python ~/Science/Reproduction/Reproduction/repos/robosuite/robosuite/scripts/collect_human_demonstrations.py \
  --environment Lift \
  --robots Panda \
  --device keyboard \
  --renderer mjviewer \
  --camera agentview \
  --directory /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0
```


生成数据集：`/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5`
```
data/demo_1/action_dict/gripper             shape=(310, 1)   dtype=float32
data/demo_1/action_dict/rel_pos             shape=(310, 3)   dtype=float32
data/demo_1/action_dict/rel_rot_6d          shape=(310, 6)   dtype=float32
data/demo_1/action_dict/rel_rot_axis_angle  shape=(310, 3)   dtype=float32
data/demo_1/actions                         shape=(310, 7)   dtype=float64
data/demo_1/states                          shape=(310, 32)  dtype=float64
```

- 310：Demo的时间长度
- `actions` (7)：3 维末端位移 `dx, dy, dz` + 3 维末端旋转增量 + 1 维夹爪开合。
- `states` (32)：MuJoCo 展平状态， `time + qpos + qvel`，主要给仿真回放和状态恢复用，不是最适合直接拿来训练的观测。
- `gripper` (1)：夹爪开 / 合。
- `rel_pos` (3)：末端相对位移 `dx, dy, dz`。
- `rel_rot_axis_angle` (3)：末端相对旋转，用 axis-angle 表示。
- `rel_rot_6d` (6)：同一个旋转的 6D 表示。

**回放**

```bash
python robosuite/scripts/playback_demonstrations_from_hdf5.py \
  --folder  /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063
```
---
### 转成 robomimic 数据集
`demo.hdf5` 是 `robosuite` 的原始轨迹格式， `robomimic` 中用带有 `obs / next_obs `的内容的格式训练
- `low_dim.hdf5`：位置、姿态、夹爪、物体状态等。
- `image.hdf5`：相机图像。

#### **convert**
* 读取 demo.hdf5 里的环境元信息和轨迹
* 按 robomimic 的约定补齐 / 整理 metadata
* 给数据集加上 mask/train、mask/valid 
  
```bash
conda activate robomimic

python repos/robomimic/robomimic/scripts/conversion/convert_robosuite.py \
  --dataset /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5
```
#### **low-dim**

```bash
cd /home/willzhang/Science/Reproduction/Reproduction/repos/robomimic/robomimic/scripts

python repos/robomimic/robomimic/scripts/dataset_states_to_obs.py \
  --dataset /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5 \
  --output_name low_dim.hdf5 \
  --done_mode 2
```
`/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/low_dim.hdf5`

- `object` (10)：方块位置(3) + 方块四元数(4) + gripper 到方块的相对位置(3)
- `robot0_eef_pos` (3)：机械臂末端位置 [x, y, z]。
- `robot0_eef_quat` (4)：机械臂末端姿态四元数。
- `robot0_gripper_qpos` (2)：左右夹爪手指关节位置。
- `robot0_joint_pos` (7)：Panda 7 个关节角。
- `robot0_joint_vel` (7)：Panda 7 个关节速度。

#### **image**

```bash
python repos/robomimic/robomimic/scripts/dataset_states_to_obs.py \
  --dataset /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5  \
  --output_name image.hdf5 \
  --done_mode 2 \
  --camera_names agentview \
  --camera_height 84 \
  --camera_width 84 \
  --compress \
  --exclude-next-obs
```

`/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/image.hdf5` ：

```text
obs/agentview_image            (310, 84, 84, 3)   uint8
obs/robot0_eye_in_hand_image   (310, 84, 84, 3)   uint8
obs/object                     (310, 10)
obs/robot0_eef_pos             (310, 3)
obs/robot0_eef_quat            (310, 4)
obs/robot0_gripper_qpos        (310, 2)
actions                        (310, 7)
rewards                        (310,)
dones                          (310,)
states                         (310, 32)
```

- `agentview_image` (84, 84, 3)：主相机[H, W, C]。
- `robot0_eye_in_hand_image` (84, 84, 3)：局部相机 [H, W, C]。

#### **depth**

```bash
python repos/robomimic/robomimic/scripts/dataset_states_to_obs.py \
  --dataset /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5 \
  --output_name depth.hdf5 \
  --done_mode 2 \
  --camera_names agentview \
  --camera_height 84 \
  --camera_width 84 \
  --depth
```
`/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/depth.hdf5`

#### 对robomimic数据集做一个可视化
```bash
python repos/robomimic/robomimic/scripts/visualize_robot_dataset.py \
  --dataset /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/image.hdf5 \
  --output-dir outputs/dataset_viz/your_run
```

![alt text](./Files/imgs/image.png)

![alt text](./Files/imgs/image-1.png)

---
## 数据增广

### 当前 pipeline

1. 采 source demo，得到 `demo.hdf5 / low_dim.hdf5 / depth.hdf5`
2. 用 一系列 convert 生成 `source zarr`
3. 生成`generated zarr`
4. 对 generated `zarr` 做 replay，检验是否正常
5. 跑 consistency validation / success rate
6. 前面都过了，进入训练 / eval

### 为什么要转 `zarr`
`demogen.py`输入： 
* `action`：`data/action`，单条 310 帧 demo 时 shape = `(310, 7)`
  7 = 3 维平移控制 `dx, dy, dz` + 3 维旋转控制 + 1 维夹爪控制
* `agent_pos`：`data/agent_pos`，单条 310 帧 demo 时 shape = `(310, 7)`
  7 = 3 维末端位置 `eef_pos` + 3 维姿态 `rotvec` + 1 维夹爪开口 `gripper_gap`
* `point_cloud`：`data/point_cloud`，单条 310 帧 demo 时 shape = `(310, 1024, 6)`
  每一帧采样 1024 个点`[x, y, z, r, g, b]`

将`hdf5`转换成 `source zarr`，要求：
* generate 出来 replay 轨迹正常 
* 点云视频正常


### 从`hdf5`到`zarr`

#### 问题

Lift 的 raw `action[:3]` 是来自 controller 的 pulse，demogen motion 段的增广需要逐帧的几何位移，如果像这样在pulse空间做增广，generate出来会变形。

#### 为什么要加 motion_action 修正？
DemoGen 在 motion 段思路是：
* 先生成一条新轨迹：根据 source 轨迹的起点、终点和新物体位置，构造 motion 段每一帧的 step_action 决定新 demo 动作
* 因为点云和state需要和轨迹同步，所以需要逐帧计算轨迹偏差： step_action - source_action[:3]，得到这一帧相对 source 的增量差
* 把这个差累积成 trans_sofar，用 trans_sofar 去平移 robot 的 state 和 point_cloud

![alt text](./Files/imgs/image-2.png)

但是`step_action` 是几何位移，`raw_action[:3]` 是控制脉冲，两个量不在同一语义空间。这会导致平移出来的state有误差

#### 怎么做？

* source action 首先被转换为一份近似几何运动参考 motion_action，这一步仍然带有近似误差；
* DemoGen 基于这份近似几何量，在 motion 段中计算相对 source 的额外平移增量，这样的语义比直接对 raw pulse 做减法求得平移量更稳定；
* 这部分几何增量再通过 pulse encoder 映射回 controller action，并叠加到原始 executable pulse 上；
* 最终写回 generated zarr 的 data/action 仍然是 pulse 语义的 executable action。


![alt text](./Files/imgs/image-3.png)

#### **motion_action 怎么来的？**

1.  通过 `convert_robomimic_hdf5_to_zarr_exec_xzfullfir4sum.py` 得到 base source zarr。得到一版几何接口。**这版误差体现在：如果直接用这个去做generate，出来的物体位置不算准，伸的深度不够**

如何构造一版几何的 `motion_action`：
  - x / z：用 full FIR 响应把 pulse 映射成几何效果
  - y：直接用 `forward_delta[:, 1]`
  - gripper：沿用 `action[:, 6]`


```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python repos/DemoGen/real_world/convert_robomimic_hdf5_to_zarr_exec_xzfullfir4sum.py \
  --demo-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5 \
  --low-dim-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/low_dim.hdf5 \
  --depth-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/depth.hdf5 \
  --source-name lift_0_v9_execmotion_xzfullfir4sum \
  --output-zarr /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/datasets/source/lift_0_v9_execmotion_xzfullfir4sum.zarr
```

2. 用 `convert_source_zarr_original_schedule_motion.py` 处理 source zarr，读取`state`/ `action`/ `motion_action`/`skill1_frame`，然后重写 motion 段的 `motion_action[:3]`，调用 `build_original_one_stage_schedule(...)`，应用和demogen相同的原则，假设平移量是0，会生成什么样的轨迹？这个轨迹之后在demogen中直接做减法求得平移量：
  - 取 source 轨迹在 motion 段的起点 `state[0, :3]`
  - 取终点 `state[skill1_frame - 1, :3]`
  - 把 z 方向拆成固定步长，再把剩余帧平均分给 xy，翻转后得到先走 xy 下探 z的schedule

写回motion_action：

- `new_motion[start : start + ep_skill1, :3] = schedule_xyz`上面生成的schedule
- `new_motion[start : start + ep_skill1, 6] = action[start : start + ep_skill1, 6]`，夹爪沿用相同维度的动作


```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python repos/DemoGen/real_world/convert_source_zarr_original_schedule_motion.py \
  --input-zarr /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/datasets/source/lift_0_v9_execmotion_xzfullfir4sum.zarr \
  --output-zarr /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/datasets/source/lift_0_v21_originalschedule_motion_v9_s220.zarr \
  --source-name lift_0_v21_originalschedule_motion_v9_s220 \
  --skill1-frame 190 \
  --z-step-size 0.015 \
  --copy-sam-mask
```

#### **motion_action 在 generate 里怎么被用？**
`demogen_lift_phase_copy.py`：外挂一个模块，override了demogen.py的one_stage_augment，读取和保存还用demogen完成，只是在这里进行增广

输入：
- `state`/`pcd`
- `source_exec_action`：controller action，来自 `data/action`
- `source_motion_action`：给 DemoGen 参考的几何 motion，来自 `data/motion_action`。

每一帧：
- 先按 schedule 把总平移拆成每帧的 translation_increments （这次 retarget 的增广平移量）， `extra_step = translation_increments[j]`。
- 再构造这一帧理论上想实现的几何 motion：
  - `step_action = source_motion_action[:3] + extra_step`
- 然后回到 `demogen.py`中增加的`source_plus_correction`：
  - `correction_step = step_action - source_motion_action[:3]`。表示：新目标下这一帧理论 motion，减去 source 原本这一帧的参考 motion，还差多少
  - `correction_xyz = _encode_motion_exec_xyz(correction_step, ...)`。差值再经过 pulse 编码，变成加回 controller 的 correction
  - `exec_xyz = source_exec_action[:3] + correction_xyz`，映射回 pulse 空间

#### Notice
1. motion_action本身存在误差，需要`translation_correction_scale_xyz = [0.5, 0.5, 1.0]`用来修正extrastep
- xy correction 只加0.5，z加1倍。用来抑制接近物体和下探阶段易出现的横向过修正。当前pipeline里 xy 更容易因为 pulse 量化和 schedule 近似被放大，所以只修一半，但是针对不同任务可能需要调整，故此为一优化方向

2. state_based 和原版 legacy 的区别 
- `legacy` 把 source 的 motion 起点近似写成 `state[start] - action[start]`，默认认为 `action[:3]` 本身就接近逐帧 motion；同时它按 `trans_this_frame = step_action - source_motion_action[:3]` 去逐帧累积 `trans_sofar`；
- `state_based` 把 `state[start][:3]` 当作 motion 起点；后续维护 `desired_pos += step_action`，再用 `trans_sofar = desired_pos - source_pos`， 比 `legacy` 更少依赖 `action[:3]` 本身的几何语义
- 这更适合 pulse ：因为 controller pulse 不是逐帧几何位移，用 `state` 比用 `action` 反推轨迹更稳

3. 配置文件在：`lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220.yaml` 

### 做增广
```bash 
cd /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/demo_generation

bash gen_demo.sh lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220 test grid 4 False

```

其中：
- `bash gen_demo.sh lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220 ...`
- `gen_demo.sh` 调用 `gen_demo.py`读取 yaml，并根据其中的 `_target_` 实例化 generator，指向 `demogen_lift_phase_copy.py`
- 每条增广16条
### 增广数据可视化验证

```bash
conda run -n demogen python /home/willzhang/Science/Reproduction/Reproduction/scripts/replay_zarr_episode.py \
  --zarr /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/datasets/generated/lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_16.zarr \
  --source-demo /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5 \
  --episode 0 \
  --control-steps 1 \
  --output-video /home/willzhang/Science/Reproduction/Reproduction/videos/replay_lift_0_v28_test16_ep0.mp4
```

### 增广数据 Gate

用 generated zarr 里的 action 在 robosuite 里重放一遍，i再将 replay 出来的 agent_pos（state）和 zarr 里原本存的 agent_pos 逐帧对比末端位置 xyz。计算轨迹整体的 RMSE 和最后一帧的 final error。目的是看这条轨迹的state和action是否自洽。同时还附带脚本检测增广数据中抓取是否成功。

```bash
conda run -n demogen python /home/willzhang/Science/Reproduction/Reproduction/scripts/validate_generated_zarr_consistency.py \
  --zarr /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/datasets/generated/lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_16.zarr \
  --source-demo /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/demo.hdf5 \
  --control-steps 1 \
  --rmse-threshold 0.015 \
  --final-threshold 0.015 \
  --output-json /home/willzhang/Science/Reproduction/Reproduction/outputs/analysis/lift_0_v28_test16_consistency.json
```
结果：144条全部Lift成功并且通过gate测试，说明该pipeline有效

### 失误 
思维惯性用demogen里面的dp3去训练，但是由于前述原因“最终写回 generated zarr 的 data/action 仍然是 pulse 语义的 action”。导致学到的仍是 pulse-like action distribution，这样的动作标签里有很多接近 0 的帧，夹杂少量离散脉冲，再加上 encode / threshold / residual 的量化效应，训练会更难、更不稳定。

通过fork许多版本的dp3训练Pipeline达到了近似的效果，能够取得非常不稳定的成功，如：

/home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/ckpts/lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9-dp3phasebias-seed0-pb_a/checkpoints/79.ckpt

- `79.ckpt` 对应 `phasebias v1`训练线。它修改了 sampler：把 pre-grasp descent 和 gripper switch 附近的 window 重复采样，让 DP3 更频繁看到下探和闭合片段。
```bash
cd /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/diffusion_policies

SOURCE_DATASET=/home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_keyboard_1/1774355871_95818/demo.hdf5 \
EVAL_EPISODES=3 \
SAVE_VIDEO=True \
bash eval_panda_phasebias.sh \
  lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9 \
  0 \
  79 \
  1 \
  pb_a
```

### 三层误差
* source_motion_action 本身就只是近似几何 proxy，不是真实物理真值。
* extra_step 也是按 schedule 分配出来的增广位移。
* 最后把 correction 再 encode 回 pulse，这里又会有 threshold / residual / 饱和误差。
---

## 在 robomimic 上训练 DP
路线纠正过后，直接把 `lift_0` 的 generated zarr 导回 robomimic，再用 diffusion policy 训练。

###  导出 `lift_0` 训练集

用
- generated zarr（共153条数据）：`lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_16.zarr`
- source low-dim：`data/raw/lift_0/1774702988_8036063/low_dim.hdf5`
- 输出：`data/processed/robomimic/lift_0_v28_demogen_lowdim.hdf5`

```bash
cd /home/willzhang/Science/Reproduction/Reproduction

conda run -n demogen python repos/DemoGen/real_world/export_demogen_zarr_to_robomimic_lowdim.py \
  --generated-zarr /home/willzhang/Science/Reproduction/Reproduction/repos/DemoGen/data/datasets/generated/lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_16.zarr \
  --source-low-dim-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/raw/lift_0/1774702988_8036063/low_dim.hdf5 \
  --output-hdf5 /home/willzhang/Science/Reproduction/Reproduction/data/processed/robomimic/lift_0_v28_demogen_lowdim.hdf5 \
  --include-source-demos \
  --overwrite
```

### 启动训练

```bash
ROOT=/home/willzhang/Science/Reproduction/Reproduction
RUN_NAME=lift_0_v28_demogen_dp_$(date +%Y%m%d_%H%M%S)_1

conda run -n robomimic python $ROOT/repos/robomimic/robomimic/scripts/train.py \
  --config $ROOT/configs/robomimic/diffusion_policy_lift_demogen_lowdim.json \
  --dataset $ROOT/data/processed/robomimic/lift_0_v28_demogen_lowdim.hdf5 \
  --name "$RUN_NAME" \
  > /tmp/${RUN_NAME}.stdout 2>&1 &

sleep 3

# 这里补充了训练进度可视化

RUN_ROOT=$(find "$ROOT/outputs/robomimic/diffusion_policy_demogen" -maxdepth 1 -type d -name "$RUN_NAME" | sort | tail -n 1)
RUN_DIR=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

tail -f "$RUN_DIR/logs/log.txt"
```

tensorboard：
```bash
conda run -n robomimic tensorboard \
  --logdir "$RUN_DIR/logs" \
  --bind_all --port 6006
```

配置文件 `diffusion_policy_lift_demogen_lowdim.json`:
- `train.num_epochs = 1200`
- `save.every_n_epochs = 150`
- `rollout.rate = 70`
- `rollout.n = 10`
- `rollout.horizon = 800`
- `always_save_latest = false`
- `checkpoint_state_mode = policy_only`



### Eval checkpoint

只看成功率：

```bash
conda run -n robomimic python /home/willzhang/Science/Reproduction/Reproduction/repos/robomimic/robomimic/scripts/run_trained_agent.py \
  --agent /home/willzhang/Science/Reproduction/Reproduction/outputs/robomimic/diffusion_policy_demogen/lift_0_v28_demogen_dp_20260328_230323_1/20260328230325/models/model_epoch_150.pth \
  --n_rollouts 50 \
  --horizon 800 \
  --seed 1
```

- `eval_epoch150_rollout100` 成功率 `0.73`
- `eval_epoch200_rollout100` 成功率 `0.56`
- `eval_epoch300_rollout100` 成功率 `0.48`

结论：153条数据在batch_size=80的配置下，在 150-200 epoch间达到eval最佳成功率。
