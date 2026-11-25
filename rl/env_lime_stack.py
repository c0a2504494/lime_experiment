"""Task 2 of the decomposed curriculum: STACK learning (start already holding cube A).

Episode starts with cube A grasped between the pads (canonical carry pose + noise),
cube B randomly placed on the table. Success: A stacked on B (same criteria as the
original env).

Initialization method:
  1. arm -> canonical carry pose (validated: home pose cradle carry, joint noise added)
  2. cube A teleported between the pads at the measured grasp offset
  3. gripper close command + short settle so the cube seats in the grasp
  4. episodes where the cube drops during init are still usable — the policy just
     sees a failure start (rare; measured in smoke test).

Known simplification (documented in AGENTS.md): the initial pose distribution is
canonical+noise, NOT the true terminal distribution of the grasp policy. Once task 1
is trained, record its terminal states and use them here for distribution matching.
"""
import numpy as np
import mujoco

from env_lime_cube import LimeCubeStackEnv
from math import radians

# canonical carry pose (validated cradle carry: cube held at z~0.14)
CARRY_POSE = np.array([0.0, radians(23), radians(80), 0.0, -radians(14), 0.0])


class LimeStackEnv(LimeCubeStackEnv):
    def __init__(self, scene_xml_path, max_steps=100, pose_noise=0.05, **kw):
        super().__init__(scene_xml_path, max_steps=max_steps, **kw)
        self.pose_noise = pose_noise
        self._gl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_link")
        self._gr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_link")

    def reset(self, seed=None, options=None):
        # randomize cube B the same way as parent, but NOT cube A (it's in the gripper)
        gym_obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random

        # arm to noisy carry pose
        noise = rng.uniform(-self.pose_noise, self.pose_noise, size=6)
        qpose = np.clip(CARRY_POSE + noise, self.arm_qmin, self.arm_qmax)
        self.data.qpos[self.arm_qpos_adr] = qpose
        self.data.ctrl[self.arm_act_idx] = qpose
        mujoco.mj_forward(self.model, self.data)

        # teleport cube A between the pads: at padmid, dropped by the finger offset.
        # validated grasp geometry: cube center ~0.032 below padmid, slight xy noise.
        pm = (self.data.xpos[self._gl] + self.data.xpos[self._gr]) / 2
        jitter = rng.uniform(-0.003, 0.003, size=2)
        self.data.qpos[self.cube_a_qpos:self.cube_a_qpos + 7] = [
            pm[0] + jitter[0], pm[1] + jitter[1], pm[2] - 0.032, 1, 0, 0, 0,
        ]
        self.data.qvel[self.cube_a_dof:self.cube_a_dof + 6] = 0
        # close gripper and settle so the cube seats
        self.data.ctrl[self.grip_act_idx] = -0.010
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # desired goal: cube A stacked on cube B
        cb_pos = self.data.qpos[self.cube_b_qpos:self.cube_b_qpos + 3].copy()
        self._desired_goal = cb_pos + np.array([0, 0, self.cube_size])
        self._step_count = 0
        return self.make_obs(), info

    def held(self):
        """debug helper: is cube A plausibly held (well above table)?"""
        return bool(self.data.qpos[self.cube_a_qpos + 2] > 0.05)
