"""Task 2 of the decomposed curriculum: STACK learning (start already holding cube A).

Episode starts with cube A grasped in the validated cradle-carry configuration,
cube B randomly placed on the table. Success: A stacked on B (same criteria as the
original env).

Initialization method (reproduces the validated physics path — do NOT shortcut):
  1. arm -> canonical vertical pinch pose GRASP_POSE (extracted from validate_pinch
     on the integrated scene) + small noise
  2. cube A teleported into the vertical grasp (padmid-relative, measured offset)
  3. gripper close + settle so friction seating occurs exactly as in validation
  4. joint-space interpolation to noisy carry pose over 120 steps — the cradle
     transition that lifts the cube to z~0.14 (validated in validate_stack)
  Directly teleporting the cube into the cradle pose does NOT work (cube falls out
  or gets squeezed out — measured 0/5 held in smoke test).

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

# canonical vertical pinch grasp pose over cube at (0.17, 0), extracted by running
# the validate_pinch IK sequence on the integrated scene.xml (2026-11-25):
# cube held after close, carry transition lifts to z 0.140.
GRASP_POSE = np.array([0.0, 0.4569, 1.2681, 0.0001, 1.4166, 0.0])

# measured padmid-to-cube-center offset at the moment of the closed grasp
GRASP_OFFSET_Z = 0.0217


class LimeStackEnv(LimeCubeStackEnv):
    def __init__(self, scene_xml_path, max_steps=100, pose_noise=0.05, **kw):
        super().__init__(scene_xml_path, max_steps=max_steps, **kw)
        self.pose_noise = pose_noise
        self._gl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_link")
        self._gr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_link")

    def _padmid(self):
        return (self.data.xpos[self._gl] + self.data.xpos[self._gr]) / 2

    def reset(self, seed=None, options=None):
        # randomize cube B the same way as parent; cube A gets overridden below
        gym_obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random

        # measured 2026-11-25: the parent's random cube B can land on/near the canonical
        # grasp point (0.17, 0) — seeds with dist < 45mm ALL failed init (B collides with
        # A during the close). Resample B until clear of the grasp region.
        half = self.cube_size / 2.0
        while True:
            b = self.data.qpos[self.cube_b_qpos:self.cube_b_qpos + 2]
            if np.hypot(b[0] - 0.17, b[1] - 0.0) > 0.07:
                break
            bx = rng.uniform(0.13, 0.22)
            by = rng.uniform(-0.12, 0.12)
            self.data.qpos[self.cube_b_qpos:self.cube_b_qpos + 7] = [bx, by, half + 0.001, 1, 0, 0, 0]
            self.data.qvel[self.cube_b_dof:self.cube_b_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # 1. arm to the canonical vertical pinch pose. NOTE: no noise here —
        # measured 2026-11-25: even +-0.02 rad joint noise tilts the fingers off
        # vertical and breaks the face-contact grasp (init hold rate drops to 3/5).
        # Distribution spread comes from the carry-pose noise + xy jitter instead.
        q_grasp = np.clip(GRASP_POSE, self.arm_qmin, self.arm_qmax)
        self.data.qpos[self.arm_qpos_adr] = q_grasp
        self.data.ctrl[self.arm_act_idx] = q_grasp
        self.data.qvel[self.arm_dof_adr] = 0
        self.data.ctrl[self.grip_act_idx] = 0.019
        # let the arm settle under gravity: kinematic padmid z (0.0535) sags to the
        # validated dynamic position (0.0483). Without this the teleport height is wrong.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # 2. teleport cube A RESTING ON THE TABLE inside the straddle (xy padmid-relative,
        # z absolute at table rest). Measured 2026-11-25: teleporting at padmid-relative z
        # leaves the cube floating 4mm — it falls onto the pad edges and the close ejects it.
        pm = self._padmid()
        jitter = rng.uniform(-0.0015, 0.0015, size=2)
        self.data.qpos[self.cube_a_qpos:self.cube_a_qpos + 7] = [
            pm[0] + jitter[0], pm[1] + jitter[1], self.cube_size / 2 + 0.0002, 1, 0, 0, 0,
        ]
        self.data.qvel[self.cube_a_dof:self.cube_a_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # 3. close gripper and settle — friction seating (validated physics path)
        self.data.ctrl[self.grip_act_idx] = 0.0
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.ctrl[self.grip_act_idx] = -0.010
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

        # 4. cradle transition: joint-space interpolation to noisy carry pose
        cnoise = rng.uniform(-self.pose_noise, self.pose_noise, size=6)
        q_carry = np.clip(CARRY_POSE + cnoise, self.arm_qmin, self.arm_qmax)
        q0 = self.data.ctrl[self.arm_act_idx].copy()
        for i in range(120):
            a = (i + 1) / 120
            self.data.ctrl[self.arm_act_idx] = q0 + (q_carry - q0) * a
            mujoco.mj_step(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # desired goal: cube A stacked on cube B
        cb_pos = self.data.qpos[self.cube_b_qpos:self.cube_b_qpos + 3].copy()
        self._desired_goal = cb_pos + np.array([0, 0, self.cube_size])
        self._step_count = 0
        return self.make_obs(), info

    def held(self):
        """debug helper: is cube A plausibly held (well above table)?"""
        return bool(self.data.qpos[self.cube_a_qpos + 2] > 0.05)
