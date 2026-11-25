"""Lime (SARA arm) cube-stacking environment for MuJoCo + Gymnasium + HER.

Goal-conditioned environment: pick cube_a and stack it on cube_b.
Action: 4D = (EE delta xyz scaled, gripper open/close command).
Control: differential IK (damped least squares, position + fixed downward orientation).

Observation layout (shared with the future ROS2 real-robot node — keep make_obs identical):
  qpos_arm (8), qvel_arm (8), ee pos+quat (7),
  cube_a pos+quat+linvel+angvel (13), cube_b same (13),
  ee->a vec (3), a->b vec (3)  => 55 dims
achieved_goal: cube_a position (3)
desired_goal:  cube_b position + one cube height (3)
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


CUBE_SIZE = 0.057


def _quat2mat(quat):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, quat)
    return m.reshape(3, 3)


def _mat2quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


class LimeCubeStackEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_xml_path,
        frame_skip=25,          # 0.002 * 25 = 50 ms -> 20 Hz policy
        max_steps=100,          # 5 s episodes at 20 Hz
        pos_action_scale=0.05,  # m per unit action
        ik_damping=1e-2,
        success_xy_tol=0.025,
        success_z_tol=0.015,
        randomize=True,
        cube_size=CUBE_SIZE,
        render_width=640,
        render_height=480,
        camera="overview",
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.pos_action_scale = pos_action_scale
        self.ik_damping = ik_damping
        self.success_xy_tol = success_xy_tol
        self.success_z_tol = success_z_tol
        self.randomize = randomize
        self.cube_size = cube_size
        self.camera = camera
        self.render_width = render_width
        self.render_height = render_height

        # --- indices -------------------------------------------------------
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.arm_qpos_idx = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.arm_joint_names
        ])
        # joint id -> qpos address
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_names]
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_names]

        jid_gl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_joint")
        jid_gr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_right_joint")
        self.grip_qpos_adr = np.array([self.model.jnt_qposadr[jid_gl], self.model.jnt_qposadr[jid_gr]])
        self.grip_dof_adr = np.array([self.model.jnt_dofadr[jid_gl], self.model.jnt_dofadr[jid_gr]])

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.cube_a_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
        self.cube_b_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_b")
        self.cube_a_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_free")
        self.cube_b_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_b_free")
        self.cube_a_qpos = self.model.jnt_qposadr[self.cube_a_jnt]
        self.cube_b_qpos = self.model.jnt_qposadr[self.cube_b_jnt]
        self.cube_a_dof = self.model.jnt_dofadr[self.cube_a_jnt]
        self.cube_b_dof = self.model.jnt_dofadr[self.cube_b_jnt]

        # actuator indices: joints 1-6 position actuators, then gripper
        self.arm_act_idx = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}_pos")
            for i in range(1, 7)
        ])
        self.grip_act_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_pos")

        # joint limits for clamping IK targets
        self.arm_qmin = self.model.jnt_range[self.arm_joint_names, 0]
        self.arm_qmax = self.model.jnt_range[self.arm_joint_names, 1]

        # fixed downward gripper orientation target: gripper z axis -> world -z.
        # Determined at reset from the home pose (see _set_home).
        self.ee_target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # action space: (dx, dy, dz, gripper) in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        obs_dim = 8 + 8 + 7 + 13 + 13 + 3 + 3
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        })

        self._step_count = 0
        self._renderer = None
        self._home_key_qpos = np.array([0.0, -1.1, 1.9, 0.0, 1.0, 0.0])  # rough downward pose, tuned in scripted test

    # ---------------------------------------------------------------- utils
    def _ee_pos_quat(self):
        pos = self.data.site_xpos[self.ee_site_id].copy()
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        quat = _mat2quat(mat)
        return pos, quat

    def _cube_state(self, qpos_adr, dof_adr):
        pos = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
        quat = self.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
        linvel = self.data.qvel[dof_adr:dof_adr + 3].copy()
        angvel = self.data.qvel[dof_adr + 3:dof_adr + 6].copy()
        return pos, quat, linvel, angvel

    def make_obs(self):
        qpos_arm = self.data.qpos[self.arm_qpos_adr]
        qpos_grip = self.data.qpos[self.grip_qpos_adr]
        qvel_arm = self.data.qvel[self.arm_dof_adr]
        qvel_grip = self.data.qvel[self.grip_dof_adr]
        ee_pos, ee_quat = self._ee_pos_quat()
        ca = self._cube_state(self.cube_a_qpos, self.cube_a_dof)
        cb = self._cube_state(self.cube_b_qpos, self.cube_b_dof)
        obs = np.concatenate([
            qpos_arm, qpos_grip,
            qvel_arm, qvel_grip,
            ee_pos, ee_quat,
            ca[0], ca[1], ca[2], ca[3],
            cb[0], cb[1], cb[2], cb[3],
            ca[0] - ee_pos,
            cb[0] - ca[0],
        ])
        return {
            "observation": obs.astype(np.float64),
            "achieved_goal": ca[0].astype(np.float64),
            "desired_goal": self._desired_goal.astype(np.float64),
        }

    # ------------------------------------------------------------- HER hook
    def compute_reward(self, achieved_goal, desired_goal, info):
        # sparse: 0 on success, -1 otherwise (vectorized for HER)
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = np.where(d < self.success_xy_tol, 0.0, -1.0)
        return reward if reward.size > 1 else reward[0]

    def _is_success(self):
        ca = self._cube_state(self.cube_a_qpos, self.cube_a_dof)
        cb = self._cube_state(self.cube_b_qpos, self.cube_b_dof)
        xy = np.linalg.norm(ca[0][:2] - cb[0][:2])
        zdiff = ca[0][2] - cb[0][2]
        # cube_a resting one cube-height above cube_b center
        return bool(xy < self.success_xy_tol and abs(zdiff - self.cube_size) < self.success_z_tol)

    # ------------------------------------------------------------------- IK
    def _ik_step(self, target_pos, target_quat):
        """One damped-least-squares IK iteration; writes position-actuator targets."""
        ee_pos, ee_quat = self._ee_pos_quat()
        err_p = target_pos - ee_pos

        # rotational error in world frame: small-angle cross-product form
        r_cur = _quat2mat(ee_quat)
        r_des = _quat2mat(target_quat)
        err_r = 0.5 * (
            np.cross(r_cur[:, 0], r_des[:, 0])
            + np.cross(r_cur[:, 1], r_des[:, 1])
            + np.cross(r_cur[:, 2], r_des[:, 2])
        )
        err = np.concatenate([err_p, err_r])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp, jacr])[:, self.arm_dof_adr]  # 6x6

        lam = self.ik_damping
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), err)
        dq = np.clip(dq, -0.5, 0.5)

        q_now = self.data.qpos[self.arm_qpos_adr].copy()
        q_target = np.clip(q_now + dq, self.arm_qmin, self.arm_qmax)
        self.data.ctrl[self.arm_act_idx] = q_target

    # ------------------------------------------------------------ gymnasium
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # home pose for the arm; capture its orientation as the fixed target
        self.data.qpos[self.arm_qpos_adr] = self._home_key_qpos
        self.data.qpos[self.grip_qpos_adr] = [0.019, 0.019]
        mujoco.mj_forward(self.model, self.data)
        _, home_quat = self._ee_pos_quat()
        self.ee_target_quat = home_quat

        # ctrl targets = current (no initial jump)
        self.data.ctrl[self.arm_act_idx] = self._home_key_qpos
        self.data.ctrl[self.grip_act_idx] = 0.019

        # --- cube randomization -------------------------------------------
        rng = self.np_random
        if self.randomize:
            while True:
                ax = rng.uniform(0.13, 0.26)
                ay = rng.uniform(-0.12, 0.12)
                bx = rng.uniform(0.13, 0.26)
                by = rng.uniform(-0.12, 0.12)
                if 0.05 < np.hypot(ax - bx, ay - by) < 0.12:
                    break
            jitter = lambda: rng.uniform(-0.005, 0.005, size=2)
        else:
            ax, ay, bx, by = 0.20, -0.08, 0.20, 0.10
            jitter = lambda: np.zeros(2)

        half = self.cube_size / 2.0
        for qadr, (x, y) in [(self.cube_a_qpos, (ax, ay)), (self.cube_b_qpos, (bx, by))]:
            jx, jy = jitter()
            self.data.qpos[qadr:qadr + 7] = [x + jx, y + jy, half + 0.001, 1, 0, 0, 0]

        # friction / mass domain randomization (in-place model edit)
        if self.randomize:
            slide = rng.uniform(0.3, 1.0)
            mass = rng.uniform(0.070, 0.105)
            for body in (self.cube_a_body, self.cube_b_body):
                geom_adr = self.model.body_geomadr[body]
                self.model.geom_friction[geom_adr, 0] = slide
                self.model.body_mass[body] = mass

        mujoco.mj_forward(self.model, self.data)
        # settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # desired goal: cube_a stacked on cube_b
        cb_pos = self.data.qpos[self.cube_b_qpos:self.cube_b_qpos + 3].copy()
        self._desired_goal = cb_pos + np.array([0, 0, self.cube_size])
        return self.make_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ee_pos, _ = self._ee_pos_quat()
        target_pos = ee_pos + action[:3] * self.pos_action_scale
        # table safety clamp
        target_pos[2] = max(target_pos[2], 0.01)

        grip_cmd = action[3]  # -1 close .. +1 open
        grip_target = 0.019 if grip_cmd > 0 else -0.010
        self.data.ctrl[self.grip_act_idx] = grip_target

        # several IK iterations + physics steps within one policy step
        n_sub = 5
        for _ in range(n_sub):
            self._ik_step(target_pos, self.ee_target_quat)
            for _ in range(self.frame_skip // n_sub):
                mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        success = self._is_success()
        terminated = success
        truncated = self._step_count >= self.max_steps
        reward = 0.0 if success else -1.0
        info = {"is_success": success}
        return self.make_obs(), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
