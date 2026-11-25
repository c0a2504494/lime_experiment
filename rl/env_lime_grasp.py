"""Task 1 of the decomposed curriculum: GRASP learning.

Episode starts with cube A randomly placed on the table (same DR as the stack env).
Success: cube A is lifted to a point 8cm above its spawn location (within 3cm 3D tol).

Design notes:
- achieved_goal = cube A position; desired_goal = spawn pos + 8cm z.
  HER relabeling then teaches reaching/grasping/lifting progressively.
- horizon 80 steps (4s at 20Hz) — grasp+lift measured ~55-80 steps with fast play.
- gamma 0.95 is adequate at this horizon (effective horizon 20 steps vs task ~40-60).
"""
import numpy as np

from env_lime_cube import LimeCubeStackEnv

LIFT_HEIGHT = 0.08
SUCCESS_TOL = 0.03


class LimeGraspEnv(LimeCubeStackEnv):
    def __init__(self, scene_xml_path, max_steps=80, **kw):
        super().__init__(scene_xml_path, max_steps=max_steps, **kw)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # goal: lift cube A straight up by LIFT_HEIGHT from its spawn position
        a_pos = self.data.qpos[self.cube_a_qpos:self.cube_a_qpos + 3].copy()
        self._desired_goal = a_pos + np.array([0, 0, LIFT_HEIGHT])
        self._spawn_a = a_pos
        return self.make_obs(), info

    def _is_success(self):
        ca = self._cube_state(self.cube_a_qpos, self.cube_a_dof)
        d = np.linalg.norm(ca[0] - self._desired_goal)
        return bool(d < SUCCESS_TOL)

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = np.where(d < SUCCESS_TOL, 0.0, -1.0)
        return reward if reward.size > 1 else reward[0]
