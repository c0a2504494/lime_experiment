"""Generate scripted demonstration episodes for the GRASP task inside LimeGraspEnv.

The demos are a privileged expert: they use TRUE cube positions (not noisy obs),
but act purely through the env's normal 4D action interface (EE delta + gripper),
so the recorded transitions are exactly what SAC+HER needs for buffer prefilling.

Control design (hard-won facts applied):
- IK in the env is POSITION-ONLY and acts on ee_site, which is NOT at the gripper
  centerline (~6cm lateral offset). So we servo the PAD MIDPOINT instead:
  padmid = mean(xpos[gripper_left_link], xpos[gripper_right_link]).
  Since padmid moves rigidly with the arm, ee delta == padmid delta.
- Phase machine: MOVE-ABOVE -> DESCEND -> CLOSE -> LIFT.
- Grasp height: validate15 used padmid z ~0.068 descend target (pads straddle cube
  mid-height). Tunable: if scripted success is low, inspect and adjust.

Usage on Colab:
  python make_grasp_demos.py [--episodes 100] [--out /content/work/demos_grasp.npz]
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/content/work")
import mujoco

from env_lime_grasp import LimeGraspEnv, LIFT_HEIGHT

SCENE = "/content/work/scene.xml"

ABOVE_Z = 0.12       # safe travel height for padmid
GRASP_Z = 0.062      # padmid descend target (validate15-ish; tune if success low)
CLOSE_STEPS = 14     # steps to keep closing the gripper
MAX_XY_SPEED = 0.5   # clip xy action (2.5cm/step at 0.05 scale)
MAX_Z_SPEED = 0.5


class ScriptedGrasper:
    """Phase-machine expert policy acting through env actions using padmid servo."""

    def __init__(self, env):
        self.env = env
        m = env.model
        self.b_gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_link")
        self.b_gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_link")
        if self.b_gl < 0 or self.b_gr < 0:
            raise RuntimeError("gripper bodies not found - check names in scene.xml")

    def reset(self):
        self.phase = 0
        self.phase_t = 0

    def _padmid(self):
        d = self.env.data
        return 0.5 * (d.xpos[self.b_gl] + d.xpos[self.b_gr])

    def _cube_a_true(self):
        return self.env.data.qpos[self.env.cube_a_qpos:self.env.cube_a_qpos + 3].copy()

    def _servo_action(self, target):
        err = target - self._padmid()
        act = err / self.env.pos_action_scale
        act[0] = np.clip(act[0], -MAX_XY_SPEED, MAX_XY_SPEED)
        act[1] = np.clip(act[1], -MAX_XY_SPEED, MAX_XY_SPEED)
        act[2] = np.clip(act[2], -MAX_Z_SPEED, MAX_Z_SPEED)
        return act

    def act(self):
        cube = self._cube_a_true()
        spawn_goal = self.env._desired_goal  # spawn + LIFT_HEIGHT z
        self.phase_t += 1

        if self.phase == 0:  # MOVE-ABOVE: xy center over cube at safe height
            tgt = np.array([cube[0], cube[1], ABOVE_Z])
            a = self._servo_action(tgt)
            pm = self._padmid()
            xy_err = np.hypot(pm[0] - cube[0], pm[1] - cube[1])
            if xy_err < 0.004 and abs(pm[2] - ABOVE_Z) < 0.02:
                self.phase = 1
                self.phase_t = 0
            grip = 1.0  # open wide
        elif self.phase == 1:  # DESCEND to grasp height
            tgt = np.array([cube[0], cube[1], GRASP_Z])
            a = self._servo_action(tgt)
            pm = self._padmid()
            if abs(pm[2] - GRASP_Z) < 0.008:
                self.phase = 2
                self.phase_t = 0
            grip = 1.0
        elif self.phase == 2:  # CLOSE gripper in place
            tgt = np.array([cube[0], cube[1], GRASP_Z])
            a = self._servo_action(tgt)
            a[:] = 0.0  # hold position while closing
            if self.phase_t >= CLOSE_STEPS:
                self.phase = 3
                self.phase_t = 0
            grip = -1.0  # close
        else:  # LIFT toward the goal point (spawn + 8cm, small margin)
            tgt = spawn_goal + np.array([0, 0, 0.04])
            a = self._servo_action(tgt)
            grip = -1.0  # keep closed
        return np.array([a[0], a[1], a[2], grip], dtype=np.float32)


def generate(env, episodes, out_path, seed0=0, verbose=True):
    successes = []
    rec = {k: [] for k in ['obs_observation', 'obs_achieved_goal', 'obs_desired_goal',
                            'nobs_observation', 'nobs_achieved_goal', 'nobs_desired_goal',
                            'actions', 'rewards', 'dones']}
    total_steps = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        pol = ScriptedGrasper(env)
        pol.reset()
        ep_transitions = []
        done = False
        while not done:
            act = pol.act()
            nobs, rew, term, trunc, info = env.step(act)
            ep_transitions.append((obs, nobs, act, rew, term))
            obs = nobs
            done = term or trunc
        succ = bool(info.get('is_success', False))
        successes.append(succ)
        if succ:
            for (o, no, a, r, t) in ep_transitions:
                rec['obs_observation'].append(o['observation'])
                rec['obs_achieved_goal'].append(o['achieved_goal'])
                rec['obs_desired_goal'].append(o['desired_goal'])
                rec['nobs_observation'].append(no['observation'])
                rec['nobs_achieved_goal'].append(no['achieved_goal'])
                rec['nobs_desired_goal'].append(no['desired_goal'])
                rec['actions'].append(a)
                rec['rewards'].append(r)
                rec['dones'].append(t)
            total_steps += len(ep_transitions)
        if verbose and (ep + 1) % 10 == 0:
            print('[demo] ep %d/%d success %d/%d (len %d)' % (
                ep + 1, episodes, sum(successes), ep + 1, len(ep_transitions)))

    rate = np.mean(successes) if successes else 0.0
    print('[demo] scripted success rate: %.2f (%d/%d), recorded transitions: %d' % (
        rate, sum(successes), len(successes), total_steps))

    if total_steps > 0:
        out = {
            'obs_observation': np.array(rec['obs_observation']),
            'obs_achieved_goal': np.array(rec['obs_achieved_goal']),
            'obs_desired_goal': np.array(rec['obs_desired_goal']),
            'next_obs_observation': np.array(rec['nobs_observation']),
            'next_obs_achieved_goal': np.array(rec['nobs_achieved_goal']),
            'next_obs_desired_goal': np.array(rec['nobs_desired_goal']),
            'actions': np.array(rec['actions']),
            'rewards': np.array(rec['rewards']),
            'dones': np.array(rec['dones']),
            'obs_keys': np.array(['observation', 'achieved_goal', 'desired_goal']),
            'episode_success': np.array(successes),
        }
        np.savez(out_path, **out)
        print('[demo] saved to', out_path)
    return rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--out', type=str, default='/content/work/demos_grasp.npz')
    args = p.parse_args()
    env = LimeGraspEnv(SCENE)
    generate(env, args.episodes, args.out)
    env.close()
    print('DEMOGEN-DONE')


if __name__ == '__main__':
    main()
