"""SAC+HER training for the Lime cube-stacking task (Colab).

Frozen design (see AGENTS.md / docs/rl_cube_stack_roadmap.md):
  SAC + HER, gamma 0.95, lr 3e-4, batch 256, buffer 1e6,
  n_sampled_goal 4, goal_selection future, max episode 100 steps.

Usage on Colab:
  MUJOCO_GL=egl python train_sac_her.py [--steps N] [--out DIR]

Artifacts (saved to out dir):
  checkpoints/    Emodel zip every 50k steps
  eval_videos/    Emp4 of greedy eval rollouts every 25k steps
  curve.png       Etraining curve (success rate + reward)
  final_model.zip
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, "/content/work")

import gymnasium as gym

# Colab ships a broken cv2 (no .ocl) which crashes SB3's atari_wrappers import.
# We never use atari/cv2  Epatch the attribute if missing.
try:
    import cv2
    if not hasattr(cv2, "ocl"):
        class _Ocl:
            @staticmethod
            def setUseOpenCL(x):
                pass
        cv2.ocl = _Ocl
except ImportError:
    pass

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from env_lime_cube import LimeCubeStackEnv
from env_lime_grasp import LimeGraspEnv
from env_lime_stack import LimeStackEnv

SCENE = "/content/work/scene.xml"

TASKS = {
    "full": LimeCubeStackEnv,    # end-to-end pick+stack (original, hardest)
    "grasp": LimeGraspEnv,       # task 1: lift cube A 8cm (horizon 80)
    "stack": LimeStackEnv,       # task 2: stack starting from grasped state (horizon 100)
}


def make_env(task="full"):
    return TASKS[task](SCENE)


class EvalCallback(BaseCallback):
    """Periodic greedy eval: success rate over n episodes + video frames."""

    def __init__(self, eval_every=25_000, n_eval=5, out_dir="/content/work/artifacts", task="full", video=False):
        super().__init__()
        self.eval_every = eval_every
        self.n_eval = n_eval
        self.out_dir = out_dir
        self.history = []
        self.task = task
        self.video = video
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every < self.model.n_envs:
            self._run_eval()
        if self.num_timesteps % 50_000 < self.model.n_envs:
            self.model.save(os.path.join(self.out_dir, "checkpoints", f"sac_{self.num_timesteps}"))
        return True

    def _run_eval(self):
        env = TASKS[self.task](SCENE)
        succ = []
        rews = []
        frames = []
        for ep in range(self.n_eval):
            obs, _ = env.reset(seed=10_000 + ep)
            done = False
            ep_rew = 0.0
            while not done:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                ep_rew += r
                done = term or trunc
                if self.video and ep == 0 and len(frames) < 100:
                    frames.append(env.render())
            succ.append(bool(info.get("is_success", False)))
            rews.append(ep_rew)
        rate = float(np.mean(succ))
        self.history.append((self.num_timesteps, rate, float(np.mean(rews))))
        print(f"[eval] t={self.num_timesteps} success={rate:.2f} rew={np.mean(rews):.1f}")
        self.logger.record("eval/success_rate", rate)
        if self.video and frames:
            import imageio.v2 as imageio
            os.makedirs(os.path.join(self.out_dir, "eval_videos"), exist_ok=True)
            imageio.mimsave(
                os.path.join(self.out_dir, "eval_videos", f"eval_{self.num_timesteps}.mp4"),
                frames, fps=20,
            )
        env.close()
        self._sync_artifacts()
        self._save_curve()

    def _sync_artifacts(self):
        """Push artifacts to a secret gist so runtime death never loses progress. NEVER raises."""
        try:
            import json, urllib.request, base64 as b64mod
            gist_id = os.environ.get('ARTIFACT_GIST', '')
            tok_path = '/content/work/.ghtok'
            if not gist_id or not os.path.exists(tok_path):
                return
            tok = open(tok_path).read().strip()
            files = {}
            files['eval_history.json'] = {'content': json.dumps(self.history)}
            files['train_meta.json'] = {'content': json.dumps({'task': self.task, 'steps': self.num_timesteps, 'ts': time.time()})}
            cpng = os.path.join(self.out_dir, 'curve.png')
            if os.path.exists(cpng):
                files['curve_png.b64'] = {'content': b64mod.b64encode(open(cpng, 'rb').read()).decode()}
            ckdir = os.path.join(self.out_dir, 'checkpoints')
            if os.path.isdir(ckdir):
                zips = sorted(f for f in os.listdir(ckdir) if f.endswith('.zip'))
                if zips and zips[-1] != getattr(self, '_synced_ckpt', None):
                    raw = open(os.path.join(ckdir, zips[-1]), 'rb').read()
                    if len(raw) < 5000000:
                        key = 'ckpt_latest_zip.b64'
                        files[key] = {'content': b64mod.b64encode(raw).decode()}
                        files['ckpt_latest.txt'] = {'content': zips[-1]}
                        self._synced_ckpt = zips[-1]
            body = json.dumps({'files': files}).encode()
            req = urllib.request.Request('https://api.github.com/gists/' + gist_id, data=body, method='PATCH',
                headers={'Authorization': 'Bearer ' + tok, 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=30).read()
            print('[sync] artifacts pushed at t=%d' % self.num_timesteps)
        except Exception as e:
            print('[sync] failed (non-fatal):', e)
    def _save_curve(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ts = [h[0] for h in self.history]
            sr = [h[1] for h in self.history]
            rw = [h[2] for h in self.history]
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(ts, sr, "g-o", label="success rate")
            ax1.set_ylabel("success rate")
            ax1.set_ylim(-0.05, 1.05)
            ax2 = ax1.twinx()
            ax2.plot(ts, rw, "b--", alpha=0.5, label="mean reward")
            ax2.set_ylabel("reward")
            ax1.set_xlabel("timesteps")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "curve.png"), dpi=80)
            plt.close(fig)
        except Exception as e:
            print("curve save failed:", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument('--out', type=str, default='/content/work/artifacts')
    p.add_argument('--task', type=str, default='grasp', choices=['full','grasp','stack'])
    args = p.parse_args()

    env = DummyVecEnv([lambda: make_env(args.task)])
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        gamma=0.95,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        verbose=1,
    )
    cb = EvalCallback(out_dir=args.out, task=args.task)
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(os.path.join(args.out, "final_model"))
    cb._sync_artifacts()
    print("TRAINING-DONE")


if __name__ == "__main__":
    main()


