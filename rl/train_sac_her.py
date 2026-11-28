"""SAC+HER training for the Lime cube-stacking task (Colab).

Frozen design (see AGENTS.md / docs/rl_cube_stack_roadmap.md):
  SAC + HER, gamma 0.95, lr 3e-4, batch 256, buffer 1e6,
  n_sampled_goal 4, goal_selection future, max episode 100 steps.

Usage on Colab:
  MUJOCO_GL=egl python train_sac_her.py [--steps N] [--out DIR] [--task grasp|stack|full]
                                        [--demos PATH] [--resume]

Artifacts (saved to out dir AND auto-synced to ARTIFACT_GIST every eval):
  checkpoints/   model zip every 25k steps
  curve.png      training curve (success rate + reward)
  final_model.zip

Sync design lessons baked in (2026-11-28):
- checkpoints sorted NUMERICALLY (lexicographic sort bug made 'sac_100000' < 'sac_50000'
  and only the 50k checkpoint ever uploaded -> trained weights lost in round 3)
- latest checkpoint is re-uploaded EVERY sync unconditionally (no staleness tracking)
- checkpoint saved BEFORE eval at each 25k boundary so sync includes fresh weights
- sync never raises (network failures must not kill training)
- --resume downloads latest checkpoint from the artifact gist and warm-starts
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import argparse
import time
import re
import numpy as np

sys.path.insert(0, "/content/work")

import gymnasium as gym

# Colab ships a broken cv2 (no .ocl) which crashes SB3's atari_wrappers import.
# We never use atari/cv2 - patch the attribute if missing.
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
from stable_baselines3.common.vec_env import DummyVecEnv

from env_lime_cube import LimeCubeStackEnv
from env_lime_grasp import LimeGraspEnv
from env_lime_stack import LimeStackEnv

SCENE = "/content/work/scene.xml"

TASKS = {
    "full": LimeCubeStackEnv,    # end-to-end pick+stack (original, hardest)
    "grasp": LimeGraspEnv,       # task 1: lift cube A 8cm (horizon 80)
    "stack": LimeStackEnv,       # task 2: stack starting from grasped state (horizon 100)
}

CKPT_EVERY = 25_000


def make_env(task="full"):
    return TASKS[task](SCENE)


def _ckpt_step(filename):
    m = re.search(r"sac_(\d+)\.zip$", filename)
    return int(m.group(1)) if m else -1


def latest_checkpoint(ckdir):
    """Return (path, step) of numerically-latest checkpoint, or (None, -1)."""
    if not os.path.isdir(ckdir):
        return None, -1
    cands = [(f, _ckpt_step(f)) for f in os.listdir(ckdir) if f.endswith(".zip")]
    cands = [c for c in cands if c[1] >= 0]
    if not cands:
        return None, -1
    f, st = max(cands, key=lambda c: c[1])
    return os.path.join(ckdir, f), st


class EvalCallback(BaseCallback):
    """Periodic greedy eval: success rate over n episodes + artifact gist sync."""

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
        # checkpoint BEFORE eval: sync inside eval then carries fresh weights
        if self.num_timesteps % CKPT_EVERY < self.model.n_envs:
            self.model.save(os.path.join(self.out_dir, "checkpoints", f"sac_{self.num_timesteps}"))
        if self.num_timesteps % self.eval_every < self.model.n_envs:
            self._run_eval()
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
        self._save_curve()
        self._sync_artifacts()

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
            # brute-force: ALWAYS upload numerically-latest checkpoint (no staleness tracking)
            ckdir = os.path.join(self.out_dir, "checkpoints")
            zpath, zstep = latest_checkpoint(ckdir)
            if zpath:
                raw = open(zpath, 'rb').read()
                if len(raw) < 5_000_000:
                    files['ckpt_latest_zip.b64'] = {'content': b64mod.b64encode(raw).decode()}
                    files['ckpt_latest.txt'] = {'content': os.path.basename(zpath)}
            body = json.dumps({'files': files}).encode()
            req = urllib.request.Request('https://api.github.com/gists/' + gist_id, data=body, method='PATCH',
                headers={'Authorization': 'Bearer ' + tok, 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=60).read()
            print('[sync] artifacts pushed at t=%d ckpt=%s' % (self.num_timesteps, os.path.basename(zpath) if zpath else 'none'))
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


def fetch_latest_ckpt_from_gist(gist_id, tok_path='/content/work/.ghtok', dest='/content/work/resume_ckpt.zip'):
    """Download latest synced checkpoint from artifact gist. Returns path or None."""
    try:
        import json, urllib.request, base64 as b64mod
        tok = open(tok_path).read().strip()
        req = urllib.request.Request('https://api.github.com/gists/' + gist_id,
            headers={'Authorization': 'Bearer ' + tok, 'Accept': 'application/vnd.github+json'})
        meta = json.loads(urllib.request.urlopen(req, timeout=30).read())
        if 'ckpt_latest_zip.b64' not in meta.get('files', {}):
            print('[resume] no checkpoint in gist')
            return None
        raw_url = meta['files']['ckpt_latest_zip.b64']['raw_url']
        b64 = urllib.request.urlopen(raw_url, timeout=120).read().decode()
        raw = b64mod.b64decode(b64)
        open(dest, 'wb').write(raw)
        print('[resume] checkpoint downloaded: %d bytes' % len(raw))
        return dest
    except Exception as e:
        print('[resume] fetch failed:', e)
        return None


def prefill_demos(model, demos_path):
    """Load demo transitions (npz) into the replay buffer before learning."""
    d = np.load(demos_path, allow_pickle=True)
    n = len(d['actions'])
    obs_keys = list(d['obs_keys']) if 'obs_keys' in d else ['observation', 'achieved_goal', 'desired_goal']
    for i in range(n):
        obs = {k: d['obs_' + k][i] for k in obs_keys}
        nobs = {k: d['next_obs_' + k][i] for k in obs_keys}
        act = d['actions'][i]
        rew = float(d['rewards'][i])
        done = bool(d['dones'][i])
        model.replay_buffer.add(obs, nobs, act, rew, done, [{}])
    print('[demos] prefilled %d transitions (successes: %d)' % (n, int(d['episode_success'].sum()) if 'episode_success' in d else -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument('--out', type=str, default='/content/work/artifacts')
    p.add_argument('--task', type=str, default='grasp', choices=['full', 'grasp', 'stack'])
    p.add_argument('--demos', type=str, default='', help='npz of demo transitions to prefill')
    p.add_argument('--resume', action='store_true', help='warm-start from artifact gist checkpoint')
    args = p.parse_args()

    env = DummyVecEnv([lambda: make_env(args.task)])

    model = None
    if args.resume:
        gist_id = os.environ.get('ARTIFACT_GIST', '')
        ck = fetch_latest_ckpt_from_gist(gist_id) if gist_id else None
        if ck:
            try:
                model = SAC.load(ck, env=env, replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
                    verbose=1)
                print('[resume] warm-started from checkpoint')
            except Exception as e:
                print('[resume] load failed, starting fresh:', e)
                model = None

    if model is None:
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

    if args.demos and os.path.exists(args.demos):
        prefill_demos(model, args.demos)
    elif args.demos:
        print('[demos] WARNING: demo file not found:', args.demos)

    cb = EvalCallback(out_dir=args.out, task=args.task)
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(os.path.join(args.out, "final_model"))
    cb._sync_artifacts()
    print("TRAINING-DONE")


if __name__ == "__main__":
    main()
