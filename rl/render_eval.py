"""Render eval episodes of the current trained policy to mp4 — runs as a SEPARATE process
so EGL rendering crashes can never hurt training.

Usage:
  python render_eval.py --task grasp --ckpt /content/work/run_grasp/checkpoints/sac_25000.zip \
      --out /content/work/eval_grasp.mp4 --episodes 3

Loads the checkpoint, runs N greedy episodes, concatenates frames into one mp4.
Prints per-episode success and a final RENDER-DONE marker.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import argparse
import numpy as np

sys.path.insert(0, "/content/work")

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
from stable_baselines3.common.vec_env import DummyVecEnv

from env_lime_cube import LimeCubeStackEnv
from env_lime_grasp import LimeGraspEnv
from env_lime_stack import LimeStackEnv

SCENE = "/content/work/scene.xml"
TASKS = {"full": LimeCubeStackEnv, "grasp": LimeGraspEnv, "stack": LimeStackEnv}


def latest_ckpt(run_dir):
    import re
    ckdir = os.path.join(run_dir, "checkpoints")
    cands = []
    for f in os.listdir(ckdir):
        m = re.search(r"sac_(\d+)\.zip$", f)
        if m:
            cands.append((int(m.group(1)), os.path.join(ckdir, f)))
    if not cands:
        return None
    return max(cands)[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--run_dir", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    ckpt = args.ckpt
    if not ckpt and args.run_dir:
        ckpt = latest_ckpt(args.run_dir)
    if not ckpt or not os.path.exists(ckpt):
        print("no checkpoint found:", ckpt)
        sys.exit(1)
    print("loading", ckpt)

    env = TASKS[args.task](SCENE)
    model = SAC.load(ckpt, env=DummyVecEnv([lambda: TASKS[args.task](SCENE)]))

    import imageio.v2 as imageio
    all_frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=20_000 + ep)
        done = False
        ep_frames = []
        while not done:
            ep_frames.append(env.render())
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        ep_frames.append(env.render())
        succ = bool(info.get("is_success", False))
        print("[render] ep%d success=%s frames=%d" % (ep, succ, len(ep_frames)))
        # pad each episode boundary with a few duplicated frames as a visual separator
        all_frames.extend(ep_frames)
        all_frames.extend([ep_frames[-1]] * 10)
    env.close()
    imageio.mimsave(args.out, all_frames, fps=20)
    print("video saved:", args.out, "frames:", len(all_frames))
    print("RENDER-DONE")


if __name__ == "__main__":
    main()
