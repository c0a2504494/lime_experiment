# RL Cube Stacking — operational handoff notes (2025-12)

## Colab cloud runtime (colab-cloud MCP)
- Runtime: `runtimes/KL8rekbAbs8mB9UpuGoeMZKvlI0vbgv5W1_soMIufUk.` (T4 standard, Colab Pro)
- Session: `runtimes/KL8rekbAbs8mB9UpuGoeMZKvlI0vbgv5W1_soMIufUk./sessions/s-6e10550e-9a62-414f-bc66-40a512c495a4`
- If dead: `list_runtimes` → reuse or `create_runtime` (T4) → `create_session(parent=...)`
- Work dir on Colab: `/content/work` (scene.xml, env_lime_cube.py, build_scene.py, stack_test2.py, stack_test.mp4)
- OAuth: colab-cloud already authenticated (google account with Colab Pro). Re-auth via `devin mcp login colab-cloud` if expired.

## File transfer protocol (PC → Colab) — IMPORTANT
- Single large base64 blob through execute_code LOSES 2-3 chars in transcription. Use chunked protocol:
  gzip compress → base64 → split 100-char chunks → send as `chunks[i]='...'` in ~25-chunk batches →
  Colab verifies len==100 per chunk → join → `gzip.decompress` (CRC validates). Store chunks in
  `/content/work/_transfer.json` between batches.
- Small patches (<1KB) can go directly as string replacements in execute_code.
- Images/mp4 Colab → PC: print base64 in stdout, recover full output from overflow file in
  `%LOCALAPPDATA%\Temp\devin.exe-overflows\*\content.txt`, parse nested JSON (`stdout` field), decode locally.
  PNGs should be saved with `optimize=True` to reduce size.

## Key findings (do not rediscover!)
- URDF joint frictionloss=1.0Nm / damping=0.7 makes position servos unable to move the arm.
  Fix applied in env `reset()`: frictionloss 0.02 (arm) / 0.01 (gripper), damping 0.1.
- Strict z-down gripper orientation drives joint5 to its 1.781 rad limit → workspace boundary;
  incremental IK gets stuck. Orientation-including IK DESTROYS position tracking (j2/j3 cancel motions).
  Current solution: position-only IK, home pose `[0, 0.1396, 0.903, 0, 1.7813, 0]` tilted ~25° fwd,
  zax=(0.31, 0, -0.95). ee home pos ≈ (0.17, 0, 0.098).
- Gripper fingers extend along local +z of ee frame; pads close along local y. Pad opening ~22-80mm.
- Kinematic chain: pitch joints j2/j3/j5 (y axes), yaw/roll j1/j4/j6 (z axes). Upper arm 124mm.
  z-down requires total pitch ≈ 180° — inherently at joint5 limit for far targets.
- Upstream URDF bug: duplicate inertial in link6 (removed placeholder, kept CAD value).

## Current state (scripted stacking test result, stack_test2.py)
- EE position tracking works after friction fix (err 1-3cm, was 10-16cm before).
- Grasp FAILS: descend target ee z 0.055 → actual 0.044; gripper closes but cube gets pushed
  (cube moved from (0.2,-0.08) to (0.189,-0.102)), no grip. cube z stayed 0.027.
- NEXT debugging steps:
  1. Measure gripper pad inner distance vs the 57mm cube. Full close ctrl=-0.010 may be too tight
     OR the pads may be too high/low. Grasp needs partial close for 57mm cube — compute correct
     ctrl from pad geometry (measure in sim: pad inner x positions at several ctrl values).
  2. Check finger tip z positions at descend height — fingers must reach below cube center (0.0285).
     ee z 0.044 may leave fingertips too high → adjust descend target lower (~0.035) or account
     for finger length offset from ee_site.
  3. Re-run scripted sequence, iterate on heights/positions. Video at /content/work/stack_test.mp4.

## Env design decisions (fixed)
- Action 4D (EE dxyz×0.05m + gripper ±), 20Hz policy, frame_skip 25, physics 500Hz (ts 0.002).
- Sparse reward -1/0 + HER; desired_goal = cubeB pos + 57mm z; achieved = cubeA pos. max_steps 100.
- Domain randomization at reset: slide friction 0.3-1.0, mass 70-105g, cube xy x∈0.13-0.22/y∈±0.12,
  separation 5-12cm.
- success: xy dist < 2.5cm, z diff within 1.5cm of 57mm. Lenient first, stricter later.
- make_obs layout must stay identical for sim/real ROS2 porting.

## Remaining pipeline after grasp works
1. scripted stack success validated → 2. train_sac_her.py (SB3 SAC+HER: gamma .95, lr 3e-4,
   batch 256, buffer 1e6, n_sampled_goal 4, goal_selection future, max ep 100) with artifact
   callbacks (training curve PNGs, eval mp4 videos, checkpoints) → 3. artifact sync via base64 pull
   → 4. curriculum expansion (workspace size, parking positions, grasp quality levels L1-L4).
- Long-term: hierarchical parking-position selection + sim-to-real via ROS2/ros2_control.
