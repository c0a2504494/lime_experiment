# RL Cube Stacking — operational handoff notes (2025-12)

## Colab cloud runtime (colab-cloud MCP)
- Runtime: `runtimes/r-44f74e91-55cf-4f14-894b-546895ea1760` (CPU standard — T4 was unavailable, capacity issue)
- Session: `runtimes/r-44f74e91-55cf-4f14-894b-546895ea1760/sessions/s-629cf6d0-67f6-426f-abb5-21b8ddebf080`
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
- Strict z-down gripper orientation drives joint5 to its 1.781 rad limit for HIGH/FAR targets,
  but z-down IS reachable exactly for low close targets (grasp height: zdown=-1.000, j5=1.54).
  Offline IK via scipy SLSQP works well: cost = 30*pos_err² + orient_err², chain waypoints by
  seeding each solve from previous solution. Orientation target Rdes=diag([-1,1,-1]).
- **world(table)×link2 penetration**: table geom extends under the arm base; link2 mesh dips 4.4mm
  into it at zero pose. Contact friction LOCKED joint1 completely (qfrc_constraint cancels actuator
  force exactly, phantom forces up to 43 million). FIX: `<exclude body1="world" body2="link2"/>`
  inside `<contact>` inserted into scene.xml post-generation. Bodies: no link1 exists (merged into
  world); moving chain is link2..link7 + gripper_left/right_link. After fix joint1 converges
  (err 0.003). TODO: bake the exclude into build_scene.py itself.
- **ee_site is NOT on gripper centerline**: end_effector_link is a side-mounted tool frame with
  ~6cm lateral offset from pad center — using it for IK meant the gripper closed on empty space
  6cm away from the cube. Pads sit ~4.4cm ABOVE ee_site (ee is near fingertip tips). DO NOT use
  ee_site for grasp positioning. Use pad-midpoint frame instead:
  padmid = (xpos[gripper_left_link] + xpos[gripper_right_link])/2, position cost on padmid +
  orientation cost on ee_site rotation (same rigid body rotation).
- Gripper fingers extend along local +z of ee frame; pads close along local y. Pad opening ~22-80mm.
- Kinematic chain: pitch joints j2/j3/j5 (y axes), yaw/roll j1/j4/j6 (z axes). Upper arm 124mm.
- Upstream URDF bug: duplicate inertial in link6 (removed placeholder, kept CAD value).

## Current state (validate2/3.py, 2025-12-16 session 2)
- Offline-IK waypoint solving in pad-midpoint frame + joint-space interpolated execution.
- Cube IS grasped and carried partway: lifted to z≈0.039, transported toward B, but SLIPS out
  during transport. Final xy err 9.6cm, no stack yet.
- Suspects for slip: (a) gripper squeeze force limited ±1N vs cube weight 0.9N — marginal with
  friction; (b) servo steady-state sag 2-3cm (joint forcerange ±3Nm binds at err>0.03 rad);
  (c) grip happens high on the cube edge (padmid z 0.0575 at close vs cube center 0.0285).
- validate3.py has forcerange patch (joints ±8, gripper ±4) but output was bit-identical —
  VERIFY the patch actually applied (string replace may have missed).
- Waypoints used (padmid frame): preA z0.12, graspA z0.048, lift z0.15, preB z0.15, place z0.108,
  cubes at (0.16,∓0.06). Solver residuals: graspA exact, high waypoints 4-6cm off (unreachable
  z-down high — fine for transit).
- NEXT: (1) verify/apply servo force strengthening, (2) render keyframes of close/lift to see
  exactly where contact happens, (3) consider lowering grasp target z or longer settle before
  close, (4) if slip persists increase gripper forcerange/friction or close harder (ctrl range).
- Session 2 end findings: validate4/5/6 — gripper DOES engage cube (gq stops at 0.0117 vs ctrl
  -0.010 target = cube blocks closing) but cube slips out vertically during lift (gq → 0.003,
  then -0.011 = fully closed empty). KEY: padmid z NEVER settles below ~0.057 regardless of
  target (0.048 or 0.035) or servo strength (forcerange ±8 + kp400 in validate6) — strongly
  suggests FINGERTIPS ARE ALREADY ON THE TABLE at padmid z 0.057, i.e. effective finger length
  joint→tip ≈ 5.7cm (not 4.4cm — that was pad-body-to-ee offset). If confirmed, grip geometry is
  table-constrained: fingers can't wrap lower. NEXT SESSION: (a) measure fingertip geom z and
  table contact at settled grasp pose; (b) render CLOSE-UP keyframes (zoom camera at cube);
  (c) if fingers too long to wrap cube bottom, options: accept high grip + increase friction
  (cube/pad contact friction), tilt approach, or grip cube higher deliberately and transport
  slowly; (d) also check pad taper/step shape — contact may be at cube top edge only.

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
