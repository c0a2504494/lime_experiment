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
- Session 3 conclusion (validate7-9 + mesh analysis + closeup image): **the 57mm cube is at the
  gripper's geometric limit and CANNOT be stably grasped in sim** — root cause is finger taper:
  - Finger inner-gap width varies with height: ~86mm near tips (z 0-0.02) → ~57mm at z 0.04-0.06.
    Tips flare outward. Contact on the 57mm cube concentrates at the TOP EDGE band (z 0.049-0.056)
    regardless of centering (xy err <0.5mm achieved), squeeze force (tested kp250/±4N ≈ 4N per
    finger — way above the 0.9N needed) or friction (geom friction 3.0 test).
  - Close-up render CONFIRMS: finger tips flare outward at the bottom; right finger clips the
    cube corner. Contacts are edge/point-like, cube rotates out during lift.
  - Fingertips DO reach the table at padmid z≈0.057 (mesh z min -0.009 world) — long fingers
    (mesh z span ~9cm) are not the blocker; the TAPER is.
  - xy closed-loop servo correction WORKS (servo_xy in validate9.py — iterative jacBody position
    correction in padmid frame, error <0.5mm). Reusable for RL env control.
  - Servo tuning applied in validate7-9: joints kp400/kv40/forcerange±8, gripper kp250/kv25/±4.
  - USER DECISION: verify on REAL robot whether 57mm cube can be grasped at all before further
    sim work. Sim development PAUSED pending result. If real grasp works, replicate its method
    (approach height/angle/grip width) in sim. If not: smaller cubes (30-40mm fit inside taper),
    rubber friction pads, or flattened finger collision geoms (sim-only) are the fallback options.
- Key validation scripts on Colab /content/work: validate9.py (full pipeline: offline IK waypoints
  + joint-interpolated execution + closed-loop xy servo + lift test), pf.py (inner-face profile),
  mm.py (mesh vertex measurement), vi.py (closeup render). Copy into repo if runtime dies.

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

## Session 3 update (2025-12) — rubber pad debugging IN PROGRESS

User direction: increase friction BUT central grasp is essential (摩擦を大きく、中央をつかむ).

### Approach being tested
Flat rubber pads added as box geoms to gripper fingers (real-world analog: gluing thick rubber blocks).
Implementation in `/content/work/scene_pad.xml` on Colab (NOT yet in git):
- finger mesh geoms set contype=0 conaffinity=0 (collision disabled)
- pad_l/pad_r box geoms: size [0.012, 0.004, 0.028], local pos [0, ±0.008, 0.032], friction="3 0.1 0.05"
- pad inner separation vs ctrl: ctrl 0.019 → 88mm, ctrl 0.010 → 70mm, ctrl 0.0 → 50mm
- cube is 57mm wide → contact occurs around ctrl ≈ +0.0035; full close target -0.010 = sep 30mm (way past cube)

### BREAKTHROUGH — contact solver compliance was the root cause (SOLVED)
Isolated test (no arm, no gravity, box-box): MuJoCo DEFAULT contacts allow **22mm penetration** under
kp1500 servo squeeze — the pads were passing through the cube in all earlier tests. This was NEVER a
geometry problem for the padded gripper — it was solver softness.

FIX (verified in isolated test): add to BOTH cube and pad geoms:
```
solref="0.002 1" solimp="0.99 0.999 0.001"
```
penetration drops to 0.9mm (0.2mm with solimp last param 0.0001). Applied in scene_pad.xml to pads+cubes.
CAREFUL: python string escaping wrote literal `\"` into the XML once — check for backslashes if XML parse error.

### State after fix (validate15.py run, grasp phase now CORRECT):
- pad inner sep 56.6mm vs cube 57mm — proper blocking, no penetration
- contacts at z 0.002-0.003 and 0.056 = corners of the contact patch (NORMAL for box-box: contact points
  are generated at patch corners only, this IS correct face-face contact)
- cube unrotated (quat identity), gripper z-axis exactly down, pad z centered 0.030 = cube mid-height
- xy centering error < 0.2mm — closed-loop servo works
- CENTRAL GRASP ACHIEVED geometrically — user's requirement satisfied

### LIFT BLOCKER — ROOT CAUSE IDENTIFIED: pads drag on the table
Diagnosed facts:
- lift IK for z 0.13 target reaches only z 0.101 (workspace limit, err 31mm) BUT still 4cm above grasp height — should visibly rise
- arm servos track fine under load (direct j3 test: err 0.008, pads move)
- THE REAL PROBLEM: pad half-height 0.028, pad center z ~0.030 at grasp → **pad bottom edge at z 0.002 = touching the table**. Friction 3.0 between pads and table anchors the whole arm — rigid chain arm→pads→table. This is why NOTHING moves during lift and why contacts cluster at z 0.002.
THE FIX (do this first next session):
1. Shorten pads: size z half 0.028 → 0.021 (span z 0.008-0.050 still covers cube mid), AND/OR
2. Raise descend target a few mm so pad bottom edge clears table by ≥5mm
3. Re-run validate15 — lift should now work; then verify full stack sequence
4. IMPORTANT: pads must still span cube mid-height (cube z 0-0.057, mid 0.0285) for the central grasp requirement
Secondary: lift IK reaching only z 0.101 is fine for validation but note workspace limits; consider lifting along a more vertical path or accepting lower lift height (z 0.09 is enough to transport).

### Minimal test warnings
- Mid-air placement tests fail trivially: cube falls before pads close (gravity race). Must start in contact or disable gravity.
- iso2.py/pt.py on Colab: isolated penetration test harnesses — reuse them for contact param tuning.

### validate script inheritance chain
validate_grasp.py (git) → validate9/11/13/14/15.py (Colab only). validate15 = current best.
Key params in validate15: gripper kp1500 kv15, arm kp400 kv40 forcerange ±8, gripper forcerange ±4,
descend target padmid z 0.068, close ctrl -0.010 for 500 steps, lift target z 0.13 over 800 steps.

### Colab runtime (current)
runtimes/r-44f74e91-55cf-4f14-894b-546895ea1760 / sessions/s-629cf6d0-67f6-426f-abb5-21b8ddebf080
Scripts: validate11-15.py (grasp validation iterations), mt.py/st.py/vi2.py (minimal tests), pad2.py, pm.py, pa.py

## Session 4 (2025-12) — GRASP SUCCESS! Friction-based lift achieved

User direction: reproduce official LimeSimulDemo grasping method in MuJoCo + take 10 photos.

### Official demo analysis (LimeSimulDemo, cloned at C:\Users\c0a2504494\Documents\dojo\lime\LimeSimulDemo)
- **Uses gazebo_grasp_plugin — grasping is FAKED** (object gets fixed-jointed to gripper), NOT friction physics
- Picks cola cans lightened to 39g (0.39→0.039kg — heavy objects can't be lifted even officially)
- Grasp style: HORIZONTAL pinch — pick pose joint=[dir,1.75,1.4,0,-1.6,0] has fingers horizontal (+x world),
  joint5 NEGATIVE (vs our z-down +1.781 approach). Base drives forward (shift 0.175) so object ends between fingers
- Pure horizontal pinch is KINEMATICALLY UNREACHABLE for our fixed elevated base (IK err 8cm at tilt 0-35°,
  6mm at tilt 65°, exact at tilt 80-90°) — official geometry depends on the mobile base's low mounting

### THE THREE FIXES that made friction grasping work (ALL required):
1. Flat pad box geoms + contact stiffening (session 3): solref="0.002 1" solimp="0.99 0.999 0.001" on pads AND cubes
2. **Equality/mimic constraint stiffening — THE missing piece**: under contact load the default soft equality
   let the right finger get pushed open (+0.0134 vs left -0.0065!) — cube squeezed out one-sidedly.
   Fix: same solref/solimp on the <equality><joint> element. Now both joints blocked at +0.0032 at cube contact.
   Free space test alone doesn't reveal this — MUST test under load. Now baked into make_pad_scene.py.
3. **Vertical grasp orientation (tilt 90)**: at tilt 65 the pads are tilted 25° from vertical → edge-only contact,
   lost instantly on lift. At tilt 90 pad faces vertical = full face contact with cube faces.
   ALSO: axis-aligned approach matters — closing direction must align with cube faces (cube A moved to (0.17, 0)).

### Verified results (validate_pinch.py in git, pinch90.py on Colab):
- closed gq +0.0032 = properly blocked at cube width (sep 56.5mm vs cube 57mm)
- normal force only ~0.93N total but friction 3.0 → capacity ~2.8N >> weight 0.88N
- **LIFT-OK: cube z 0.0631 airborne held (one run), 0.0455 another (workspace-limited lift height varies)**
- tilt 80 partially works (cube z 0.0376 then slips)
- Photos: rl_artifacts/pinch_success.jpg (10-frame grid showing straddle→close→airborne lift)

### STACK SUCCESS (validate_stack.py in git) — full pick-and-stack verified!
Result: xy err 8.6mm, zdiff exactly 0.057. Photos: rl_artifacts/stack_success.jpg (10-frame grid).
Method (details in validate_stack.py header comments):
1. Vertical pinch grasp at A=(0.17,0) → 2. home pose transition: cube CRADLED high (z 0.144 —
   beats the vertical-carry workspace ceiling z~0.064 which makes stacking impossible)
   → 3. B placed at measured carry radius, 45° away → PURE joint1 rotation swings cube over B
   (NO IK during transport — orientation constraints fight at rotated yaw and drop the cube!)
   → 4. descend by joint2 increments (+0.06rad/40steps, sign auto-detect, only 3 iterations needed)
   → 5. release: cube moves only 0.0856→0.0845 (gentle), retract.

### PRE-TRAINING STAGE COMPLETE (2025-12, commit 9fbe2d3) — ready to train!
1. scene.xml now INTEGRATES all validated physics: build_scene.py bakes servo gains
   (arm kp400 kv40 ±8, gripper kp1500 kv15 ±4), joint frictionloss 0.02/damping 0.1 (XML postproc),
   contact exclude; then chains make_pad_scene.py (pads + contact stiffening + equality stiffening).
   VALIDATED: st5.py stacking on integrated scene reproduced exactly (xy 8.6mm, zdiff 0.057).
2. env_lime_cube.py fixed: IK was broken (6x6 jacobian vs 3d err — crashed instantly! env had never
   run since position-only IK change) → now 3x6. compute_reward rewritten to match _is_success
   (xy+zdiff) — HER relabeling was inconsistent before.
   SMOKE TESTED: 5 seeds × 30 random steps stable, success teleport detected, rewards correct.
3. train_sac_her.py READY: SAC+HER frozen hyperparams (gamma .95, lr 3e-4, batch 256, buffer 1e6,
   n_sampled_goal 4, future). EvalCallback: success rate + mp4 videos every 25k, checkpoints 50k,
   curve.png. VERIFIED end-to-end with 600-step mini run (TRAINING-DONE, 299 updates).
4. Colab setup notes: pip install stable_baselines3 imageio imageio-ffmpeg; cv2 is BROKEN on colab
   (no .ocl) — train script patches it internally; MUJOCO_GL=egl needed for render (videos).
   Measured ~34 fps on cpu device → 300k steps ≈ 2.5h per run. MuJoCo physics is CPU-bound;
   consider --steps 300k for first real run.

### TASK DECOMPOSITION DESIGN (user decision 2026-11) — 2-task curriculum
User directive: split learning into two tasks instead of end-to-end.
Rationale: full stack needs 150-200 steps (gamma 0.95 effective horizon 20 can't propagate
final reward back to grasp actions); subtasks have short horizons where frozen gamma works.
Navigation/cube-discovery is the MOBILE BASE's job (real robot) — arm tasks may assume the
cube is already in the workspace. Base alignment means real-world spawn distribution is
EASIER than training DR — train wide, deploy easy.

- TASK 1 grasp (env_lime_grasp.py): cube A random on table, success = lift to spawn+8cm
  within 3cm tol. horizon 80 steps. desired_goal = spawn pos +8cm z (HER-compatible).
- TASK 2 stack (env_lime_stack.py): episode starts with cube A ALREADY GRASPED
  (canonical carry pose CARRY_POSE + joint noise ±0.05rad, cube teleported between pads,
  close gripper, settle 100 steps). horizon 100 steps. Success/reward = original env.
- train_sac_her.py --task {grasp,stack,full}; default grasp.
- KNOWN SIMPLIFICATION: stack init distribution is canonical+noise, not the true terminal
  distribution of the grasp policy. AFTER task 1 is trained: record terminal grasp states,
  use them as task-2 initial distribution (distribution matching — critical for chaining).
- Runtime chaining: state machine — grasp policy → success detect → stack policy (~10 lines).
- Order: train grasp first, then stack. Can run on separate colab runtimes in parallel if desired.

### NEW RUNTIME (2026-11-25, old ones all expired):
runtimes/r-a65415d6-e1f1-419c-b6d9-e510279f94f2 sessions/s-6e730616-6a8c-4d45-9dbb-3e10da3e9188
T4 standard. mujoco 4.0.1. NOTE: mujoco major version bumped (was 3.x) — watch for API diffs.
Bootstrap: pip install xacro mujoco scipy stable_baselines3 imageio imageio-ffmpeg;
clone turtlebot3_lime (humble-devel); transfer rl/*.py via push_gist.ps1 raw urls.

### NEXT (training phase):
1. Launch first real training run: `python train_sac_her.py --steps 300000` in BACKGROUND on colab
   (nohup, since MCP calls time out) — monitor via artifacts_test-style logs + curve.png
2. Expect: initial success rate 0 — sparse reward means HER is critical; watch for first nonzero evals.
   If still 0 at 100k+ steps, consider reward shaping or scripted-demo bootstrapping (BC init).
3. Known limitation: env uses position-only IK with home-tilt orientation — NOT the validated vertical
   grasp. RL must discover its own grasp within this constraint; if learning stalls, revisit orientation
   control or action space (joint-space actions alternative).
4. Remaining design Q: cube mass DR 70-105g vs official demo's 39g — keep as-is until evidence.

### Real-robot correspondence note
pads = physical rubber blocks glued inside fingers (hardware mod required!). Official demo lightened
cans to 39g even with fake grasping — real gripper torque is weak.

### OAuth token refresh procedure (recurring issue)
When colab-cloud MCP fails: token in C:\Users\c0a2504494\AppData\Roaming\devin\mcp\oauth\f0ebc51063be8558.json
expires hourly. Refresh: POST https://oauth2.googleapis.com/token with client_id/secret from
AppData\Roaming\devin\mcp_config.json + refresh_token from the json, update access_token+expires_at.

## LONG-TERM VISION (user decision 2026-11-25) — self-calibrating digital-twin manipulation stack

User's project goal: publish on GitHub a complete, reproducible system for TurtleBot3-class robots:
1. MEASUREMENT TOOLS: measure real-robot friction, mass, servo response curves, camera
   intrinsics/distortion. Re-runnable anytime = aging/wear adaptation (recalibration on degradation).
2. AUTO SIM CONSTRUCTION: bake measured parameters into MuJoCo scene automatically
   (build_scene.py is already parametric — extend it), run RL training automatically.
3. DIGITAL TWIN synchronized execution: run the sim in parallel with the real robot;
   fuse camera observations with physics-based prediction (particle-filter-like state estimation)
   so the system keeps estimating cube state even when camera loses it (occlusion, too close/blurry).
   NOTE: cubes-on-table are mostly-static targets — this is an EASY case for twin fusion.
4. BEHAVIOR STACK: rule-based navigation/discovery (mobile base) + RL grasp + RL stack = full autonomy.
5. PORTABILITY: works on any TurtleBot3-variant via URDF-driven scene generation; publish with
   README, reproduction steps, demo videos.

### KEY DESIGN DECISIONS from vision discussion (fixed unless user overrides)
- RL uses STATE observations, NOT raw camera pixels (same as OpenAI hide-and-seek / game AIs).
  End-to-end image RL costs 10-100x compute — rejected for now, maybe later as fine-tuning stage.
- Perception is a SEPARATE MODULE: camera + HSV color detection / ArUco markers -> same obs layout.
  Rubik's cube saturated colors are an ADVANTAGE for segmentation. OpenCV cannot distinguish sim
  renders from real images -> perception code develops in sim, ports unchanged to real robot.
- Perception-in-the-loop training (final form): render sim camera -> run REAL perception pipeline ->
  feed its output (noise/failures included) to policy. Agent never sees ground truth. Cost ~1.5-2x
  training time (not 10x like end-to-end CNN).
- OBSERVATION NOISE RANDOMIZATION: add noise to cube pose obs during training (pos +-5-10mm,
  orient few deg). Measured mean = DR center; measured spread = DR width. SysID and DR are
  complementary, not alternatives. Post-training noise-tolerance curve = perception spec sheet.
- Cube velocities unmeasurable on real robot: test zero-velocity robustness post-training;
  if degraded, retrain with cube vel/angvel zeroed (layout unchanged).
- Real-robot RL from scratch REJECTED (QT-Opt: 7 robots x months x 580k grasps for grasping alone;
  auto-reset problem unsolved; hardware wear). Path = sim pretrain -> real fine-tune (hundreds-thousands
  of real trials, feasible) -> continual learning with real-data sysid correction (real2sim2real).
- System identification measurements (real hardware, can start NOW):
  friction via incline-slip test, cube mass via scale, servo response fitting (MOST IMPORTANT),
  checkerboard camera calibration, drop-test for contact stiffness. Trajectory-matching auto-calib
  (real2sim optimization) is the advanced form.
- MuJoCo rigid-body approximation means residual gap NEVER zero -> real fine-tuning always needed,
  but small.

### PHASED ROADMAP (vertical slice first, generalize last)
- Phase 1 (NOW): complete 2-task RL in sim (grasp, stack) + chaining demo. Includes obs-noise option.
- Phase 2: measurement tools + first real-robot sysid; reflect into sim params.
- Phase 3: perception module dev in sim (color cube faces, render, color/ArUco detection);
  real-robot zero-shot eval + fine-tuning; perception-in-the-loop retraining.
- Phase 4: digital twin synchronized state estimation (physics prediction + observation fusion).
- Phase 5: portability, full automation, GitHub polish (docs, videos, repro instructions).
  GitHub repo currently PRIVATE github.com/c0a2504494/lime_experiment — decide when to go public.

### CLARIFIED MISCONCEPTIONS (user education — do not re-explain at length)
- RL here IS genuine trial-and-error RL; scripted validation was physics-proof only, never training data.
- "Ground truth observations" are standard practice (hide-and-seek/AlphaZero also use state, not pixels).
- Task decomposition is human curriculum design; each task is still learned purely by RL.

### IMMEDIATE NEXT ACTIONS (Phase 1 continuation)
1. Finish Colab bootstrap (interrupted mid-transfer): download rl files from gist 9a43eb4dbffcc5575f567e0fc109f1eb,
   run bootstrap_colab.py (urdf expand + scene build), verify mujoco 4.0.1 compatibility.
2. Smoke-test env_lime_grasp.py / env_lime_stack.py (esp. stack init: does cube stay grasped after settle?).
3. Decide with user: add obs-noise option BEFORE first training run or after baseline.
4. Train task=grasp 300k steps in background; monitor curve.png; decision gate at 100k (if success rate 0,
   analyze and consult user re: demos/shaping).
5. Record terminal grasp states -> use as task-2 init distribution (distribution matching).

