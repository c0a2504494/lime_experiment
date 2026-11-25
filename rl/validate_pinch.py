import os
os.environ["MUJOCO_GL"] = "egl"
import sys, numpy as np
sys.path.insert(0, "/content/work")
import mujoco
from scipy.optimize import minimize

model = mujoco.MjModel.from_xml_path("/content/work/scene_pad.xml")
data = mujoco.MjData(model)
model.dof_frictionloss[:8] = 0.02
model.dof_damping[:8] = 0.1
for i in range(6):
    model.actuator_gainprm[i, 0] = 400.0
    model.actuator_biasprm[i, 1] = -400.0
    model.actuator_biasprm[i, 2] = -40.0
    model.actuator_forcerange[i, 0] = -8.0
    model.actuator_forcerange[i, 1] = 8.0
model.actuator_forcerange[6, 0] = -4.0
model.actuator_forcerange[6, 1] = 4.0
model.actuator_gainprm[6, 0] = 1500.0
model.actuator_biasprm[6, 1] = -1500.0
model.actuator_biasprm[6, 2] = -15.0

jn = ["joint" + str(i) for i in range(1,7)]
jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn])
qadr = model.jnt_qposadr[jids]
dadr = model.jnt_dofadr[jids]
lo = model.jnt_range[jids, 0]; hi = model.jnt_range[jids, 1]
act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint" + str(i) + "_pos") for i in range(1,7)])
gact = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_pos")
gl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_link")
gr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_link")
jgl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_joint")
gqadr = model.jnt_qposadr[jgl]
cqa = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_free")
cqb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_b_free")
qa = model.jnt_qposadr[cqa]; qb = model.jnt_qposadr[cqb]

def padmid():
    return (data.xpos[gl] + data.xpos[gr]) / 2

def Rgl():
    return data.xmat[gl].reshape(3, 3)

def cubeA():
    return data.qpos[qa:qa+3].copy()

TILT = 90.0
CA = np.array([0.17, 0.0])
AD = CA / np.linalg.norm(CA)

def rdes_pinch():
    t = np.radians(TILT)
    zax = np.array([AD[0]*np.cos(t), AD[1]*np.cos(t), -np.sin(t)])
    yax = np.array([-AD[1], AD[0], 0.0])
    xax = np.cross(yax, zax); xax = xax / np.linalg.norm(xax)
    yax = np.cross(zax, xax)
    return np.column_stack([xax, yax, zax])

def solve_pinch(target, q0, wpos=30.0):
    Rdes = rdes_pinch()
    def cost(q):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        return wpos*np.sum((padmid() - target)**2) + np.sum((Rgl() - Rdes)**2)
    res = minimize(cost, q0, method="SLSQP", bounds=list(zip(lo, hi)), options={"maxiter": 500})
    return res.x

def move_to(qtarget, steps, grip=None):
    q0 = data.ctrl[act].copy()
    for i in range(steps):
        a = (i + 1) / steps
        data.ctrl[act] = q0 + (qtarget - q0) * a
        if grip is not None:
            data.ctrl[gact] = grip
        mujoco.mj_step(model, data)

def hold(steps, grip=None):
    for i in range(steps):
        if grip is not None:
            data.ctrl[gact] = grip
        mujoco.mj_step(model, data)

def servo_xy(target_xy, grip, iters=6):
    for it in range(iters):
        err3 = np.array([target_xy[0] - padmid()[0], target_xy[1] - padmid()[1], 0.0])
        Jl = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, Jl, None, gl)
        mujoco.mj_jacBody(model, data, Jr, None, gr)
        J = (Jl + Jr) / 2
        Ja = J[:, dadr]
        dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-4*np.eye(3), err3)
        dq = np.clip(dq, -0.1, 0.1)
        data.ctrl[act] = np.clip(data.ctrl[act] + dq, lo, hi)
        hold(150, grip=grip)

# 初期化
data.qpos[qadr] = [0.0, 0.1396, 0.903, 0.0, 1.7813, 0.0]
data.qpos[qa:qa+7] = [0.17, 0.0, 0.0285, 1, 0, 0, 0]
data.qpos[qb:qb+7] = [0.10, 0.10, 0.0285, 1, 0, 0, 0]
data.ctrl[act] = data.qpos[qadr]
data.ctrl[gact] = 0.019
mujoco.mj_forward(model, data)
for _ in range(200):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, 480, 360)
cam = mujoco.MjvCamera()
frames = {}
def snap(tag, az=60, el=-18, dist=0.32):
    c = cubeA()
    cam.lookat = [c[0], c[1], 0.06]
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    renderer.update_scene(data, cam)
    frames[tag] = renderer.render().copy()

qseed = np.array([0.0, 0.1396, 0.903, 0.0, 1.7813, 0.0])
t = np.radians(TILT)
# ピンチ目標: パッド中心がキューブ中央に
ptgt = np.array([CA[0] - AD[0]*0.032*np.cos(t), CA[1] - AD[1]*0.032*np.cos(t), 0.0285 + 0.032*np.sin(t) - 0.007])
# 事前位置: 接近方向に8cm手前、6cm上
pre = ptgt.copy()
pre[0] -= AD[0]*0.08; pre[1] -= AD[1]*0.08; pre[2] += 0.07

snap("01_home")
q_pre = solve_pinch(pre, qseed)
move_to(q_pre, 500, grip=0.019)
hold(200, grip=0.019)
snap("02_pre")

q_pinch = solve_pinch(ptgt, q_pre)
move_to(q_pinch, 500, grip=0.019)
hold(200, grip=0.019)
snap("03_straddle")

# センタリング(接近方向成分は除きたいが、まずxy全体で)
ca_now = cubeA()
servo_xy([ca_now[0] - AD[0]*0.032*np.cos(t), ca_now[1] - AD[1]*0.032*np.cos(t)], 0.019, iters=5)
snap("04_centered")
print("pre-close padmid:", np.round(padmid(), 4).tolist(), "cube:", np.round(cubeA(), 4).tolist())

# ゆっくり閉じる
data.ctrl[gact] = 0.0
hold(200, grip=0.0)
snap("05_closing")
data.ctrl[gact] = -0.010
hold(500, grip=-0.010)
snap("06_closed")
print("closed gq:", round(float(data.qpos[gqadr]), 4), "cube z:", round(float(cubeA()[2]), 4))
pl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad_l")
pr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad_r")
print("pad sep:", round(float(data.geom_xpos[pl][1] - data.geom_xpos[pr][1] - 0.008), 4))

# リフト: padmid z +0.09
lift_tgt = ptgt.copy(); lift_tgt[2] += 0.09
q_lift = solve_pinch(lift_tgt, q_pinch)
move_to(q_lift, 300, grip=-0.010)
snap("07_liftstart")
move_to(q_lift, 300, grip=-0.010)
snap("08_midlift")
move_to(q_lift, 300, grip=-0.010)
hold(300, grip=-0.010)
snap("09_lifttop")
print("after lift cube z:", round(float(cubeA()[2]), 4), "gq:", round(float(data.qpos[gqadr]), 4))
print("pinch-LIFT-OK" if cubeA()[2] > 0.05 else "pinch-lift-fail")
snap("10_final")

# フレーム保存
from PIL import Image
for tag, im in frames.items():
    Image.fromarray(im).save("/content/work/pinch_" + tag + ".jpg", quality=60)
print("frames saved:", len(frames))
