import sys, numpy as np
sys.path.insert(0, "/content/work")
import mujoco
from scipy.optimize import minimize

model = mujoco.MjModel.from_xml_path("/content/work/scene.xml")
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
model.actuator_gainprm[6, 0] = 250.0
model.actuator_biasprm[6, 1] = -250.0
model.actuator_biasprm[6, 2] = -25.0
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
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
Rdes = np.diag([-1., 1., -1.])

def padmid():
    return (data.xpos[gl] + data.xpos[gr]) / 2

def solve(target, q0, wpos=30.0):
    def cost(q):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        R = data.site_xmat[site_id].reshape(3,3)
        return wpos*np.sum((padmid() - target)**2) + np.sum((R - Rdes)**2)
    res = minimize(cost, q0, method="SLSQP", bounds=list(zip(lo,hi)), options={"maxiter":400})
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

def cubeA():
    return data.qpos[qa:qa+3].copy()

# 位置のみIK補正ステップ (padmid枠): 2ボディのjacobian平均
def servo_xy(target_xy, iters=6):
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
        hold(150, grip=0.019)

ax_, ay_, bx_, by_ = 0.16, -0.06, 0.16, 0.06
data.qpos[qadr] = [0.0, 0.1396, 0.903, 0.0, 1.7813, 0.0]
data.qpos[qa:qa+7] = [ax_, ay_, 0.0285, 1, 0, 0, 0]
data.qpos[qb:qb+7] = [bx_, by_, 0.0285, 1, 0, 0, 0]
data.ctrl[act] = data.qpos[qadr]
data.ctrl[gact] = 0.019
mujoco.mj_forward(model, data)
qseed = np.array([0.0, 0.1396, 0.903, 0.0, 1.7813, 0.0])

qa_pre = solve(np.array([ax_, ay_, 0.12]), qseed)
move_to(qa_pre, 400, grip=0.019)
qa_mid = solve(np.array([ax_, ay_, 0.06]), qa_pre)
move_to(qa_mid, 300, grip=0.019)
hold(200, grip=0.019)

# 低高度へ
ca = cubeA()
qa_low = solve(np.array([ca[0], ca[1], 0.025]), qa_mid)
move_to(qa_low, 300, grip=0.019)
hold(200, grip=0.019)

# 閉ループxy補正
ca = cubeA()
servo_xy(ca[:2], iters=6)
print("after servo: xy err", np.round(cubeA()[:2] - padmid()[:2], 4).tolist(), "padmid z", round(float(padmid()[2]), 4))

hold(300, grip=-0.010)
print("closed: gq", round(float(data.qpos[gqadr]), 4))
zs = []
for i in range(data.ncon):
    c = data.contact[i]
    b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
    pair = sorted([str(b1), str(b2)])
    if "cube_a" in pair and "gripper" in pair[0] + pair[1]:
        zs.append(round(float(c.pos[2]), 3))
print("contact z list:", sorted(zs))

ca = cubeA()
qa_lift = solve(np.array([ca[0], ca[1], 0.13]), qa_low)
move_to(qa_lift, 400, grip=-0.010)
hold(200, grip=-0.010)
print("after lift: cube z", round(float(cubeA()[2]), 4), "gq", round(float(data.qpos[gqadr]), 4))
print("LIFT-OK" if cubeA()[2] > 0.05 else "lift-fail")
