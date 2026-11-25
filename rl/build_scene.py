"""Build the MuJoCo scene: fixed-base SARA arm + table + 2 cubes + actuators + cameras.

Input:  /content/work/sara_arm.urdf  (or pass urdf_path)
Output: mujoco.MjSpec; compile() and to_xml() are left to the caller.
"""
import mujoco
import numpy as np


def _lookat(pos, target, up=(0, 0, 1)):
    pos = np.array(pos, dtype=float)
    target = np.array(target, dtype=float)
    z = pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(np.array(up, dtype=float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return list(x) + list(y)


def build_scene(cube_size=0.057, cube_mass=0.09, friction_slide=0.5,
                urdf_path="/content/work/sara_arm.urdf"):
    spec = mujoco.MjSpec.from_file(urdf_path)
    spec.option.timestep = 0.002
    half = cube_size / 2.0
    wb = spec.worldbody

    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
                pos=[0.15, 0, -0.021], rgba=[0.3, 0.3, 0.3, 1])
    wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.005],
                pos=[0, 0, -0.005], rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.4, 0.4, 0.01],
                pos=[0.15, 0, -0.011], rgba=[0.7, 0.55, 0.35, 1])
    wb.add_light(name="toplight", pos=[0.3, 0, 1.0], dir=[-0.3, 0, -1],
                 diffuse=[0.8, 0.8, 0.8])
    wb.add_camera(name="overview", pos=[0.5, -0.4, 0.4],
                  xyaxes=_lookat([0.5, -0.4, 0.4], [0.12, 0, 0.05]), fovy=55)
    wb.add_camera(name="front", pos=[0.55, 0.0, 0.15],
                  xyaxes=_lookat([0.55, 0.0, 0.15], [0.1, 0, 0.08]), fovy=50)
    wb.add_camera(name="top", pos=[0.15, 0.001, 0.7],
                  xyaxes=_lookat([0.15, 0.001, 0.7], [0.15, 0, 0]), fovy=50)

    friction = [friction_slide, 0.005, 0.0001]
    for name, pos, col in [
        ("cube_a", [0.20, -0.08, half], [0.9, 0.1, 0.1, 1]),
        ("cube_b", [0.20, 0.10, half], [0.1, 0.1, 0.9, 1]),
    ]:
        body = wb.add_body(name=name, pos=pos)
        body.add_freejoint(name=name + "_free")
        body.add_geom(name=name + "_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[half, half, half], mass=cube_mass, friction=friction, rgba=col)
        body.add_site(name=name + "_center", pos=[0, 0, 0], size=[0.005])

    for b in spec.bodies:
        if b.name == "end_effector_link":
            b.add_site(name="ee_site", pos=[0, 0, 0], size=[0.005], rgba=[1, 1, 0, 1])

    jr = {
        "joint1": (-2.827, 2.827),
        "joint2": (-1.698, 1.766),
        "joint3": (-1.759, 1.856),
        "joint4": (-1.571, 1.571),
        "joint5": (-1.668, 1.781),
        "joint6": (-2.827, 2.827),
    }
    for jname, (lo, hi) in jr.items():
        act = spec.add_actuator(name=jname + "_pos", target=jname)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = 100.0
        act.biasprm[1] = -100.0
        act.biasprm[2] = -20.0
        act.ctrlrange = [lo, hi]
        act.forcerange = [-3.0, 3.0]

    act = spec.add_actuator(name="gripper_pos", target="gripper_left_joint")
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0] = 50.0
    act.biasprm[1] = -50.0
    act.biasprm[2] = -5.0
    act.ctrlrange = [-0.010, 0.019]
    act.forcerange = [-1.0, 1.0]
    return spec


if __name__ == "__main__":
    spec = build_scene()
    model = spec.compile()
    print("scene compile OK nq:", model.nq, "nu:", model.nu)
    with open("/content/work/scene.xml", "w") as f:
        f.write(spec.to_xml())
    print("scene.xml written")
