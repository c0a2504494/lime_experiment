"""Bootstrap the Lime cube-stacking sim on a fresh Colab runtime.

Does everything needed after a runtime reset:
  1. clones turtlebot3_lime (public repo, shallow)
  2. expands the SARA arm xacro into a plain URDF (static package-path replacement,
     no ROS/ament needed)
  3. removes the upstream duplicate-inertial bug in link6
  4. builds scene.xml (fixed-base arm + table + 2 cubes + actuators + cameras)

Usage on Colab:  python bootstrap_colab.py
Output: /content/work/sara_arm.urdf, /content/work/scene.xml
"""
import os
import re
import subprocess
import sys

WORK = "/content/work"
REPO = "/content/turtlebot3_lime"
DESC = os.path.join(REPO, "turtlebot3_lime_description")
REPO_URL = "https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_lime.git"


def clone_repo():
    if os.path.isdir(REPO):
        print("repo already cloned")
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", "humble-devel", REPO_URL, REPO],
        check=True, capture_output=True,
    )
    print("repo cloned")


def expand_xacro():
    """Expand the arm macro xacro -> flat URDF without ROS."""
    os.makedirs(WORK, exist_ok=True)
    src_path = os.path.join(DESC, "urdf", "open_manipulator_sara.urdf.xacro")
    with open(src_path) as f:
        macro = f.read()

    # wrapper that instantiates the macro with empty prefix
    wrapper = (
        '<?xml version="1.0"?>\n'
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="sara_arm">\n'
        + re.search(r"<robot[^>]*>(.*)</robot>", macro, re.S).group(1)
        + '\n  <xacro:open_manipulator_sara prefix=""/>\n'
        "</robot>\n"
    )
    wrapper_path = os.path.join(WORK, "sara_wrapper.xacro")
    with open(wrapper_path, "w") as f:
        f.write(wrapper)

    import xacro

    doc = xacro.process_file(wrapper_path)
    urdf = doc.toprettyxml(indent="  ")

    # static replacement of package:// uris with absolute mesh paths
    urdf = urdf.replace("package://turtlebot3_lime_description", DESC)

    # upstream bug: link6 has two <inertial> elements. Remove the crude placeholder
    # (mass 0.033 / inertia 0.1) and keep the accurate CAD one (mass 0.14327573).
    link6 = re.search(r'<link name="link6">.*?</link>', urdf, re.S)
    assert link6, "link6 not found"
    block = link6.group(0)
    inertials = list(re.finditer(r"<inertial>.*?</inertial>", block, re.S))
    if len(inertials) > 1:
        placeholder = next(i for i in inertials if "0.033" in i.group(0))
        newblock = block.replace(placeholder.group(0), "")
        urdf = urdf.replace(block, newblock)
        print("duplicate inertial removed")

    out = os.path.join(WORK, "sara_arm.urdf")
    with open(out, "w") as f:
        f.write(urdf)
    print("urdf written:", out)
    return out


if __name__ == "__main__":
    clone_repo()
    try:
        import xacro  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xacro"], check=True)
    expand_xacro()

    # build the scene (build_scene.py must already be in WORK)
    sys.path.insert(0, WORK)
    import build_scene

    spec = build_scene.build_scene()
    model = spec.compile()
    print("scene compile OK nq:", model.nq, "nu:", model.nu)
    with open(os.path.join(WORK, "scene.xml"), "w") as f:
        f.write(spec.to_xml())
    print("scene.xml written")
