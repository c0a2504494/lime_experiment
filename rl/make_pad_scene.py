"""make_pad_scene.py — scene.xml にラバーパッド付きグリッパの衝突ジオメトリを追加する。

実機で指の内側に厚めのラバーブロックを貼るのと等価なモデル:
- 指メッシュの衝突を無効化 (contype=0 conaffinity=0)
- 平板ボックスパッド pad_l/pad_r を追加。面が常に平行 → 中央高さで面接触が可能
- 接触剛性 solref/solimp を強化 (デフォルト接触は22mmも貫通する — session3の重要な発見)

使い方: build_scene.py で scene.xml を生成した後に実行
    python rl/make_pad_scene.py scene.xml scene_pad.xml
"""
import sys
import re


def make_pad_scene(src_path: str, dst_path: str) -> None:
    xml = open(src_path).read()

    def disable_finger_geoms(x: str, body: str) -> str:
        m = re.search(r'<body name="' + body + r'"[^>]*>(.*?)</body>', x, re.S)
        seg = m.group(1)
        seg2 = re.sub(r"<geom ", '<geom contype="0" conaffinity="0" ', seg)
        return x[: m.start(1)] + seg2 + x[m.end(1) :]

    xml = disable_finger_geoms(xml, "gripper_left_link")
    xml = disable_finger_geoms(xml, "gripper_right_link")

    # パッド: size [x half, y half(thickness), z half]
    # z half 0.021 — 下端がテーブルに触れないよう短めに (session3の教訓: 0.028だとテーブルに引っかかる)
    stiff = 'solref="0.002 1" solimp="0.99 0.999 0.001"'
    pad_l = (
        '<geom name="pad_l" type="box" size="0.012 0.004 0.021" pos="0 0.008 0.032" '
        f'friction="3 0.1 0.05" {stiff} rgba="0.9 0.1 0.1 1"/>'
    )
    pad_r = (
        '<geom name="pad_r" type="box" size="0.012 0.004 0.021" pos="0 -0.008 0.032" '
        f'friction="3 0.1 0.05" {stiff} rgba="0.9 0.1 0.1 1"/>'
    )
    # mimic等式制約も硬化しないと負荷時に右指が押し開かれる (session4の発見)
    xml = xml.replace(
        '<joint joint1="gripper_right_joint" joint2="gripper_left_joint"/>',
        f'<joint joint1="gripper_right_joint" joint2="gripper_left_joint" {stiff}/>',
    )
    m = re.search(r'(<body name="gripper_left_link"[^>]*>)', xml)
    xml = xml[: m.end()] + pad_l + xml[m.end() :]
    m = re.search(r'(<body name="gripper_right_link"[^>]*>)', xml)
    xml = xml[: m.end()] + pad_r + xml[m.end() :]

    # キューブにも接触剛性を追加 (両方必要 — 柔らかい方に引っ張られる)
    xml = re.sub(
        r'(<geom name="cube_[ab]_geom")',
        rf"\1 {stiff}",
        xml,
    )

    open(dst_path, "w").write(xml)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "scene.xml"
    dst = sys.argv[2] if len(sys.argv) > 2 else "scene_pad.xml"
    make_pad_scene(src, dst)
    print(f"written: {dst}")
