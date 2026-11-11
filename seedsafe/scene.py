"""Builds the MuJoCo scene (plain MJCF text, so it works on any MuJoCo >= 3.1.3).

Chain:  Flexiv Rizon 4 arm (7-DoF stand-in for the Enlight)
          -> Wuji Hand 2 (20 DoF, official model)
            -> tweezers (sleeve seated in the hand, two elastic arms)
World:  strawberry on a soft cradle, target seeds, a tray, three cameras.
"""
import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import ASSETS, Config

N_DECOR = 200          # decorative seed geoms on the berry
FLANGE_Z = 0.081       # Rizon 4: joint-7 frame -> flange
ADAPTER = 0.012        # printed adapter between flange and palm
ADAPTER_TILT = -0.6     # tilt of the angled adapter about the flange y axis [rad]


def load_grasp():
    return json.loads((ASSETS / "grasp.json").read_text())


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def _mat_from_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _tool_xml(cfg: Config, grasp):
    tw = cfg.tweezers
    L = tw.length
    theta0 = float(np.arcsin(tw.tip_half_gap_rest / L))
    k_arm = tw.close_force * grasp["l_pinch"] / theta0
    i_arm = tw.arm_mass * L ** 2 / 3
    seat = ""
    for ax, name in (("1 0 0", "x"), ("0 1 0", "y"), ("0 0 1", "z")):
        seat += (f'<joint name="seat_t{name}" type="slide" axis="{ax}" stiffness="{tw.seat_stiff_lin}" '
                 f'damping="{2 * np.sqrt(tw.seat_stiff_lin * 0.04):.4g}" armature="0.01"/>\n')
    for ax, name in (("1 0 0", "x"), ("0 1 0", "y"), ("0 0 1", "z")):
        seat += (f'<joint name="seat_r{name}" type="hinge" axis="{ax}" stiffness="{tw.seat_stiff_rot}" '
                 f'damping="{2 * np.sqrt(tw.seat_stiff_rot * 2e-4):.4g}" armature="2e-5"/>\n')

    def arm(tag, sign):
        axis = "0 -1 0" if sign < 0 else "0 1 0"
        return f"""
      <body name="arm_{tag}" gravcomp="1">
        <joint name="jaw_{tag}" type="hinge" axis="{axis}" range="0 0.09" stiffness="{k_arm:.5g}"
               springref="{theta0:.6g}" damping="{tw.arm_damping}" armature="1e-6" solreflimit="0.002 1" solimplimit="0.99 0.999 0.0002"/>
        <inertial pos="{sign * 0.0005} 0 {L / 2}" mass="{tw.arm_mass}" diaginertia="{i_arm / 4:.3g} {i_arm / 4:.3g} 1e-8"/>
        <geom name="arm_{tag}_body" type="box" pos="{sign * 0.0005} 0 0.055" size="0.0005 0.004 0.04"
              material="steel" contype="0" conaffinity="0"/>
        <geom name="arm_{tag}_pad" type="box" pos="{sign * 0.0006} 0 {grasp['l_pinch']}" size="0.0006 0.003 0.005"
              rgba="0.2 0.2 0.22 1" contype="4" conaffinity="0" friction="0.9 0.005 0.0001" solref="0.004 1" condim="4"/>
        <geom name="arm_{tag}_tip" type="box" pos="{sign * 0.0004} 0 {0.095 + (L - 0.095) / 2}"
              size="0.0004 0.0009 {(L - 0.095) / 2}" material="steel" contype="0" conaffinity="0"/>
        <site name="tip_{tag}" pos="0 0 {L - 0.0002}" size="0.0002" rgba="1 0 0 0"/>
      </body>"""

    return f"""
    <body name="tool" pos="{_fmt(grasp['tool_pos'])}" quat="{_fmt(grasp['tool_quat'])}" gravcomp="1">
      {seat}
      <inertial pos="0 0 0.012" mass="{tw.sleeve_mass}" diaginertia="2e-6 2e-6 6e-7"/>
      <geom name="sleeve" type="box" pos="0 0 0.010" size="0.0055 0.0065 0.022" rgba="0.15 0.35 0.75 1"
            contype="0" conaffinity="0"/>
      <site name="tcp" pos="0 0 {L - 0.0002}" size="0.0003" rgba="0 1 0 0"/>
      {arm('a', -1)}
      {arm('b', +1)}
    </body>"""


def _berry_xml(cfg: Config):
    b = cfg.berry
    a = b.semi_axes
    damp = 2 * np.sqrt(b.cradle_k * b.mass)
    joints = "".join(
        f'<joint name="berry_{n}" type="slide" axis="{ax}" stiffness="{b.cradle_k}" damping="{damp:.3g}"/>'
        for n, ax in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")))
    decor = "".join(
        f'<geom name="dseed_{i}" type="ellipsoid" size="{b.seed_radius} {b.seed_radius * 1.3} {b.seed_radius * 0.8}" '
        f'pos="0 0 0" rgba="0.95 0.85 0.25 1" contype="0" conaffinity="0" mass="0"/>' for i in range(N_DECOR))
    leaves = ""
    for k in range(6):
        ang = k * np.pi / 3
        leaves += (f'<geom type="ellipsoid" size="0.006 0.0012 0.003" pos="{0.006 * np.cos(ang):.4f} {a[1] * 0.97:.4f} '
                   f'{0.006 * np.sin(ang):.4f}" euler="0 {-ang:.3f} 0" rgba="0.2 0.55 0.2 1" contype="0" conaffinity="0" mass="0"/>')
    cx, cy, cz = b.center
    seeds = "".join(
        f'<body name="seed_{i}" mocap="true" pos="0 0 -1"><geom name="seed_{i}" type="ellipsoid" '
        f'size="{b.seed_radius} {b.seed_radius * 1.3} {b.seed_radius * 0.8}" rgba="0.95 0.85 0.25 1" contype="0" conaffinity="0"/></body>'
        for i in range(b.max_targets))
    tx, ty, tz = cfg.tray_pos
    return f"""
    <body name="pedestal" pos="{cx} {cy} 0">
      <geom type="box" pos="0 0 {(cz - a[2] - 0.012) / 2}" size="0.035 0.045 {(cz - a[2] - 0.012) / 2}" rgba="0.82 0.82 0.85 1" contype="0" conaffinity="0"/>
      <geom name="cradle" type="cylinder" pos="0 0 {cz - a[2] - 0.006}" size="0.019 0.006" rgba="0.95 0.93 0.8 1" contype="0" conaffinity="0"/>
    </body>
    <body name="tray" pos="{tx} {ty} 0">
      <geom type="box" pos="0 0 {(tz - 0.004) / 2}" size="0.02 0.02 {(tz - 0.004) / 2}" rgba="0.82 0.82 0.85 1" contype="0" conaffinity="0"/>
      <geom name="tray" type="cylinder" pos="0 0 {tz - 0.002}" size="0.02 0.002" rgba="0.25 0.25 0.3 1" contype="0" conaffinity="0"/>
    </body>
    <body name="berry" pos="{cx} {cy} {cz}">
      {joints}
      <inertial pos="0 0 0" mass="{b.mass}" diaginertia="3e-6 3e-6 3e-6"/>
      <geom name="berry_flesh" type="ellipsoid" size="{_fmt(a)}" rgba="0.85 0.08 0.12 1" contype="0" conaffinity="0"/>
      {leaves}
      {decor}
    </body>
    {seeds}"""


def build_xml(cfg: Config) -> str:
    grasp = load_grasp()
    arm = ET.parse(ASSETS / "flexiv_rizon4/flexiv_rizon4.xml").getroot()
    hand = ET.parse(ASSETS / "wuji_hand/mjcf/right.xml").getroot()

    # --- assets ---------------------------------------------------------
    assets = []
    for el in arm.find("asset"):
        el = copy.deepcopy(el)
        if el.tag == "mesh":
            el.set("file", "flexiv_rizon4/assets/" + el.get("file"))
        assets.append(el)
    for el in hand.find("asset"):
        el = copy.deepcopy(el)
        el.set("file", "wuji_hand/meshes/right/" + el.get("file"))
        el.attrib.pop("content_type", None)
        assets.append(el)

    # --- arm: no collisions (the only contacts we need are finger <-> tweezers)
    arm_default = copy.deepcopy(arm.find("default"))
    arm_default.find("default").find("geom").set("contype", "0")
    arm_default.find("default").find("geom").set("conaffinity", "0")
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    link7 = [b for b in base.iter("body") if b.get("name") == "link7"][0]

    # --- hand -----------------------------------------------------------
    palm = copy.deepcopy(hand.find("worldbody").find("body"))
    palm.set("pos", f"0 0 {FLANGE_Z + ADAPTER}")
    # angled adapter: keeps the wrist away from its singularity when the tool points down
    palm.set("quat", _fmt([np.cos(ADAPTER_TILT / 2), 0, np.sin(ADAPTER_TILT / 2), 0]))
    for body in palm.iter("body"):
        body.set("gravcomp", "1")
        keep_col = body.get("name") in ("right_finger1_link4", "right_finger2_link4")
        for g in list(body.findall("geom")):
            if g.get("group") == "1":          # visual
                continue
            if keep_col:                        # finger pads that press the tweezers
                g.set("contype", "0"); g.set("conaffinity", "4")
                g.set("friction", "0.9 0.005 0.0001"); g.set("solref", "0.004 1"); g.set("condim", "4")
                g.set("group", "3")
            else:
                body.remove(g)
        for j in body.findall("joint"):
            j.attrib.setdefault("armature", "0.0002")
            j.set("damping", "0.002")
    palm.set("gravcomp", "1")
    # pad sites (used by the grasp design and for plots)
    for k, key in ((1, "pad_thumb"), (2, "pad_index")):
        b4 = [b for b in palm.iter("body") if b.get("name") == f"right_finger{k}_link4"][0]
        ET.SubElement(b4, "site", name=f"pad{k}", pos=_fmt(grasp[key]), size="0.001", rgba="0 0 1 0")

    # adapter + tool + wrist camera live in the palm frame
    R_pt = _mat_from_quat(grasp["tool_quat"])
    p_pt = np.array(grasp["tool_pos"])
    cam_off = np.array([0.0, 0.052, 0.045])
    R_tc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # camera looks along +z of the tool
    cam_pos = p_pt + R_pt @ cam_off
    cam_quat = _quat_from_mat(R_pt @ R_tc)
    palm.append(ET.fromstring(
        f'<geom name="adapter" type="cylinder" pos="0 0 {-ADAPTER / 2}" size="0.03 {ADAPTER / 2}" rgba="0.2 0.2 0.22 1" contype="0" conaffinity="0" mass="0.03"/>'))
    palm.append(ET.fromstring(
        f'<camera name="wrist" pos="{_fmt(cam_pos)}" quat="{_fmt(cam_quat)}" fovy="{cfg.sensors.cam_fovy}"/>'))
    cam_box = cam_pos - R_pt @ np.array([0, 0, 0.012])
    palm.append(ET.fromstring(
        f'<geom name="cam_body" type="box" pos="{_fmt(cam_box)}" quat="{_fmt(grasp["tool_quat"])}" size="0.011 0.011 0.01" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" mass="0.02"/>'))
    palm.append(ET.fromstring(_tool_xml(cfg, grasp)))
    link7.append(palm)

    # --- actuators --------------------------------------------------------
    acts = [copy.deepcopy(a) for a in arm.find("actuator")] + [copy.deepcopy(a) for a in hand.find("actuator")]
    for a in acts:        # arm: raise the stock (very soft, ~1.5 Hz) position gains to ~6 Hz
        if a.get("name").startswith("joint"):
            a.set("kp", str(float(a.get("kp")) * 16)); a.set("kv", str(float(a.get("kv")) * 4))
    for a in acts:        # stiffer position gains on the two pinching fingers (thumb, index)
        if "finger1" in a.get("name") or "finger2" in a.get("name"):
            g = cfg.tweezers.pinch_gain_scale * (cfg.tweezers.thumb_gain_extra if "finger1" in a.get("name") else 1.0)
            a.set("kp", str(float(a.get("kp")) * g))
            a.set("kv", str(float(a.get("kv")) * g ** 0.5))

    cx, cy, cz = cfg.berry.center
    root = ET.fromstring(f"""
<mujoco model="seed_removal">
  <compiler angle="radian" meshdir=".." autolimits="true"/>
  <option timestep="{cfg.control.dt}" integrator="implicitfast" cone="elliptic" impratio="5" noslip_iterations="4"/>
  <size memory="40M"/>
  <visual>
    <global offwidth="1280" offheight="960" azimuth="140" elevation="-25"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.32 0.32 0.32" specular="0.05 0.05 0.05"/>
    <map znear="0.002" zfar="20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.82 0.9" rgb2="0.95 0.95 0.97" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.62 0.64 0.66" rgb2="0.56 0.58 0.6" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/>
    <material name="steel" rgba="0.75 0.77 0.8 1" specular="0.9" shininess="0.9"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.4 1.2" dir="0.1 0.3 -1" diffuse="0.45 0.45 0.45" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="0.8 0.5 0.9" dir="-0.3 -0.4 -1" diffuse="0.2 0.2 0.2" specular="0.1 0.1 0.1" castshadow="false"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" material="grid" contype="0" conaffinity="0"/>
    <camera name="overview" pos="1.05 -0.85 0.62" xyaxes="0.63 0.78 0 -0.3 0.24 0.92" fovy="38"/>
    <camera name="closeup" pos="{cx + 0.075} {cy - 0.105} {cz + 0.07}" xyaxes="0.81 0.58 0 -0.25 0.35 0.9" fovy="30"/>
    <camera name="macro" pos="{cx + 0.02} {cy - 0.045} {cz + 0.032}" xyaxes="0.91 0.41 0 -0.16 0.36 0.92" fovy="28"/>
    {_berry_xml(cfg)}
  </worldbody>
</mujoco>""")
    root.insert(2, arm_default)
    hand_def = ET.SubElement(root, "default")
    for el in assets:
        root.find("asset").append(el)
    root.find("worldbody").append(base)
    act = ET.SubElement(root, "actuator")
    for a in acts:
        act.append(a)
    root.remove(hand_def)
    return ET.tostring(root, encoding="unicode")


def build_model(cfg: Config):
    """Compile the scene. Returns (model, grasp dict)."""
    out = ASSETS / "_generated"
    out.mkdir(exist_ok=True)
    path = out / "scene.xml"
    path.write_text(build_xml(cfg))
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, load_grasp()
