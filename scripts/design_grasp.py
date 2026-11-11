"""Design the tweezer grasp of the Wuji Hand 2 (offline, run once).

Finds a thumb-index pinch pose, places the tweezers through the pinch point,
and tabulates finger joint targets for a closure parameter c in [0, 1]:

    c = 0  -> finger pads 16 mm apart (not touching the tweezers)
    c = 1  -> pad targets 10 mm *past* each other (the soft finger gains turn
              this over-travel into squeezing force)

Output: assets/grasp.json (tool pose in the palm frame + the pinch table).
Needs scipy; everything else in the project runs without it.
"""
import json
import pathlib

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAD_THUMB = (0.0065, 0.0, 0.026)   # pad point in the distal-link frame
PAD_INDEX = (0.0065, 0.0, 0.020)
L_PINCH = 0.075                     # tweezer base -> pinch point [m]
W_OPEN, W_CLOSED = 0.016, -0.010    # skin-to-skin target separation at c=0 / c=1 [m]


def hand_model():
    xml = (ROOT / "assets/wuji_hand/mjcf/right.xml").read_text()
    xml = xml.replace('meshdir="../meshes/right/"', f'meshdir="{ROOT}/assets/wuji_hand/meshes/right/"')
    for k, pad in ((1, PAD_THUMB), (2, PAD_INDEX)):
        key = f'<body name="right_finger{k}_link4"'
        i = xml.index(key)
        j = xml.index(">", i) + 1
        xml = xml[:j] + f'<site name="pad{k}" pos="{pad[0]} {pad[1]} {pad[2]}" size="0.002"/>' + xml[j:]
    return mujoco.MjModel.from_xml_string(xml)


def main():
    m = hand_model()
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:8, 0] + 0.03, m.jnt_range[:8, 1] - 0.03
    mid, half = (lo + hi) / 2, (hi - lo) / 2

    def fk(q):
        d.qpos[:] = 0
        d.qpos[:8] = q
        mujoco.mj_kinematics(m, d)
        return (d.site("pad1").xpos.copy(), d.site("pad2").xpos.copy(),
                d.site("pad1").xmat.reshape(3, 3)[:, 0].copy(),
                d.site("pad2").xmat.reshape(3, 3)[:, 0].copy())

    # 1) pinch pose: pads 14 mm apart, facing each other, joints near mid-range
    def res_pinch(q):
        pt, pi, nt, ni = fk(q)
        dv = pi - pt
        w = np.linalg.norm(dv)
        a = dv / w
        return np.r_[(w - 0.014) * 200, 2 * (nt @ a - 1), 2 * (ni @ a + 1), 0.25 * ((q - mid) / half) ** 3]

    q_seed = np.array([0.38, 0.82, 0.97, -0.18, 0.69, -0.14, 0.87, 1.42])
    q0 = least_squares(res_pinch, q_seed, bounds=(lo, hi)).x
    pt, pi, _, _ = fk(q0)
    P = (pt + pi) / 2
    a = (pi - pt) / np.linalg.norm(pi - pt)          # thumb -> index
    z_palm = np.array([0.0, 0.0, 1.0])
    t = z_palm - (z_palm @ a) * a                    # tool axis: along the fingers
    t /= np.linalg.norm(t)
    y = np.cross(t, a)
    R = np.column_stack([a, y, t])                   # tool frame in palm frame
    base = P - L_PINCH * t
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())

    # how far each finger's skin sticks out beyond its pad point (ray cast from the pinch point),
    # so that both pads touch the tweezers at the same closure
    fk(q0)
    mujoco.mj_forward(m, d)
    gid = np.zeros(1, np.int32)
    d_thumb = mujoco.mj_ray(m, d, P, -a, None, 1, -1, gid)
    d_index = mujoco.mj_ray(m, d, P, a, None, 1, -1, gid)
    off_t, off_i = 0.007 - d_thumb, 0.007 - d_index
    print(f"skin offsets: thumb {off_t * 1e3:.2f} mm, index {off_i * 1e3:.2f} mm")

    # 2) closure table by continuation (w = separation of the two skin surfaces)
    def res_c(q, w, q_prev, off_t, off_i):
        pt, pi, nt, ni = fk(q)
        return np.r_[(pt - (P - a * (w / 2 + off_t))) * 1000, (pi - (P + a * (w / 2 + off_i))) * 1000,
                     0.3 * (nt @ a - 1), 0.3 * (ni @ a + 1), 0.05 * (q - q_prev)]

    def make_table(off_t, off_i):
        cs = np.linspace(0, 1, 27)
        table, q_prev = [], q0.copy()
        for c in cs:
            w = W_OPEN + c * (W_CLOSED - W_OPEN)
            s = least_squares(res_c, q_prev, args=(w, q_prev, off_t, off_i), bounds=(lo, hi))
            q_prev = s.x
            table.append(s.x.tolist())
        return cs, table

    def save(off_t, off_i):
        cs, table = make_table(off_t, off_i)
        out = dict(
            tool_pos=base.tolist(), tool_quat=quat.tolist(), pinch_point=P.tolist(), pinch_axis=a.tolist(),
            l_pinch=L_PINCH, w_open=W_OPEN, w_closed=W_CLOSED, c_grid=cs.tolist(), q_thumb_index=table,
            # middle / ring / little fingers curl around the printed sleeve (kinematic pose)
            q_wrap=[1.05, 0.0, 1.15, 0.75, 1.10, 0.0, 1.15, 0.80, 1.15, 0.0, 1.20, 0.85],
            pad_thumb=PAD_THUMB, pad_index=PAD_INDEX, skin_offsets=[off_t, off_i],
        )
        (ROOT / "assets/grasp.json").write_text(json.dumps(out, indent=1))

    # 3) balance the two fingers: measure in the full simulation at which closure each
    #    finger starts to move its tweezer arm, then shift the offsets until both agree
    for it in range(6):
        save(off_t, off_i)
        c_touch = measure_touch()
        w_touch = [W_OPEN + c * (W_CLOSED - W_OPEN) for c in c_touch]
        print(f"pass {it}: thumb moves its arm at c={c_touch[0]:.3f}, index at c={c_touch[1]:.3f}")
        if abs(c_touch[0] - c_touch[1]) < 0.004:
            break
        mean = 0.5 * (w_touch[0] + w_touch[1])
        off_t += (w_touch[0] - mean) / 2
        off_i += (w_touch[1] - mean) / 2
    print("tool base (palm frame):", base.round(4), "axis:", t.round(3), "pinch axis:", a.round(3))


def measure_touch():
    """Closure at which the thumb / the index starts to move its tweezer arm."""
    import sys
    sys.path.insert(0, str(ROOT))
    import seedsafe.env as E
    E._MODEL_CACHE.clear()
    env = E.SeedEnv()
    theta0 = env.m.qpos_spring[env.jaw_q[0]]
    out = [1.0, 1.0]
    c = 0.0
    while c < 1.0 and max(out) == 1.0:
        c += 0.004
        env.closure = c
        for _ in range(12):
            env.tick()
        jaw = env.d.qpos[env.jaw_q]
        for k in range(2):
            if out[k] == 1.0 and jaw[k] < theta0 - 0.0015:
                out[k] = c
    return out


if __name__ == "__main__":
    main()
