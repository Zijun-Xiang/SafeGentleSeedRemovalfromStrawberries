"""Seed localization with the wrist camera.

survey(env):  move the camera above the berry, take one colour + depth image, find the
              yellow seeds, and return for each seed its surface point and surface normal
              in world coordinates.

Error sources that are modelled: pixel resolution, depth noise, hand-eye calibration
error, shading / occlusion in the rendered image. If no OpenGL context is available
(headless machine without EGL) a geometric camera model with the same noise is used.
"""
import mujoco
import numpy as np

from .env import frame_from_axis

_REN = dict(model=None, ren=None, failed=False)


def _renderer(env):
    """One off-screen renderer per process; re-created when the model changes (new berry)."""
    if _REN["failed"]:
        return None
    if _REN["model"] is not env.m:
        if _REN["ren"] is not None:
            _REN["ren"].close()
        s = env.cfg.sensors
        try:
            _REN["ren"] = mujoco.Renderer(env.m, s.cam_height, s.cam_width)
            _REN["model"] = env.m
        except Exception as exc:          # no GL: fall back to the geometric model
            _REN["ren"], _REN["model"], _REN["failed"] = None, None, True
            print(f"[perception] no renderer ({type(exc).__name__}); using the geometric camera model")
    return _REN["ren"]


def survey_pose(env, height=0.105):
    """Tool pose that puts the camera `height` above the nominal berry top, looking down."""
    top = np.array(env.cfg.berry.center) + np.array([0, 0, env.cfg.berry.semi_axes[2]])
    R = frame_from_axis([0, 0, -1.0], (0.0, 1.0, 0.0))
    cam_in_tool = np.array([0.0, 0.052, 0.045])
    tcp_in_tool = np.array([0.0, 0.0, env.cfg.tweezers.length - 0.0002])
    cam_goal = top + np.array([0, 0, height])
    return cam_goal + R @ (tcp_in_tool - cam_in_tool), R


def _label(mask):
    """Connected components (8-neighbourhood) of a boolean image. Returns list of pixel index arrays."""
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    ys, xs = np.nonzero(mask)
    comps = []
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0] >= 0:
            continue
        k = len(comps)
        stack = [(y0, x0)]
        lab[y0, x0] = k
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] < 0:
                        lab[yy, xx] = k
                        stack.append((yy, xx))
        comps.append(np.array(pix))
    return comps


def detect_seeds(rgb, depth, fovy_deg, min_area=8):
    """Returns list of dict(p_cam, n_cam, uv, area) in the camera frame (x right, y up, looking along -z)."""
    H, W = depth.shape
    f = 0.5 * H / np.tan(np.radians(fovy_deg) / 2)
    img = rgb.astype(float) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    seed = (r > 0.5) & (g > 0.42) & (b < 0.55 * r) & (g > 0.55 * r)
    flesh = (r > 0.3) & (g < 0.45 * r) & (b < 0.5 * r)
    vv, uu = np.mgrid[0:H, 0:W]
    out = []
    for pix in _label(seed):
        if len(pix) < min_area:
            continue
        cy, cx = pix.mean(axis=0)
        rad = np.sqrt(len(pix) / np.pi)
        d2 = (vv - cy) ** 2 + (uu - cx) ** 2
        ring = flesh & (d2 > (1.6 * rad) ** 2) & (d2 < (4.5 * rad) ** 2)
        if ring.sum() < 30:
            continue
        z = depth[ring]
        X = np.stack([(uu[ring] - W / 2 + 0.5) / f * z, -(vv[ring] - H / 2 + 0.5) / f * z, -z], axis=1)
        c = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - c, full_matrices=False)       # local tangent plane of the flesh
        n = vt[2] if vt[2][2] > 0 else -vt[2]
        ray = np.array([(cx - W / 2 + 0.5) / f, -(cy - H / 2 + 0.5) / f, -1.0])
        t = (c @ n) / (ray @ n)                                    # seed centre ray -> flesh plane
        out.append(dict(p_cam=ray * t, n_cam=n, uv=(cx, cy), area=len(pix)))
    return out, seed


def survey(env, move=True, return_image=False):
    """Take the survey image and return (targets, info). targets: list of dict(p, n, id)."""
    s = env.cfg.sensors
    if move:
        p_goal, R = survey_pose(env)
        env.goto(p_goal, R, speed=0.08, timeout=10.0)
        for _ in range(40):
            env.tick()
    cam = env.m.camera("wrist").id
    mujoco.mj_forward(env.m, env.d)
    R_true = env.d.cam_xmat[cam].reshape(3, 3)
    t_true = env.d.cam_xpos[cam].copy()
    # what the robot believes about the camera pose: forward kinematics + hand-eye calibration error
    R_bel = R_true @ _small_rotation(env.handeye_rot)
    t_bel = t_true + R_true @ env.handeye_trans
    ren = _renderer(env)
    rgb = None
    if ren is not None:
        ren.update_scene(env.d, camera="wrist")
        rgb = ren.render().copy()
        ren.enable_depth_rendering()
        ren.update_scene(env.d, camera="wrist")
        depth = ren.render().copy()
        ren.disable_depth_rendering()
        depth = depth + env.rng.normal(0, s.depth_noise, depth.shape)
        dets, mask = detect_seeds(rgb, depth, s.cam_fovy)
    else:
        dets = _geometric_detections(env, R_true, t_true)
    pts = [t_bel + R_bel @ d["p_cam"] for d in dets]
    nrm = [R_bel @ d["n_cam"] for d in dets]
    targets, matched = _match_to_patch(env, pts, nrm)
    info = dict(n_detected=len(dets), n_matched=len(targets), n_true=env.n_targets, rendered=ren is not None,
                missed=[j for j in range(env.n_targets) if j not in matched])
    if return_image:
        info["rgb"] = rgb
        info["uv"] = [d.get("uv") for d in dets]
    return targets, info


def _small_rotation(rv):
    q = np.zeros(4); R = np.zeros(9)
    ang = np.linalg.norm(rv)
    mujoco.mju_axisAngle2Quat(q, rv / max(ang, 1e-12), ang)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _geometric_detections(env, R_cam, t_cam):
    """No image: project every visible seed with pixel + depth noise (same error budget)."""
    s = env.cfg.sensors
    f = 0.5 * s.cam_height / np.tan(np.radians(s.cam_fovy) / 2)
    out = []
    for j in range(env.n_targets):
        p, n = env.seed_surface_world(j)
        pc = R_cam.T @ (p - t_cam)
        nc = R_cam.T @ n
        if nc[2] < 0.3:
            continue
        z = -pc[2]
        pc = pc + np.r_[env.rng.normal(0, 0.6 * z / f, 2), env.rng.normal(0, s.depth_noise)]
        nc = nc + env.rng.normal(0, 0.03, 3)
        out.append(dict(p_cam=pc, n_cam=nc / np.linalg.norm(nc)))
    return out


def _match_to_patch(env, pts, nrm, tol=0.0012):
    """Keep detections that belong to the modelled target patch (only those seeds are simulated)."""
    targets, used = [], set()
    truth = [env.seed_surface_world(j)[0] for j in range(env.n_targets)]
    for p, n in zip(pts, nrm):
        d = [np.linalg.norm(p - q) for q in truth]
        j = int(np.argmin(d)) if d else -1
        if j >= 0 and d[j] < tol and j not in used:
            used.add(j)
            targets.append(dict(p=np.asarray(p), n=np.asarray(n) / np.linalg.norm(n), id=j))
    targets.sort(key=lambda t: t["id"])
    return targets, used


def localization_errors(env, targets):
    """Per-seed errors against ground truth: lateral [m], height along the true normal [m]
    (positive = true surface is higher than estimated), normal angle [rad]."""
    lat, hgt, ang = [], [], []
    for t in targets:
        p, n = env.seed_surface_world(t["id"])
        d = t["p"] - p
        h = d @ n
        lat.append(np.linalg.norm(d - h * n))
        hgt.append(-h)
        ang.append(np.arccos(np.clip(t["n"] @ n, -1, 1)))
    return np.array(lat), np.array(hgt), np.array(ang)
