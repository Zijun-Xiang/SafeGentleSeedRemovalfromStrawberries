"""Fast unit tests of the building blocks (no robot simulation needed)."""
import itertools

import numpy as np

from seedsafe.berry import ellipsoid_sdf, seed_layout, select_patch, surface_point
from seedsafe.conformal import conformal_quantile, coverage
from seedsafe.ilqr import ilqr
from seedsafe.safety import brake_speed, reach_value, solve_qp


def test_qp_returns_nominal_when_feasible():
    u = np.array([0.1, -0.2, 0.3, 0.0])
    A = np.array([[1.0, 0, 0, 0]]); b = np.array([1.0])
    assert np.allclose(solve_qp(u, np.ones(4), A, b), u)


def test_qp_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(40):
        u = rng.normal(size=3); w = rng.uniform(0.5, 2.0, 3)
        A = rng.normal(size=(4, 3)); b = rng.uniform(-0.2, 1.0, 4)
        x = solve_qp(u, w, A, b, iters=400)
        assert np.all(A @ x <= b + 1e-6)
        # brute force over active sets
        best = None
        for k in range(0, 4):
            for S in itertools.combinations(range(4), k):
                S = list(S)
                if S:
                    As, bs = A[S], b[S]
                    Hi = np.diag(1 / (2 * w))
                    lam = np.linalg.lstsq(As @ Hi @ As.T, As @ u - bs, rcond=None)[0]
                    cand = u - Hi @ As.T @ lam
                else:
                    cand = u
                if np.all(A @ cand <= b + 1e-7):
                    c = float(np.sum(w * (cand - u) ** 2))
                    best = c if best is None else min(best, c)
        if best is not None:
            assert np.sum(w * (x - u) ** 2) <= best + 1e-5


def test_conformal_coverage_holds():
    rng = np.random.default_rng(1)
    for alpha in (0.2, 0.1, 0.05):
        covs = []
        for _ in range(300):
            cal = rng.exponential(1.0, 60)
            q = conformal_quantile(cal, alpha)
            covs.append(coverage(rng.exponential(1.0, 200), q))
        assert np.mean(covs) >= 1 - alpha - 0.01


def test_ellipsoid_distance_and_normal():
    a = np.array([0.016, 0.021, 0.016])
    rng = np.random.default_rng(2)
    for _ in range(50):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        p = surface_point(d, a)
        dist, n = ellipsoid_sdf(p, a)
        assert abs(dist) < 1e-9
        for off in (-0.0005, 0.0008):
            dist2, _ = ellipsoid_sdf(p + off * n, a)
            assert abs(dist2 - off) < 0.06 * abs(off) + 2e-6      # first-order accurate near the surface


def test_seed_layout_and_patch():
    a = np.array([0.016, 0.021, 0.016])
    pts, nrm = seed_layout(a, 200, np.random.default_rng(3))
    assert len(pts) == 200
    assert np.allclose([ellipsoid_sdf(p, a)[0] for p in pts], 0, atol=1e-9)
    idx = select_patch(pts, a, 0.0085, 14)
    assert 6 <= len(idx) <= 14
    assert all(np.linalg.norm(pts[i] - [0, 0, a[2]]) <= 0.0085 for i in idx)


def test_ilqr_reaches_goal():
    goal = np.array([0.01, -0.02, 0.03, 0.4])
    res = lambda X, U: np.hstack([5 * (X - goal), 1.0 * U])
    res_T = lambda x: 10 * (x - goal)
    x = np.zeros(4); U = np.zeros((15, 4))
    for _ in range(150):
        U, X, _ = ilqr(x, U, res, res_T, 0.04, iters=1)
        x = x + 0.04 * U[0]
        U[:-1] = U[1:]
    assert np.linalg.norm(x - goal) < 1e-3


def test_brake_speed_is_boundary_of_reachable_set():
    v_touch, a, delay = 0.0015, 0.25, 0.03
    d, v, safe = reach_value(v_touch, a, delay)
    for i in (5, 40, 120, 200):
        vb = brake_speed(d[i], v_touch, a, delay)
        j = np.searchsorted(v, vb)
        assert safe[i, max(j - 2, 0)]                 # just below the curve: safe
        if j + 2 < len(v):
            assert not safe[i, j + 2]                 # just above: unsafe


def test_keyboard_teleop_mapping():
    import importlib.util, pathlib, time
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "teleop_keyboard.py"
    spec = importlib.util.spec_from_file_location("teleop_keyboard", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tele = mod.KeyTeleop(speed=0.006)
    assert np.allclose(tele.velocity(), 0)
    tele.key(mod.KEY["pgdn"]); tele.key(mod.KEY["left"])
    v = tele.velocity()
    assert v[2] < 0 and v[1] > 0 and abs(v[0]) < 1e-12
    tele.key(mod.KEY["shift"])
    assert np.linalg.norm(tele.velocity()) < 0.2 * np.linalg.norm(v)
    tele.key(ord("G")); assert tele.grip
    tele.key(ord("R")); assert not tele.grip
    tele.key(ord("T")); assert tele.events == ["T"]
    time.sleep(0.25)
    assert np.allclose(tele.velocity(), 0)        # keys released
