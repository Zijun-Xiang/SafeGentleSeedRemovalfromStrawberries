"""System tests in the simulator (about a minute)."""
import numpy as np
import pytest

from seedsafe.config import Config
from seedsafe.env import frame_from_axis
from seedsafe.runner import Stepper, make_env, run_berry, true_targets


@pytest.fixture(scope="module")
def env():
    return make_env(Config(), seed=11)


def test_scene_and_calibration(env):
    assert env.m.nu == 27 and env.m.nq == 38          # 7 arm + 20 hand actuators
    cal = env.jaw_cal
    assert 0.3 < cal.c_closed < 0.7
    assert cal.gauge_err < 0.03                        # squeeze estimate within 0.03 N on the gauge
    assert env.tool_slip() < 1e-5                      # calibration does not move the tool in the hand


def test_jaw_gap_estimate(env):
    env.command(np.zeros(3), -1.0)
    while env.closure > 0.0:
        env.tick()
    err = []
    env.command(np.zeros(3), 0.2)
    while env.closure < env.jaw_cal.c_closed - 0.02:
        env.tick()
        err.append(env.obs["gap"] - env.jaw_gap())
    env.command(np.zeros(3), -1.0)
    while env.closure > 0.2:
        env.tick()
    assert np.sqrt(np.mean(np.square(err))) < 0.0002   # 0.2 mm rms from finger encoders alone


def _press(cfg_filter_on, depth=0.002):
    """Nominal command: push the tips 2 mm into the berry at 4 mm/s (a gross planning error)."""
    cfg = Config()
    cfg.filter.enabled = cfg_filter_on
    env = make_env(cfg, seed=12)
    tg = true_targets(env)
    P = np.array([t["p"] for t in tg]); n = np.array([0.0, 0.0, 1.0])
    spot = P.mean(axis=0) + np.array([0.0012, 0.0012, 0])
    top = env.berry_center() + np.array([0, 0, env.bp.semi_axes[2]])
    spot[2] = top[2]
    R = frame_from_axis(-n, (0.0, 1.0, 0.0))
    env.goto(spot + n * 0.006, R, closure_rate=0.5)
    env.tare_force()
    st = Stepper(env, cfg)
    st.est.reset(spot, n); st.est.armed = True
    env.contact.reset_stats()
    for _ in range(int((0.006 + depth) / 0.004 / env.dt_ctrl) + 200):
        st.step(np.r_[-n * 0.004, 0.0], R)
    return env.contact.peak, env.cfg.safe


def test_filter_stops_a_deep_press():
    peak_off, safe = _press(False)
    peak_on, _ = _press(True)
    assert peak_off["indent"] > safe.visible_indent            # without the filter: visible damage
    assert peak_on["indent"] <= safe.max_indent                # with it: inside the safe set
    assert peak_on["force"] <= safe.max_tip_force


def test_filtered_pipeline_removes_seeds_safely():
    out = run_berry(21, "filtered", max_seeds=3)
    s = out["summary"]
    assert s["violation_rate"] == 0.0
    assert s["removal_rate"] >= 2 / 3
    assert s["visible_damage"] == 0
