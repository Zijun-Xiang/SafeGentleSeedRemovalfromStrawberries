# Safe, Gentle Seed Removal from Strawberries: simulation report

*EN.601.497/697 Human-Centered Robotics, Fall 2026. Team: Kai Hua Liu, Zijun Xiang, Zixuan Zhu.*

This report documents the simulation part of the project, step by step, with a screenshot for
every step. Every image and number below is produced by the code in this repository
(`scripts/make_report_assets.py`, `scripts/run_experiments.py`, `scripts/make_figures.py`,
`scripts/make_report.py`). Nothing was run on a real robot.

## Summary

A 7-joint arm carries a Wuji Hand 2. The hand holds tweezers and closes them with thumb and
index finger. A wrist camera finds the seeds, a nominal planner removes them one by one, and a
safety filter overrides commands that would leave the safe set. All results are over
{n_berries} random berries per condition, 8 target seeds each ({n_seeds} seeds per condition).

| Question from the proposal | Short answer from the simulation |
|---|---|
| 1. Safe set: can a filter keep the arm and hand inside it? | Yes, in the nominal simulation. With the filter {filt_viol} of seeds had a violation and {filt_rem} were removed. A planner mis-tuned to press too deep and approach too fast violated the safe set on {st_off_viol} of seeds; behind the filter the same planner violated it on {st_on_viol} and removed {st_on_rem} (without the filter: {st_off_rem}). |
| 2. Tool use: can the hand hold and squeeze the tweezers, and what can it sense? | Thumb and index close the jaws through contact. From finger encoders alone the jaw gap is known to {gap_rms:.2f} mm (rms). From fingertip force the squeeze on a 1 mm object is known to {gauge_err:.3f} N (rms). |
| 3. Uncertainty: how conservative must the filter be? | {q3} |
| 4. Shared control: how to split authority? | Person picks the seed and steers sideways, robot controls depth and gripping, filter on: {sh_rv}. Person steering everything: {tel_rv} without the filter, {telf_rv} with it. |
| 5. Sim-to-real: does the simulation-calibrated safe set transfer? | {q5} |

Main limits of this work are listed at the end. The most important ones: the fruit is not a
finite-element body, the human operator is a model, and "sim-to-real" has not touched hardware.

## What from the proposal is implemented

| Proposal item | Status | Where |
|---|---|---|
| MuJoCo scene: arm, Wuji Hand 2, tweezers, compliant berry, seed targets | done | `seedsafe/scene.py`, `seedsafe/env.py` |
| Force and indentation damage measure | done | `seedsafe/contact.py` |
| Scripted grasp that holds the tweezers and closes the jaws | done | `scripts/design_grasp.py`, `seedsafe/hand.py` |
| Nominal planner for arm and fingers | done (iLQR-based MPC) | `seedsafe/ilqr.py`, `seedsafe/planner.py` |
| Safety filter | done (CBF constraints in a QP, reachability-based approach speed) | `seedsafe/safety.py` |
| Seed localization with uncertainty bounds | done (wrist camera + conformal prediction) | `seedsafe/perception.py`, `seedsafe/conformal.py` |
| Tweezer pose and tip-force estimation from the finger joints | done | `seedsafe/hand.py`, `seedsafe/estimation.py` |
| Experiments against an unfiltered baseline | done | `scripts/run_experiments.py` |
| Human operator, shared autonomy | done with a simulated operator; a keyboard teleoperation script exists for real people | `seedsafe/operator.py`, `scripts/teleop_keyboard.py` |
| Calibrate the tweezer grasp away from the berry | done in simulation | `seedsafe/hand.py` |
| Port to the physical arm, proxy berry, real strawberries | not done (simulation only, as agreed) | |

Abbreviations used below, with their full names:
MPC (Model Predictive Control, 模型预测控制), iLQR (iterative Linear Quadratic Regulator, 迭代线性二次型调节器),
CBF (Control Barrier Function, 控制屏障函数), QP (Quadratic Program, 二次规划), DoF (Degrees of Freedom, 自由度),
MJCF (MuJoCo XML model Format, MuJoCo 模型文件格式), rms (root mean square, 均方根).
A glossary of the technical terms (English / 中文) is at the end.

---

## Step 1. The scene

**What was done.** The official Wuji Hand model (20 DoF) is mounted on a Flexiv Rizon 4 arm
model (7 DoF) through an angled adapter. The Rizon 4 stands in for the Flexiv Enlight because no
public Enlight model exists. A pair of tweezers (two elastic arms joined at the base, 140 mm
long) sits in the hand. A strawberry with about 200 seeds rests on a soft cradle; a tray takes
the removed seeds. The model has {nq} position coordinates and {nu} motors.

![scene](screenshots/01_scene_overview.png)

The hand holding the tweezers, from two sides. Thumb and index finger pinch the two tweezer
arms. The other three fingers wrap the printed sleeve (blue). The black box is the wrist camera.

![hand](screenshots/02_hand_holding_tweezers.png)

Left: the berry. Right: what the wrist camera sees from its survey position.

![berry](screenshots/03_berry_and_wrist_camera.png)

Modelling choices that matter:

- **Fingers on the tweezers.** Thumb and index press the tweezer arms through simulated contacts.
  The pinch pose and a table "closure command c -> finger joint targets" are designed once by
  `scripts/design_grasp.py`. The stock finger position gains are very soft, so the two pinching
  fingers use higher gains, and the thumb is made as stiff as the index along the pinch axis.
- **Sleeve in the hand.** The sleeve is held by a stiff seat that slips when the load exceeds
  the grip friction (6 N or 0.15 N m). Slip stays; the robot does not see it. This is the "tool
  must not slip" part of the safe set.
- **Arm.** Position-controlled joints with a task-space controller on top. The stock position
  gains of the arm model (about 1.5 Hz) are raised to about 6 Hz.

## Step 2. Tweezer calibration (away from the berry)

**What was done.** This is the first hardware step of the proposal, done here in simulation.
Two sweeps, run once per session:

1. *Free sweep.* Close the empty jaws slowly. Record the jaw gap and the thumb-index distance
   computed from the finger encoders. The jaws meet at closure c = {c_closed:.2f}.
2. *Gauge sweep.* Close on a rigid 1 mm ball and record the squeeze force.

After that the robot estimates the jaw gap from finger encoders (rms error {gap_rms:.3f} mm) and
the squeeze force from the fingertip force sensors (rms error {gauge_err:.3f} N on the gauge).
The sweep also showed that the two arms do not close evenly: the midpoint between the tips
drifts sideways by up to {shift_max:.2f} mm. That drift is stored and compensated. Without the
compensation the tips landed next to the seeds.

![calibration](screenshots/04_tweezer_calibration.png)

Reproduce: `python scripts/make_report_assets.py calibration`

## Step 3. Contact model, safe set and damage measure

**What was done.** MuJoCo simulates the arm, the hand, the tweezers and the finger contacts.
The millimetre-scale contact at the two tweezer tips is computed by `seedsafe/contact.py`,
because the safe set is defined on exactly these quantities:

- *tip on flesh*: normal force `F = k * depth^1.5` (a Hertz-type law with damping) plus a shear
  spring that slides at the friction limit;
- *tip on seed*: stiff contact against a 1 mm sphere; the tips are flat pads 1.2 mm wide;
- *seed in its pit*: a spring that lets go at the pull-out force (about 0.08 N). The seed centre
  sits 0.35 mm below the surface, so the tips must press slightly into the flesh to grip it.

**Safe set** (per seed, judged on ground truth):

| Quantity | Limit |
|---|---|
| tip indentation | {max_indent:.1f} mm |
| normal force per tip | {max_force:.2f} N |
| shear force per tip | {max_shear:.2f} N |
| jaw squeeze on the seed | {max_squeeze:.2f} N |
| tool slip inside the hand | {max_slip:.1f} mm |

Indentation beyond {vis_indent:.1f} mm or force beyond {vis_force:.1f} N counts as *visible damage*.

The figure shows the contact law (left) and what the same bad command does without and with
the safety filter (middle, right): "push 2 mm into the berry at 4 mm/s". Without the filter the
tips went {press_off_ind:.1f} mm deep ({press_off_f:.1f} N). With the filter they stopped at
{press_on_ind:.2f} mm ({press_on_f:.2f} N).

![contact](screenshots/05_contact_model_and_filter.png)

Reproduce: `python scripts/make_report_assets.py contact`

## Step 4. Seed localization with the wrist camera

**What was done.** The camera moves about 10 cm above the berry and takes one colour and one
depth image. Yellow blobs are seeds. For each seed the flesh around it gives the surface point
and the surface normal. Modelled error sources: pixel size, depth noise, hand-eye calibration
error, shading.

![detection](screenshots/07_seed_detection.png)

On 12 random berries ({per_n} seeds): sideways error {lat_mean:.2f} mm on average
({lat_p90:.2f} mm at the 90th percentile), surface height error {hgt_mean:+.2f} mm on average
({hgt_p90:.2f} mm at the 90th percentile), normal direction error {ang_mean:.1f} degrees.
The height error has a bias: the camera places the surface a little too low.

![errors](screenshots/08_localization_errors.png)

Only the seeds inside a fixed patch on top of the berry are physically simulated and targeted
(the proposal's "fixed patch of seeds").

Reproduce: `python scripts/make_report_assets.py perception`

## Step 5. Nominal planner and one seed from start to finish

**What was done.** The planner is a phase machine: move above the seed, descend, close the jaws,
pull, carry to the tray, release. Inside each phase the motion comes from MPC: an iLQR solver
plans 0.5 s ahead on a small model (tip position and jaw closure), 50 times per second, with
costs for reaching the goal, speed limits and staying clear of the berry between seeds.

The planner does not know the safe set. Two details make it gentle anyway:

- *Touch-referenced depth.* The camera's surface height is not accurate enough
  (Step 4). The planner descends slowly until the force estimate reports touch, and measures
  the gripping depth from there.
- *Force estimate.* Tip force is estimated from the arm's joint torque sensors and zeroed in
  free space before each descent.

The timeline below shows one seed with the filtered system: {single_time:.1f} s, peak
indentation {single_ind:.2f} mm, peak tip force {single_f:.3f} N, peak squeeze {single_sq:.2f} N.
Blue is ground truth, grey is what the robot estimates, orange is the upper bound the filter uses.

![timeline](screenshots/09_single_seed_timeline.png)

![filmstrip](screenshots/10_filmstrip.png)

Reproduce: `python scripts/make_report_assets.py single`

## Step 6. Safety filter

**What was done.** The filter sits between the planner (or the human) and the robot. Every
5 ms it solves a small QP: stay as close as possible to the commanded tip velocity and jaw
closing rate, subject to one CBF constraint per limit of the safe set.

| Constraint | Uses | Effect |
|---|---|---|
| indentation | upper bound on indentation (kinematics + touch, capped by what the measured force allows) | slows and stops the descent |
| normal force | force estimate + error bound, upper bound on flesh stiffness | slows, stops, or backs off |
| approach speed | distance to where contact becomes possible | tip arrives slowly enough to stop on touch |
| drag | "contact possible" flag | limits sideways motion while touching |
| grip stability | sideways force estimate vs. what the sleeve grip can hold | prevents tool slip |
| jaw squeeze | squeeze estimate + error bound | limits jaw closing |

The approach-speed limit is the boundary of a reachable set: the largest speed from which the
tip can still slow down to touch speed before contact becomes possible, given a reaction delay
and a braking limit. The figure shows the set computed on a grid and the closed-form boundary
the filter uses.

![reachability](screenshots/06_reachability_approach.png)

## Step 7. Uncertainty bounds from conformal prediction

**What was done.** The filter needs error bounds for its estimates. They come from split
conformal prediction: run calibration berries, record the errors against ground truth, and take
the quantile that gives the wanted coverage. {conf_n} seeds on {conf_berries} calibration
berries were used. At 90% coverage the bounds are: surface height before touch
{b_surface:.2f} mm, indentation after touch {b_indent:.2f} mm, sideways seed position
{b_lateral:.2f} mm, tip force {b_force:.3f} N, squeeze {b_squeeze:.3f} N.

![conformal](screenshots/11_conformal_calibration.png)

The sideways bound also sets how far the jaws are opened before descending, so that the seed is
surely between the tips.

---

## Step 8. Experiments

Each condition: {n_berries} random berries (size, stiffness, seed layout, pull-out force, sensor
biases and calibration errors all vary), 8 target seeds per berry. "Violation" means that at
least one limit of the safe set was exceeded while working on that seed. Error bars are 95%
Wilson intervals.

### E1. Filtered system against unfiltered baselines

![e1](figures/e1_filter_vs_baseline.png)

{t1}

Findings:

- **Camera depth only, no filter:** {open_rv}; {open_vis} spots with visible damage. Relying on
  the camera for depth is not good enough.
- **Touch-referenced depth, no filter:** {touch_rv}. With a well-tuned planner and nominal
  conditions the planner alone stayed inside the safe set.
- **With the safety filter:** {filt_rv}. The filter changed {filt_interv} of the commanded
  motion and cost {filt_time:.1f} s instead of {touch_time:.1f} s per seed. No seeds were lost
  to the filter.

So in nominal conditions the filter is not what makes the system gentle; the touch-referenced
planner already is. The filter matters when the command is wrong, which the next test shows.

### E1b. A badly tuned planner, with and without the filter

The planner is told to go twice as deep (0.8 mm instead of 0.4 mm) and to approach four times
faster (16 mm/s instead of 4 mm/s) than it should.

![e1b](figures/e1b_stress.png)

{t1b}

Without the filter: {st_off_viol} of seeds with a violation, deepest indentation
{st_off_ind:.2f} mm, {st_off_vis} spots with visible damage, {st_off_rem} of seeds removed
(pressing too deep also makes the tips miss seeds). Behind the filter, with the 90% bounds:
{st_on_viol} of seeds with a violation ({st_on_nviol} of {st_on_n}), deepest indentation
{st_on_ind:.2f} mm against the limit of {max_indent:.1f} mm, no visible damage,
{st_on_rem} of seeds removed. The filter changed {st_on_interv} of the commanded motion and the
time per seed went from {st_off_time:.1f} s to {st_on_time:.1f} s.

So the filter turned a planner that damages every seed site into one that almost never leaves
the safe set. It is not a guarantee: one seed in this test went slightly past the limit.

### E2. How conservative must the filter be?

The same system with uncertainty margins from conformal bounds at different coverage levels,
and with no margins at all.

![e2](figures/e2_uncertainty.png)

{t2}

Bounds used:

{bounds_table}

Coverage of the bounds on the test berries (they were calibrated on other berries):

![coverage](figures/e2_coverage.png)

Findings with the well-tuned planner:

- No margins: {nm_rv}. 50% bounds: {a50_rv}. 90% bounds: {a90_rv}. 99% bounds: {a99_rv}.
- Nothing changes except a little time. The reason is E1: the well-tuned planner is already
  gentle, so the filter rarely acts and its margins hardly matter.

To see what the margins do, the same sweep was repeated with the mis-tuned planner of E1b. Now
the planner always asks for too much, and the filter decides how deep the tips go.

![e2b](figures/e2b_margins.png)

{t2b}

Findings when the filter decides the depth:

- **No margins:** {m0_rv}. The filter trusts its estimates exactly, and the estimates are a
  little off, so the tips often end up beyond the limit.
- **50% bounds:** {m50_rv}.
- **90% bounds:** {m90_rv}.
- **99% bounds:** {m99_rv}. This was the best setting in this test.
- **Three times the 99% bounds:** {m3_rv}. The force margin alone ({m3_force:.2f} N) is then
  larger than the force limit ({max_force:.2f} N), so the filter never allows contact and every
  seed runs into the 40 s time limit.

Two cautions for reading the seed-removal numbers. First, the sideways bound also sets how far
the jaws open before the descent. The 50% setting opens them less ({m50_lat:.2f} mm bound
instead of {m99_lat:.2f} mm at 99%), which by itself causes more misses. Second, each setting
has 64 seeds on 8 berries, and failures cluster on single berries, so differences of 10 to 15
percentage points in removal between neighbouring settings are within the noise. The trend in
violations (from {m0_viol} down to {m99_viol}) is the reliable part.

### E3. Shared control: who commands what?

A simulated person steers the tip with a joystick model: reaction delay 0.2 s, tremor, motor
noise, and depth judgement that is off by about 0.45 mm (standard deviation). Gripping is
always automatic after the person presses "grip".

![e3](figures/e3_shared_control.png)

{t3}

Findings:

- **Person steers everything, no filter:** {tel_rv}; {tel_vis} spots with visible damage. People
  judge depth poorly, so they press too deep.
- **Person steers everything, with filter:** {telf_rv}; {telf_vis} spots with visible damage.
  Violations drop from {tel_viol} to {telf_viol}, and {telf_auth} of the person's command is
  executed, but most seeds are missed. In the missed seeds the tips were deep enough (median
  {telf_miss_ind:.2f} mm), yet no squeeze on the seed was recorded: the jaws did not close on
  it. The cause was not traced. The {telf_nviol} seeds that still had a violation (tip force up
  to {telf_maxf:.2f} N) were not traced either. A filter on top of full manual control was
  safer than no filter, but it was not a working system.
- **Person chooses the seed and steers sideways, robot controls depth, with filter:** {sh_rv};
  {sh_auth} of the person's command is executed.
- **Fully automatic with filter:** {auto_rv}.

The split that worked: the person keeps the decisions and the sideways motion; the robot keeps
depth, holding the tweezers and closing the jaws. The operator here is a model, so these
numbers say what the control structure does, not how real people will perform.

### E4. Transfer to a shifted domain ("sim-to-real" in simulation)

There is no hardware in this project, so transfer is tested against a second simulation that
the robot was not calibrated in: flesh stiffness x0.42, seed friction 0.5 instead of 0.6,
pull-out force x1.15, softer cradle, torque and fingertip sensor noise x1.5, depth noise x1.6,
tool and hand-eye calibration errors x1.5 to x1.6.

- **A: simulation calibration only.** Model, thresholds and bounds from the nominal simulation.
- **B: + noise and stiffness probes.** Before the task the robot measures its force noise at
  rest (this sets the touch threshold) and presses gently on three seed-free spots to fit the
  flesh stiffness.
- **C: + surface re-referencing.** The same presses also show where the surface really is.
  That corrects the height bias of the camera.
- **D: + new conformal bounds.** C, plus bounds recomputed from 4 calibration berries in the
  shifted domain.

![e4](figures/e4_sim_to_real.png)

{t4}

{bounds4}

Results: reference in the nominal simulation: {ref_rv}. A: {A_rv}. B: {B_rv}. C: {C_rv}.
D: {D_rv}.

{e4_reading}

---

## Step 9. Demo

`demo/seed_removal_demo.mp4` shows the filtered system removing six seeds from one berry, with
live gauges for indentation, tip force and jaw squeeze against their limits. Key frames:

![demo1](../demo/seed_removal_demo_descend.png)
![demo2](../demo/seed_removal_demo_pull.png)

Reproduce: `python scripts/make_demo.py`

## Step 10. Tests, and running on the laptop

`python -m pytest -q tests` runs 12 tests: the QP solver against brute force, conformal
coverage, the berry geometry, iLQR, the reachable-set boundary, the tweezer calibration, "the
filter stops a deep press", and a short filtered run that must remove seeds with no violation.

![tests](screenshots/12_tests.png)

The same tests were also run on the team laptop in the Linux environment that the assistant can
use there (Python 3.10, MuJoCo 3.17, no GPU rendering): 12 passed. The experiments and the
video were produced on a cloud machine (Python 3.13, MuJoCo 4.0) and copied to the repository.

---

## Limits and what is simplified

- **No hardware.** No real arm, hand, tweezers or fruit were used. The arm model is a Rizon 4,
  not an Enlight.
- **Fruit model.** The berry is a rigid ellipsoid on a soft cradle. Deformation exists only as
  the contact law at the two tips. There is no finite-element flesh, no pits, no tearing.
  Stiffness, pull-out force and damage limits are plausible guesses, not measurements.
- **Sleeve grip.** The sleeve is held by a friction-limited seat, not by simulated finger
  contacts. Thumb and index contacts are simulated.
- **Sensing.** Tip force comes from joint torques with white noise and a slow bias; real torque
  sensing has friction and model errors that are not simulated. The noise level assumed here
  (about 0.01 N at the tip after filtering) is optimistic.
- **Human operator.** A model with delay, noise and depth error. No person took part.
  `scripts/teleop_keyboard.py` is ready for real trials but was not tested with a display.
- **Open questions in the results.** Three things were observed and not explained: why the
  jaws did not close on the seed when a person steers everything behind the filter (E3), which
  of the recalibrated bounds restored seed removal in the shifted domain (E4), and the few
  violations that the filter did not prevent (1 of 64 in E1b, 4 of 64 in E3, 1 of 64 in E4 D).
- **Statistics.** Each condition was run once. 64 seeds per condition. Rates near 0% or 100% have 95% intervals of about
  0 to 6%.
- **Conformal bounds.** Calibration scores were collected with one filter setting; changing the
  margins changes the robot's behaviour a little, so the coverage guarantee is approximate.
  Seeds on the same berry share errors and are not fully independent.

## Glossary (English / 中文)

| Term | 中文 | Meaning here |
|---|---|---|
| achene | 瘦果 | the "seed" on the outside of a strawberry |
| end-effector | 末端执行器 | what the arm carries: here the hand with tweezers |
| degrees of freedom (DoF) | 自由度 | number of independent joints |
| forward kinematics | 正运动学 | computing where the tip is from the joint angles |
| Jacobian | 雅可比矩阵 | maps joint speeds to tip speed, and tip force to joint torques |
| indentation | 压入深度 | how far a tip is pressed into the flesh |
| safe set | 安全集 | all states in which every damage limit holds |
| safety filter | 安全滤波器 | module that changes a command as little as possible to stay in the safe set |
| control barrier function (CBF) | 控制屏障函数 | a function that is positive inside the safe set; the filter keeps it from decreasing too fast |
| quadratic program (QP) | 二次规划 | small optimization problem solved by the filter every 5 ms |
| reachability / reachable set | 可达性 / 可达集 | the states from which a failure can no longer be avoided (or can still be avoided) |
| model predictive control (MPC) | 模型预测控制 | plan a short time ahead, execute the first step, plan again |
| iterative LQR (iLQR) | 迭代线性二次型调节器 | the optimizer used inside the MPC |
| nominal planner | 名义规划器 | the planner that pursues the task and ignores the safe set |
| conformal prediction | 保形预测 | turns calibration errors into bounds with a coverage guarantee |
| coverage | 覆盖率 | share of new cases in which the bound holds |
| hand-eye calibration | 手眼标定 | where the camera sits relative to the robot |
| shared autonomy / shared control | 共享自主 / 共享控制 | person and robot command the motion together |
| teleoperation | 遥操作 | a person steers the robot directly |
| sim-to-real gap | 仿真到现实差距 | what works in simulation may behave differently on hardware |
| Hertz contact | 赫兹接触 | contact law in which force grows with depth to the power 1.5 |
| tare | 去皮 / 清零 | setting the force reading to zero in free space |
| back-drivable | 可反向驱动 | a joint that outside forces can move, so its torque can be sensed |

Last test run: `{tests_line}`
