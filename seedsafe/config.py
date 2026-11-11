"""All tunable numbers of the project, in SI units, in one place."""
from dataclasses import dataclass, field, replace
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


@dataclass
class TweezerParams:
    length: float = 0.140            # base -> tip [m]
    tip_half_gap_rest: float = 0.004 # half jaw gap at rest [m]
    close_force: float = 0.6         # force at the pinch point that closes the jaws [N]
    arm_mass: float = 0.006          # one arm [kg]
    sleeve_mass: float = 0.015       # printed sleeve + base [kg]
    arm_damping: float = 0.004       # hinge damping [N m s/rad]
    tip_radius: float = 0.00015      # contact radius of one tip [m]
    pad_half_width: float = 0.0006   # half width of the flat tip pad [m]
    pinch_gain_scale: float = 3.0    # thumb/index position gain relative to the stock model
    thumb_gain_extra: float = 2.2    # makes thumb and index equally stiff along the pinch axis
    # grip of the sleeve in the hand: stiff elastic seat that slips (plastically)
    # when the load exceeds the friction the wrapping fingers can provide
    seat_friction_lin: float = 6.0   # force at which the sleeve slides [N]
    seat_friction_rot: float = 0.150 # torque at which the sleeve twists [N m]
    seat_stiff_lin: float = 2.0e4    # [N/m]
    seat_stiff_rot: float = 80.0     # [N m/rad]


@dataclass
class BerryParams:
    center: tuple = (0.60, 0.0, 0.330)          # world position of the berry centre [m]
    semi_axes: tuple = (0.016, 0.021, 0.016)     # ellipsoid half sizes x, y, z [m]
    hertz_k: float = 1.0e4      # flesh stiffness, F = k * delta^1.5 [N/m^1.5]
    hunt_d: float = 30.0        # Hunt-Crossley damping factor [s/m]
    shear_k: float = 250.0      # flesh shear stiffness under a tip [N/m]
    mu_flesh: float = 0.5       # tip-flesh friction
    cradle_k: float = 2500.0    # stiffness of the soft cradle that holds the berry [N/m]
    mass: float = 0.020
    n_seeds: int = 200          # seeds on the whole berry
    patch_radius: float = 0.0085  # target patch: seeds this close to the top point [m]
    max_targets: int = 14
    seed_radius: float = 0.0005
    seed_depth: float = 0.00035   # seed centre below the flesh surface (shallow pit) [m]
    seed_k: float = 2000.0        # tip-seed contact stiffness [N/m]
    mu_seed: float = 0.6          # tip-seed friction
    seat_k: float = 250.0         # stiffness holding a seed in its pit [N/m]
    pull_force: float = 0.08      # force that pulls a seed out [N]
    grip_cone: float = 0.5        # tips must touch the seed within asin(0.5)=30 deg of its equator


@dataclass
class SafeSet:
    """Limits that define 'do not damage the berry' (and 'keep the tool in hand')."""
    max_indent: float = 0.0006     # per tip [m]
    max_tip_force: float = 0.15    # per tip, normal [N]
    max_shear: float = 0.08        # per tip, tangential [N]
    max_squeeze: float = 0.45      # jaw force on whatever is between the tips [N]
    max_slip: float = 0.0004       # tool tip displacement inside the hand [m]
    visible_indent: float = 0.0010 # beyond this the flesh is visibly punctured [m]
    visible_force: float = 0.30


@dataclass
class SensorParams:
    torque_noise: float = 0.035    # arm joint torque sensor noise, per 1 kHz sample [N m]
    torque_bias: float = 0.004     # slowly varying torque bias [N m]
    force_lpf_hz: float = 10.0     # low-pass of the tip force estimate
    finger_pos_noise: float = 0.0006   # finger encoder noise [rad]
    fingertip_force_noise: float = 0.02  # fingertip force sensor noise per sample [N]
    cam_width: int = 640
    cam_height: int = 480
    cam_fovy: float = 32.0
    depth_noise: float = 0.00025   # depth noise std [m]
    handeye_trans: float = 0.0003  # hand-eye calibration error std [m]
    handeye_rot: float = 0.004     # [rad]
    tool_calib_err: float = 0.00015  # error of the calibrated tip offset [m]


@dataclass
class ControlParams:
    dt: float = 0.001              # physics step [s]
    ctrl_decim: int = 5            # control every 5 steps (200 Hz)
    plan_decim: int = 4            # planner every 4 control ticks (50 Hz)
    v_max: float = 0.08            # tip speed limit in free space [m/s]
    v_contact: float = 0.004       # speed limit of the nominal plan near the berry [m/s]
    jaw_rate_max: float = 0.012    # jaw gap rate limit [m/s]
    clik_gain: float = 12.0        # task-space feedback gain [1/s]
    rot_gain: float = 6.0
    w_max: float = 0.9             # tool re-orientation speed limit [rad/s]
    pre_height: float = 0.008      # hover height above a seed [m]
    pre_gap: float = 0.0024        # jaw gap while descending over a seed [m]
    target_indent: float = 0.00040 # nominal tip depth for gripping [m]
    grip_force: float = 0.24       # desired squeeze on the seed [N]
    retract: float = 0.007
    search_depth: float = 0.0008   # how far below the camera surface estimate to search for touch [m]
    max_attempts: int = 2


@dataclass
class FilterParams:
    enabled: bool = True
    alpha: float = 12.0            # CBF class-K gain [1/s]
    eps_surface: float = 0.0006    # bound on surface-height error before touch [m]
    eps_indent: float = 0.00012    # bound on indentation estimate error after touch [m]
    eps_force: float = 0.03        # bound on force estimate error [N]
    eps_squeeze: float = 0.05
    eps_lateral: float = 0.0006    # bound on lateral seed position error [m]
    k_upper: float = 1.6e4         # upper bound on flesh stiffness [N/m^1.5]
    force_share: float = 0.85      # allowed total force = 2 * max_tip_force * share
    touch_force: float = 0.05     # touch is declared above this total normal force [N]
    v_touch: float = 0.0015        # max approach speed where contact is possible [m/s]
    a_brake: float = 0.25          # braking deceleration assumed for the approach [m/s^2]
    delay: float = 0.03            # sensing + actuation delay [s]
    v_drag: float = 0.0006         # max sideways tip speed while (possibly) in contact [m/s]


@dataclass
class Config:
    tweezers: TweezerParams = field(default_factory=TweezerParams)
    berry: BerryParams = field(default_factory=BerryParams)
    safe: SafeSet = field(default_factory=SafeSet)
    sensors: SensorParams = field(default_factory=SensorParams)
    control: ControlParams = field(default_factory=ControlParams)
    filter: FilterParams = field(default_factory=FilterParams)
    tray_pos: tuple = (0.55, -0.10, 0.320)
    seed: int = 0

    def copy(self, **kw):
        return replace(self, **kw)
