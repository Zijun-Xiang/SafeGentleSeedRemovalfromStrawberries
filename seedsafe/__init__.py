"""Safe, gentle seed removal from strawberries: simulation, planning and safety filtering."""
import ctypes.util
import os
import sys

# Headless Linux: use EGL for off-screen rendering when it is installed. This must happen
# before mujoco is imported. Without EGL, rendering is skipped and the camera falls back
# to a geometric model (see perception.py).
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    if ctypes.util.find_library("EGL"):
        os.environ["MUJOCO_GL"] = "egl"
