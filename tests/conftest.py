import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import seedsafe  # noqa: E402,F401  (selects the rendering backend before mujoco is imported)
