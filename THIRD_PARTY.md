# Third-party assets

| Folder | What | Source | License |
|---|---|---|---|
| `assets/flexiv_rizon4/` | Flexiv Rizon 4 arm model (MJCF + meshes) | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flexiv_rizon4), commit `370c439` | Apache-2.0 (see `assets/flexiv_rizon4/LICENSE`) |
| `assets/wuji_hand/` | Wuji Hand right-hand model (MJCF + meshes) | [wuji-technology/wuji-hand-description](https://github.com/wuji-technology/wuji-hand-description), commit `bbc2f64` | MIT (see `assets/wuji_hand/LICENSE`) |

Changes made when the scene is built (`seedsafe/scene.py`): the two models are merged into one
MJCF; the arm's collision geometry is switched off; the arm position gains are raised (x16 / x4);
the thumb and index position gains are raised; an angled adapter, the tweezers, a camera and the
strawberry are added. The files in the folders above are unmodified.

No public MuJoCo model of the Flexiv Enlight was found, so the Rizon 4 (same maker, 7 joints,
joint-torque sensing) stands in for it.
