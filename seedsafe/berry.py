"""Strawberry geometry: an ellipsoid with ~200 seeds, and a target patch on top."""
import numpy as np


def ellipsoid_sdf(p, a):
    """First-order signed distance of point(s) p (berry frame) to the ellipsoid
    with half sizes a. Accurate to a few percent within ~2 mm of the surface.
    Returns (distance, outward unit normal)."""
    p = np.asarray(p, float)
    q = p / a
    k0 = np.linalg.norm(q, axis=-1)
    g = p / (a * a)
    k1 = np.linalg.norm(g, axis=-1)
    dist = k0 * (k0 - 1.0) / k1
    n = g / np.expand_dims(k1, -1)
    return dist, n


def surface_point(direction, a):
    """Point where the ray from the centre along `direction` leaves the ellipsoid."""
    d = np.asarray(direction, float)
    t = 1.0 / np.linalg.norm(d / a, axis=-1)
    return d * np.expand_dims(t, -1)


def surface_normal(p, a):
    g = np.asarray(p, float) / (a * a)
    return g / np.linalg.norm(g, axis=-1, keepdims=True)


def seed_layout(a, n_seeds, rng, jitter=0.15):
    """Seeds on a Fibonacci spiral with jitter. Returns (points, normals) in the berry frame."""
    i = np.arange(n_seeds) + 0.5
    z = 1 - 2 * i / n_seeds
    r = np.sqrt(1 - z * z)
    phi = i * np.pi * (3 - np.sqrt(5)) + rng.uniform(0, 2 * np.pi)
    dirs = np.stack([r * np.cos(phi), z, r * np.sin(phi)], -1)   # spiral axis = berry long axis (y)
    spacing = np.sqrt(4 * np.pi / n_seeds)
    dirs += rng.normal(0, jitter * spacing, dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    pts = surface_point(dirs, a)
    return pts, surface_normal(pts, a)


def select_patch(pts, a, patch_radius, max_targets, min_spacing=0.0022):
    """Indices of the seeds inside the fixed target patch around the top point.
    Seeds closer than `min_spacing` to an already chosen one are skipped."""
    top = np.array([0.0, 0.0, a[2]])
    d = np.linalg.norm(pts - top, axis=-1)
    chosen = []
    for i in np.argsort(d):
        if d[i] > patch_radius or len(chosen) >= max_targets:
            break
        if all(np.linalg.norm(pts[i] - pts[j]) >= min_spacing for j in chosen):
            chosen.append(int(i))
    return chosen
