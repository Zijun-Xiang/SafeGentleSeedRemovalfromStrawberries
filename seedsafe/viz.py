"""Rendering helpers: still images, heads-up display, video files."""
import shutil
import subprocess

import mujoco
import numpy as np

_R = dict(key=None, model=None, ren=None)


def renderer(env, width=960, height=720):
    """Small cache: renderers are released when the model (berry) changes."""
    if _R["model"] is not env.m:
        for r in (_R["ren"] or {}).values():
            r.close()
        _R["model"], _R["ren"] = env.m, {}
    if (width, height) not in _R["ren"]:
        _R["ren"][(width, height)] = mujoco.Renderer(env.m, height, width)
    return _R["ren"][(width, height)]


def render(env, camera="overview", width=960, height=720):
    """camera: a camera name or an mujoco.MjvCamera."""
    r = renderer(env, width, height)
    r.update_scene(env.d, camera=camera)
    return r.render().copy()


def tracking_camera(env, distance=0.075, azimuth=128.0, elevation=-18.0, lookat=None):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = env.tcp_true() if lookat is None else lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def hud(img, lines, bars=(), origin=(14, 12), scale=1.0):
    """Draw text lines and labelled bars on an image. bars: list of (label, value, limit, unit)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(17 * scale))
        small = ImageFont.truetype("DejaVuSans.ttf", int(14 * scale))
    except Exception:
        font = small = ImageFont.load_default()
    x, y = origin
    h_text = int(24 * scale)
    w_box = int(392 * scale)
    d.rectangle([x - 8, y - 6, x + w_box, y + h_text * len(lines) + int(30 * scale) * len(bars) + 8], fill=(255, 255, 255, 205))
    for ln in lines:
        d.text((x, y), ln, fill=(20, 20, 20, 255), font=font)
        y += h_text
    for label, value, limit, unit in bars:
        frac = float(np.clip(value / (1.25 * limit), 0, 1))
        w = int(190 * scale)
        d.text((x, y), label, fill=(20, 20, 20, 255), font=small)
        bx = x + int(105 * scale)
        d.rectangle([bx, y + 3, bx + w, y + int(15 * scale)], outline=(90, 90, 90, 255), fill=(235, 235, 235, 255))
        col = (38, 140, 70, 255) if value <= 0.8 * limit else (225, 150, 20, 255) if value <= limit else (205, 40, 40, 255)
        d.rectangle([bx, y + 3, bx + int(w * frac), y + int(15 * scale)], fill=col)
        lx = bx + int(w / 1.25)
        d.line([lx, y, lx, y + int(18 * scale)], fill=(0, 0, 0, 255), width=2)          # the limit
        d.text((bx + w + 6, y), f"{value:.2f} {unit}", fill=(20, 20, 20, 255), font=small)
        y += int(30 * scale)
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def inset(img, small, pos="tr", margin=12, border=3):
    out = img.copy()
    h, w = small.shape[:2]
    H, W = img.shape[:2]
    x0 = W - w - margin if "r" in pos else margin
    y0 = margin if "t" in pos else H - h - margin
    out[y0 - border:y0 + h + border, x0 - border:x0 + w + border] = 255
    out[y0:y0 + h, x0:x0 + w] = small
    return out


def write_video(frames, path, fps=30):
    """Write an .mp4 (needs ffmpeg, either on the PATH or from the imageio-ffmpeg package)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if exe is None:
        raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg")
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def write_gif(frames, path, fps=12, width=480):
    from PIL import Image
    ims = []
    for f in frames:
        im = Image.fromarray(f)
        im = im.resize((width, int(im.height * width / im.width)))
        ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
