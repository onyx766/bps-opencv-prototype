"""
BPS prototype - where the six pockets are, clicked once by hand.

Why by hand. A pocket is the hardest thing on the table to find automatically:
it is a dark hole, and so is every shadow under a rail, every gap between
cushion and frame, and the black 8 sitting in a corner. Shape does not help
either - a pocket is occluded by its own cushions from overhead, so it is
rarely a circle. Every automatic attempt trades a one-time ten-second cost for
a failure mode that shows up mid-game.

The camera is fixed, so the pockets do not move. Clicking them once is exact,
takes ten seconds, and cannot fail. The coordinates go to pockets.json and are
reused on every later run of the same camera; only a resolution change (a
different camera or a different clip size) invalidates them, and that is
checked on load rather than trusted.

Usage:
    python detect_pocket.py                  # click on game.mp4's first frame
    python detect_pocket.py clip.mp4         # a different video
    python detect_pocket.py --out rig2.json  # save somewhere else
    python detect_pocket.py --frame 500      # calibrate on a later frame
    python detect_pocket.py --force          # re-click even if a file exists

In the click window:
    left click   drop a pocket marker (6 of them)
    u / BKSP     undo the last one
    r            start over
    ENTER/SPACE  accept (enabled once all six are down)
    ESC / q      cancel
"""

import argparse
import json
import os
import sys
from collections import namedtuple

import cv2

#: Where the clicked coordinates live, next to the script by default.
DEFAULT_POCKET_FILE = "pockets.json"

#: Six: four corners and two sides. The count is fixed, which is what lets the
#: picker know when it is finished and lets load() reject a truncated file.
POCKET_COUNT = 6

WINDOW = "BPS - click the 6 pockets"

#: The click window is fit inside this box. A 4K frame shown 1:1 is bigger than
#: the screen, and a window the user has to scroll makes precise clicking worse,
#: not better - the magnifier below is what buys the precision back.
MAX_VIEW_W, MAX_VIEW_H = 1280, 720

#: Magnifier: side of the inset in screen pixels, and how much it enlarges.
#: Downscaling 4K by 3x means one screen pixel is three real ones, so a click
#: judged on the fit-to-screen view is only accurate to ~3px. The loupe shows
#: the full-resolution pixels under the cursor so the click lands where meant.
LOUPE_SIZE, LOUPE_ZOOM = 180, 4

#: Marker colour for a placed pocket, and for the one being placed.
POCKET_BGR = (60, 220, 255)

Pocket = namedtuple("Pocket", "name x y")


def name_pockets(points):
    """Give the six clicked points stable names, so downstream code can say
    which pocket without depending on click order.

    The table's long axis decides the layout: on a landscape view the side
    pockets sit on the top and bottom rails (TM/BM), on a portrait view they
    sit on the left and right ones (ML/MR). Rows are split by rank rather than
    by a midpoint, which keeps the 3/3 (or 2/2/2) split correct even when the
    table is slightly rotated in frame.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    if max(xs) - min(xs) >= max(ys) - min(ys):        # landscape table
        by_row = sorted(points, key=lambda p: p[1])
        top = sorted(by_row[:3], key=lambda p: p[0])
        bottom = sorted(by_row[3:], key=lambda p: p[0])
        names = ["TL", "TM", "TR", "BL", "BM", "BR"]
        ordered = top + bottom
    else:                                             # portrait table
        by_col = sorted(points, key=lambda p: p[0])
        left = sorted(by_col[:3], key=lambda p: p[1])
        right = sorted(by_col[3:], key=lambda p: p[1])
        names = ["TL", "ML", "BL", "TR", "MR", "BR"]
        ordered = left + right

    return [Pocket(n, int(x), int(y)) for n, (x, y) in zip(names, ordered)]


def save_pockets(path, pockets, W, H, video=None):
    """Write the calibration. The frame size goes with it: coordinates are in
    pixels, so they only mean anything against the resolution they were clicked
    at, and load() refuses to reuse them at any other."""
    data = {
        "frame": [int(W), int(H)],
        "video": os.path.basename(video) if video else None,
        "pockets": [{"name": p.name, "x": p.x, "y": p.y} for p in pockets],
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_pockets(path, W=None, H=None):
    """Read a saved calibration, or None if there is nothing usable there.

    Silently ignoring a bad file is deliberate: every failure here means "ask
    the user to click again", which costs ten seconds, whereas using pockets
    from a different camera would be wrong for the whole run.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        pockets = [Pocket(p["name"], int(p["x"]), int(p["y"]))
                   for p in data["pockets"]]
    except (ValueError, KeyError, TypeError, OSError):
        print(f"Pockets: ignoring unreadable {path}")
        return None

    if len(pockets) != POCKET_COUNT:
        print(f"Pockets: ignoring {path} ({len(pockets)} pockets, expected "
              f"{POCKET_COUNT})")
        return None

    saved = data.get("frame")
    if W and H and saved and tuple(saved) != (W, H):
        print(f"Pockets: ignoring {path} - clicked at {saved[0]}x{saved[1]}, "
              f"this video is {W}x{H}")
        return None
    return pockets


def _draw_loupe(display, frame, cursor, view_scale):
    """Full-resolution pixels under the cursor, as an inset with a crosshair."""
    if cursor is None:
        return
    x, y = cursor
    H, W = frame.shape[:2]
    half = max(4, LOUPE_SIZE // (2 * LOUPE_ZOOM))
    x0 = max(0, min(W - 2 * half, x - half))
    y0 = max(0, min(H - 2 * half, y - half))
    patch = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return
    patch = cv2.resize(patch, (LOUPE_SIZE, LOUPE_SIZE),
                       interpolation=cv2.INTER_NEAREST)

    # Crosshair at the exact pixel the click would take, not at the inset's
    # centre - near a frame edge the patch is clamped and the two differ.
    cx = int((x - x0) * LOUPE_SIZE / (2.0 * half))
    cy = int((y - y0) * LOUPE_SIZE / (2.0 * half))
    cv2.line(patch, (cx, 0), (cx, LOUPE_SIZE), POCKET_BGR, 1)
    cv2.line(patch, (0, cy), (LOUPE_SIZE, cy), POCKET_BGR, 1)

    # Park the inset in the corner away from the cursor, so it never covers
    # the thing being aimed at.
    dh, dw = display.shape[:2]
    margin = 10
    px = margin if x * view_scale > dw / 2 else dw - LOUPE_SIZE - margin
    py = margin
    if py + LOUPE_SIZE > dh or LOUPE_SIZE + 2 * margin > dw:
        return
    display[py:py + LOUPE_SIZE, px:px + LOUPE_SIZE] = patch
    cv2.rectangle(display, (px, py), (px + LOUPE_SIZE, py + LOUPE_SIZE),
                  POCKET_BGR, 1)


def _draw_markers(display, points, view_scale):
    for i, (x, y) in enumerate(points):
        sx, sy = int(x * view_scale), int(y * view_scale)
        cv2.circle(display, (sx, sy), 14, POCKET_BGR, 2, cv2.LINE_AA)
        cv2.circle(display, (sx, sy), 2, (255, 255, 255), -1)
        cv2.putText(display, str(i + 1), (sx + 17, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, str(i + 1), (sx + 17, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, POCKET_BGR, 1, cv2.LINE_AA)


def _draw_help(display, placed):
    done = placed >= POCKET_COUNT
    text = ("All 6 down - ENTER to accept" if done
            else f"Click pocket {placed + 1} of {POCKET_COUNT}")
    keys = "u undo   r reset   ENTER accept   ESC cancel"
    dw = display.shape[1]
    cv2.rectangle(display, (0, 0), (dw, 34), (0, 0, 0), -1)
    cv2.putText(display, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (120, 255, 120) if done else (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(display, keys, (max(300, dw - 420), 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)


def select_pockets(frame, window=WINDOW):
    """Show the frame, collect six clicks, return them in full-frame pixels.

    Returns None if the user cancelled (ESC, q, or closing the window), which
    the caller must treat as "no calibration" rather than as an empty one.
    """
    H, W = frame.shape[:2]
    view_scale = min(1.0, MAX_VIEW_W / float(W), MAX_VIEW_H / float(H))
    base = cv2.resize(frame, (int(W * view_scale), int(H * view_scale))) \
        if view_scale < 1.0 else frame.copy()

    state = {"points": [], "cursor": None}

    def on_mouse(event, mx, my, _flags, _param):
        # Clicks arrive in view coordinates; store full-frame ones so the
        # result does not depend on how the window happened to be scaled.
        fx = min(W - 1, max(0, int(round(mx / view_scale))))
        fy = min(H - 1, max(0, int(round(my / view_scale))))
        state["cursor"] = (fx, fy)
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < POCKET_COUNT:
            state["points"].append((fx, fy))

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        # A headless OpenCV build has no window to click in. Say so plainly -
        # the alternative is a traceback from deep inside imshow().
        print("Pockets: this OpenCV build has no GUI support "
              "(opencv-python-headless?), so the pockets cannot be clicked.\n"
              "        Install opencv-python, copy a pockets.json from a "
              "machine that has one, or run with --no-pockets.")
        return None
    cv2.setMouseCallback(window, on_mouse)
    print(f"Pockets: click the {POCKET_COUNT} pockets in the window "
          f"(u undo, r reset, ENTER accept, ESC cancel)")

    try:
        while True:
            display = base.copy()
            _draw_markers(display, state["points"], view_scale)
            _draw_loupe(display, frame, state["cursor"], view_scale)
            _draw_help(display, len(state["points"]))
            cv2.imshow(window, display)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32) and len(state["points"]) == POCKET_COUNT:
                return state["points"]
            if key in (27, ord("q")):
                return None
            if key in (8, 127, ord("u")) and state["points"]:
                state["points"].pop()
            if key == ord("r"):
                state["points"].clear()

            # Closing the window with its X is a cancel too; without this the
            # loop would spin forever against a window that no longer exists.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)                  # let the window actually close on Win/macOS


def calibrate(frame, path=DEFAULT_POCKET_FILE, video=None, force=False):
    """Load the saved pockets, or ask for them once and save. None = cancelled.

    This is the whole point of the module: a fixed camera pays the ten-second
    click once and every later run starts straight into the video.
    """
    H, W = frame.shape[:2]
    if not force:
        pockets = load_pockets(path, W, H)
        if pockets:
            print(f"Pockets: {len(pockets)} loaded from {path} "
                  f"({', '.join(p.name for p in pockets)})")
            return pockets

    points = select_pockets(frame)
    if points is None:
        return None

    pockets = name_pockets(points)
    save_pockets(path, pockets, W, H, video)
    print(f"Pockets: saved {len(pockets)} to {path}")
    for p in pockets:
        print(f"  {p.name}: ({p.x}, {p.y})")
    return pockets


def draw_pockets(img, pockets, scale=1.0, radius=None):
    """Mark each pocket on an output frame: a ring, a centre dot and its name.

    Drawn under the ball overlay, and in a hue no ball class uses, so a pocket
    is never mistaken for a detection.
    """
    if not pockets:
        return img
    r = int(radius if radius else 22 * scale)
    thick = max(1, int(round(scale)))
    for p in pockets:
        cv2.circle(img, (p.x, p.y), r, POCKET_BGR, thick, cv2.LINE_AA)
        cv2.circle(img, (p.x, p.y), max(2, thick), POCKET_BGR, -1, cv2.LINE_AA)
        cv2.putText(img, p.name, (p.x + r + 4, p.y + int(5 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(img, p.name, (p.x + r + 4, p.y + int(5 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, POCKET_BGR,
                    thick, cv2.LINE_AA)
    return img


def nearest_pocket(x, y, pockets):
    """(Pocket, distance in px) closest to a point, or (None, inf).

    Here rather than in the caller because "which pocket did that ball go
    down" is the question every consumer of this file eventually asks.
    """
    if not pockets:
        return None, float("inf")
    best = min(pockets, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
    return best, ((best.x - x) ** 2 + (best.y - y) ** 2) ** 0.5


def main():
    ap = argparse.ArgumentParser(
        description="Click the six pockets once and save them for later runs.")
    ap.add_argument("video", nargs="?", default=None,
                    help="video to calibrate against (default: first .mp4 found)")
    ap.add_argument("--out", default=DEFAULT_POCKET_FILE,
                    help=f"where to save the pockets (default: {DEFAULT_POCKET_FILE})")
    ap.add_argument("--frame", type=int, default=0,
                    help="frame to click on; use a later one if the first is "
                         "blurred or blocked (default 0)")
    ap.add_argument("--force", action="store_true",
                    help="re-click even if the file already has a calibration")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    # main.py owns video discovery; reuse it so both entry points agree.
    from main import find_default_video
    if args.video is None:
        in_path = find_default_video(here)
        if in_path is None:
            sys.exit(f"No video found in {here} - pass one: "
                     f"python detect_pocket.py clip.mp4")
    else:
        in_path = args.video if os.path.isabs(args.video) \
            else os.path.join(here, args.video)
        if not os.path.exists(in_path):
            sys.exit(f"Video not found: {in_path}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {in_path}")
    if args.frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read frame {args.frame} of {in_path}")

    print(f"Input:  {in_path}  ({frame.shape[1]}x{frame.shape[0]}, "
          f"frame {args.frame})")
    pockets = calibrate(frame, out_path, video=in_path, force=args.force)
    if pockets is None:
        sys.exit("Cancelled - nothing saved.")

    preview = os.path.join(here, "pockets.png")
    draw_pockets(frame, pockets, scale=max(0.6, frame.shape[0] / 1080.0))
    if cv2.imwrite(preview, frame):
        print(f"Saved:  {preview}")


if __name__ == "__main__":
    main()
