"""
HSV range tuner - find the right color range for your ball.

Drag the sliders until ONLY your ball shows up white in the "Mask" panel
and everything else is black. Then read off the HSV values and plug them
straight into server.py.

Usage:
    python tuner.py             # reads test.png / test.jpg from this folder
    python tuner.py ball.jpg    # or point it at any image

Controls:
    - 6 sliders: low/high for Hue, Saturation, Value
    - LEFT-CLICK the ball to auto-seed the range from the pixels you clicked,
      then fine-tune with the sliders
    - RIGHT-CLICK to re-centre the view (this is how you pan when zoomed in)
    - press '1' Original / '2' Mask / '3' Result - one big panel, easiest to
      click accurately; '4' shows all three side by side (each is 1/3 the size)
    - press '+' / '-' to zoom in and out
    - press 'p' to print the current HSV range (+ a ready-to-run command)
    - press 's' to save the current mask + result to disk
    - press 'q' or ESC to quit

Panels are sized to fill your screen. On a wide image the 3-up view makes each
panel small, so the tool starts on the single "Original" panel instead.

Note on OpenCV's HSV scale:
    Hue is 0-179 (not 0-360), Saturation and Value are 0-255.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Same default-input rules as the detector, so both tools pick up the same file.
from server import DEFAULT_INPUTS, find_default_input

WIN = "Tuner"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Vertical space the 6 trackbars occupy above the image.
TRACKBAR_STRIP = 300


def nothing(_):
    pass


def screen_size(fallback=(1600, 900)):
    """Best-effort desktop size, so panels can be sized to actually fill it."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if sw > 0 and sh > 0:
            return sw, sh
    except Exception:
        pass
    try:
        import tkinter
        root = tkinter.Tk()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return sw, sh
    except Exception:
        return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None,
                    help=f"path to an image of your ball "
                         f"(default: first of {', '.join(DEFAULT_INPUTS)})")
    ap.add_argument("--max-display", type=int, default=0,
                    help="max width of EACH panel on screen (default: fit to screen)")
    ap.add_argument("--view", choices=["all", "orig", "mask", "result"], default="orig",
                    help="which panel(s) to show at startup (default: orig, the biggest)")
    args = ap.parse_args()

    # Resolve bare filenames next to this script, so `python tuner.py` works
    # from any working directory.
    here = os.path.dirname(os.path.abspath(__file__))
    if args.image is None:
        in_path = find_default_input(here)
        if in_path is None:
            sys.exit(f"No input image found in {here}\n"
                     f"Put one of {', '.join(DEFAULT_INPUTS)} there, "
                     f"or pass a path: python tuner.py myimage.jpg")
    else:
        in_path = args.image if os.path.isabs(args.image) else os.path.join(here, args.image)
        if not os.path.exists(in_path):
            sys.exit(f"Input image not found: {in_path}")

    img = cv2.imread(in_path)
    if img is None:
        raise SystemExit(f"Could not read image: {in_path}")

    print(f"Input: {in_path}  ({img.shape[1]}x{img.shape[0]})")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    screen_w, screen_h = screen_size()
    # The trackbar strip eats vertical space above the image.
    avail_w, avail_h = screen_w - 60, screen_h - TRACKBAR_STRIP

    # View, zoom and pan centre are mutable so the keyboard and the mouse
    # callback share one source of truth. HSV sampling always reads the
    # full-resolution image regardless of what is on screen.
    state = {"view": args.view, "zoom": 1.0,
             "scale": 1.0, "panel_w": 0, "x0": 0, "y0": 0,
             "cx": w // 2, "cy": h // 2}

    def viewport():
        """(scale, x0, y0, vis_w, vis_h) for the current view/zoom/pan.

        Zooming in shows a crop of the source instead of overflowing the
        screen, so there is always a way to reach every part of the image.
        """
        cols = 3 if state["view"] == "all" else 1
        if args.max_display:
            fit = min(args.max_display / w, avail_h / h)
        else:
            fit = min(avail_w / (w * cols), avail_h / h)
        scale = max(0.05, fit * state["zoom"])

        # How much of the source fits on screen at this scale.
        vis_w = max(1, min(w, int(avail_w / cols / scale)))
        vis_h = max(1, min(h, int(avail_h / scale)))
        x0 = int(min(max(state["cx"] - vis_w // 2, 0), w - vis_w))
        y0 = int(min(max(state["cy"] - vis_h // 2, 0), h - vis_h))
        return scale, x0, y0, vis_w, vis_h

    # Fail early and clearly if there's no GUI (headless build / no display).
    try:
        cv2.namedWindow(WIN)
    except cv2.error as e:
        raise SystemExit(
            "Could not open a GUI window - this tool needs a desktop display.\n"
            "  - If you installed 'opencv-python-headless', install 'opencv-python' instead.\n"
            "  - On WSL/SSH you need an X server or display forwarding.\n"
            f"(original error: {e})"
        )

    # Trackbars. Start fully open so the whole image shows, then you narrow it.
    cv2.createTrackbar("H low",  WIN, 0,   179, nothing)
    cv2.createTrackbar("H high", WIN, 179, 179, nothing)
    cv2.createTrackbar("S low",  WIN, 0,   255, nothing)
    cv2.createTrackbar("S high", WIN, 255, 255, nothing)
    cv2.createTrackbar("V low",  WIN, 0,   255, nothing)
    cv2.createTrackbar("V high", WIN, 255, 255, nothing)

    lo_bound = np.array([0, 0, 0])
    hi_bound = np.array([179, 255, 255])

    def to_source(x, y):
        """Map a click in the window back to full-resolution image coords."""
        scale, panel_w = state["scale"], state["panel_w"]
        # In "all" view the panels sit side by side; every panel shows the same
        # geometry, so a click in any of the three maps the same way.
        if state["view"] == "all":
            if x >= panel_w * 3:
                return None
            x %= panel_w
        elif x >= panel_w:
            return None
        ox = int(state["x0"] + x / scale)
        oy = int(state["y0"] + y / scale)
        if not (0 <= ox < w and 0 <= oy < h):
            return None
        return ox, oy

    def on_mouse(event, x, y, flags, param):
        # Right-click re-centres the view - that is how you pan when zoomed in.
        if event == cv2.EVENT_RBUTTONDOWN:
            pt = to_source(x, y)
            if pt:
                state["cx"], state["cy"] = pt
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pt = to_source(x, y)
        if pt is None:
            return
        ox, oy = pt
        scale = state["scale"]
        # Sample a patch sized to the zoom level: when zoomed out, one screen
        # pixel covers several source pixels, so widen the patch to match.
        rad = max(4, int(round(4 / scale)))
        y0, y1 = max(0, oy - rad), min(h, oy + rad + 1)
        x0, x1 = max(0, ox - rad), min(w, ox + rad + 1)
        patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(int)
        margin = np.array([10, 40, 40])  # widen a bit for shadows/highlights
        lo = np.clip(patch.min(0) - margin, lo_bound, hi_bound)
        hi = np.clip(patch.max(0) + margin, lo_bound, hi_bound)
        cv2.setTrackbarPos("H low",  WIN, int(lo[0]))
        cv2.setTrackbarPos("S low",  WIN, int(lo[1]))
        cv2.setTrackbarPos("V low",  WIN, int(lo[2]))
        cv2.setTrackbarPos("H high", WIN, int(hi[0]))
        cv2.setTrackbarPos("S high", WIN, int(hi[1]))
        cv2.setTrackbarPos("V high", WIN, int(hi[2]))
        print(f"sampled -> low {lo.tolist()}  high {hi.tolist()}")

    cv2.setMouseCallback(WIN, on_mouse)
    print("Drag sliders or click the ball.")
    print("Keys: 1 Original  2 Mask  3 Result  4 all three | +/- zoom | "
          "p print, s save, q quit.")

    while True:
        hl = cv2.getTrackbarPos("H low",  WIN)
        hh = cv2.getTrackbarPos("H high", WIN)
        sl = cv2.getTrackbarPos("S low",  WIN)
        sh = cv2.getTrackbarPos("S high", WIN)
        vl = cv2.getTrackbarPos("V low",  WIN)
        vh = cv2.getTrackbarPos("V high", WIN)

        low = np.array([hl, sl, vl])
        high = np.array([hh, sh, vh])
        mask = cv2.inRange(hsv, low, high)
        result = cv2.bitwise_and(img, img, mask=mask)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Size the panels for the current view/zoom/pan, then remember the
        # numbers the mouse callback needs to undo the scaling.
        scale, x0, y0, vis_w, vis_h = viewport()
        pw, ph = max(1, int(vis_w * scale)), max(1, int(vis_h * scale))
        state["scale"], state["panel_w"] = scale, pw
        state["x0"], state["y0"] = x0, y0
        crop = (slice(y0, y0 + vis_h), slice(x0, x0 + vis_w))

        view = state["view"]
        if view == "all":
            panels = [(img, "Original (click ball)", (0, 255, 0)),
                      (mask_bgr, "Mask", (255, 255, 255)),
                      (result, "Result", (0, 255, 0))]
        elif view == "mask":
            panels = [(mask_bgr, "Mask (click ball)", (255, 255, 255))]
        elif view == "result":
            panels = [(result, "Result (click ball)", (0, 255, 0))]
        else:
            panels = [(img, "Original (click ball)", (0, 255, 0))]

        shown = []
        for panel, label, color in panels:
            p = cv2.resize(panel[crop], (pw, ph))
            cv2.rectangle(p, (0, 0), (pw, 36), (0, 0, 0), -1)
            cv2.putText(p, label, (10, 26), FONT, 0.7, color, 2, cv2.LINE_AA)
            shown.append(p)
        combo = np.hstack(shown) if len(shown) > 1 else shown[0]

        hud = (f"{view}  {scale:.2f}x  |  1 2 3 4 view   +/- zoom   "
               f"right-click pan   p print   s save   q quit")
        # Solid bar behind the HUD - a text outline turns to mush at this size.
        bar_h = 30
        cv2.rectangle(combo, (0, combo.shape[0] - bar_h),
                      (combo.shape[1], combo.shape[0]), (0, 0, 0), -1)
        cv2.putText(combo, hud, (10, combo.shape[0] - 9), FONT, 0.55, (255, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.imshow(WIN, combo)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            state["view"], state["zoom"] = "orig", 1.0
        elif key == ord('2'):
            state["view"], state["zoom"] = "mask", 1.0
        elif key == ord('3'):
            state["view"], state["zoom"] = "result", 1.0
        elif key in (ord('4'), ord('0'), ord('a')):
            state["view"], state["zoom"] = "all", 1.0
        elif key in (ord('+'), ord('=')):
            state["zoom"] = min(8.0, state["zoom"] * 1.25)
        elif key in (ord('-'), ord('_')):
            state["zoom"] = max(0.2, state["zoom"] / 1.25)
        elif key == ord('p'):
            print(f"\nHSV low  = [{hl}, {sl}, {vl}]")
            print(f"HSV high = [{hh}, {sh}, {vh}]")
            print("Run your detector with:")
            print(f"  python server.py {os.path.basename(in_path)} --mode color "
                  f"--hsv-low {hl} {sl} {vl} --hsv-high {hh} {sh} {vh}\n")
        elif key == ord('s'):
            mask_path = os.path.join(here, "mask.png")
            result_path = os.path.join(here, "masked_result.png")
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(result_path, result)
            print(f"Saved {mask_path} and {result_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()