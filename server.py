"""
BPS prototype - detect a ball's position in a single image.

Three detection modes:
  1. "felt"   - top-down pool table (DEFAULT). Keeps whatever on the table is
                NOT felt, then fits one ball-sized disc everywhere it can go.
                Describing the single felt colour is easier than describing
                fifteen ball colours, and a fixed camera means every ball is
                the same size, so the radius is measured once and shared.
  2. "hough"  - shape-based (finds circles, works for any ball color)
  3. "color"  - HSV color mask + contour (fast and robust if you know the ball color)

Usage:
  python server.py                             # reads test.png/.jpg, writes result.png
  python server.py other.jpg                   # different input image
  python server.py --ball-r 20                 # force the ball radius, in pixels
  python server.py --debug                     # also dump the felt/table masks
  python server.py --mode hough --top 1        # draw only the strongest circle
  python server.py --mode color --hsv-low 5 120 120 --hsv-high 20 255 255

Output:
  Prints (x, y, radius) for every ball found and saves result.png with the detections drawn.
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

# Tried in order when no image argument is given - first one that exists wins.
DEFAULT_INPUTS = ["test.png", "test.jpg", "test.jpeg"]
DEFAULT_OUTPUT = "result.png"


def find_default_input(folder):
    """Return the first DEFAULT_INPUTS candidate present in folder, else None."""
    for name in DEFAULT_INPUTS:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def detect_hough(img, min_radius=None, max_radius=None):
    """Shape-based detection. Returns list of (x, y, r), strongest candidate first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur reduces false circles from noise/texture
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Scale the radius window to the image, so the same defaults work whether the
    # frame is 300px or 3000px wide.
    short_side = min(img.shape[:2])
    if min_radius is None:
        min_radius = max(10, int(0.08 * short_side))
    if max_radius is None:
        max_radius = int(0.50 * short_side)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,                  # inverse accumulator resolution
        minDist=short_side // 4,  # min distance between circle centers
        param1=100,              # Canny high threshold
        param2=40,               # accumulator threshold (lower = more circles)
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    return [(x, y, r) for x, y, r in circles]


# Felt colour window, read off tuner.py against this venue's footage:
#   H 86-107   S 215-255   V 215-255
# Absolute bounds beat the old "sample the median, allow +/- a tolerance"
# approach here because the tolerance had to be opened to S+/-130 to stop
# glare reading as a ball, and that wide a window also swallowed dark blue
# balls whose hue sits next to the cloth.
FELT_LOW = (86, 215, 215)
FELT_HIGH = (107, 255, 255)

# A specular highlight is still felt: same hue, saturation washed out, value
# near maximum. Held as a SECOND felt band so a reflection on the cloth does
# not come back as a phantom ball, while the main window stays tight enough
# to keep the blue balls. Measured on the reflection in this footage: S~132,
# V~255, which this band covers and no ball in the set does.
GLARE_S_MIN = 100
GLARE_V_MIN = 240


def felt_mask(hsv, low=FELT_LOW, high=FELT_HIGH):
    """Felt pixels: the tuned colour window, plus its specular-highlight band."""
    lo = np.array(low, np.uint8)
    hi = np.array(high, np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    glare_s_hi = max(int(lo[1]), GLARE_S_MIN + 1)
    glare = cv2.inRange(hsv,
                        np.array([lo[0], GLARE_S_MIN, GLARE_V_MIN], np.uint8),
                        np.array([hi[0], glare_s_hi, 255], np.uint8))
    return cv2.bitwise_or(mask, glare)


def sample_felt(hsv, H, W):
    """Median HSV of the frame centre - on a top-down table shot that is felt."""
    patch = hsv[int(H * 0.4):int(H * 0.6), int(W * 0.4):int(W * 0.6)].reshape(-1, 3)
    return np.median(patch, axis=0)


def table_region(felt_mask, H, W, shrink):
    """Largest felt blob, filled and eroded inward to drop rails/cushions/pockets."""
    fm = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((H, W), np.uint8)
    table = max(cnts, key=cv2.contourArea)
    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [table], -1, 255, -1)
    # When the table runs off the frame its contour touches the image border,
    # and erosion cannot pull inward there - leaving a strip of rail inside the
    # mask that shows up as phantom balls. Treat the border as outside.
    mask[0, :] = mask[-1, :] = 0
    mask[:, 0] = mask[:, -1] = 0
    return cv2.erode(mask, np.ones((shrink, shrink), np.uint8))


def fill_holes(mask):
    """Redraw each blob solid.

    Stripes and printed numbers punch holes in a ball's blob, which would
    otherwise leave a thin ring whose centre the split stage cannot find.
    Filling per-contour closes those holes without growing the blob or
    merging it into a neighbour, which a morphological close would do.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


#: How well a disc must match to count as a ball, as the fraction of its area
#: that is non-felt. Measured on this footage: 0.90 finds only separated balls
#: (13 of 16), 0.85 finds 15, 0.80 finds all 16, and 0.75 finds no more - so
#: 0.80 sits in the middle of the range that gets the count right.
BALL_COVERAGE = 0.80

#: Suppression distance around an accepted centre, in ball radii. Two touching
#: balls are 2r apart, so a second peak closer than this is the same ball found
#: twice rather than its neighbour.
BALL_SEPARATION = 1.6

#: Most balls that can be on the table at once: fifteen plus the cue.
MAX_BALLS = 16


def estimate_ball_radius(not_felt, default=20):
    """One radius for every ball, measured off the blobs that are single balls.

    A fixed overhead camera sees every ball at the same size, so a per-ball
    radius guess is noise, not signal - and a radius that is wrong by a few
    pixels drags the colour sample onto the cloth or onto a neighbour. Balls
    that touch merge into one blob whose area means nothing, so they are
    excluded by shape: a lone ball nearly fills its bounding box (pi/4 = 0.785)
    and is square, a cluster is neither.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(not_felt)
    radii = []
    for i in range(1, count):                     # 0 is the background
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 200 or not w or not h:
            continue
        if not 0.70 < area / float(w * h) < 0.90:
            continue
        if not 0.8 < w / float(h) < 1.25:
            continue
        radii.append(math.sqrt(area / math.pi))
    return int(round(np.median(radii))) if radii else default


def drop_oversize_blobs(not_felt, r, max_balls=MAX_BALLS):
    """Erase anything too big to be a pile of balls.

    A player's arm over the table is not felt either, and a ball-sized disc
    fits it perfectly at hundreds of positions - one frame of this footage
    produced 58 "balls", forty of them along a forearm. No local test tells the
    two apart: a forearm has edges, colour variation and a rounded outline, and
    so does a ball wedged in a rack.

    What does tell them apart is total size. Only sixteen balls exist, so one
    connected region can hold at most sixteen ball areas. The arm measured
    forty-nine; the entire fifteen-ball rack, fifteen. A ball hidden under an
    arm is lost for those frames, which costs nothing - it was not visible.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(not_felt)
    limit = max_balls * math.pi * r * r
    out = not_felt.copy()
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] > limit:
            out[labels == i] = 0
    return out


def find_balls(not_felt, r, coverage=BALL_COVERAGE, separation=BALL_SEPARATION):
    """One centre per ball, by asking where a ball-sized disc fits the mask.

    Correlating a filled disc against the mask scores every pixel by the
    fraction of that disc which is non-felt - how well a ball centred there
    explains what was seen. Taking the peaks greedily and suppressing a
    neighbourhood after each separates touching balls: the score stays high
    across a whole cluster, but only one peak per ball survives suppression.
    """
    mask = drop_oversize_blobs(not_felt, r).astype(np.float32) / 255.0
    disc = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(disc, (r, r), r, 1.0, -1)
    score = cv2.filter2D(mask, -1, disc / disc.sum())

    balls = []
    while True:
        _, best, _, (x, y) = cv2.minMaxLoc(score)
        if best < coverage:
            break
        balls.append((int(x), int(y), int(r)))
        cv2.circle(score, (x, y), int(separation * r), 0.0, -1)
    return balls


def not_felt_mask(img, table, felt_low=FELT_LOW, felt_high=FELT_HIGH):
    """Everything on the table that is not cloth - the ball candidates.

    `table` is passed in rather than measured here because table_region() is
    the most expensive step in the pipeline and a fixed camera only needs it
    once.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    not_felt = cv2.bitwise_and(cv2.bitwise_not(felt_mask(hsv, felt_low, felt_high)),
                               table)
    not_felt = cv2.morphologyEx(not_felt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return fill_holes(not_felt)


def detect_felt(img, ball_r=None, felt_low=FELT_LOW, felt_high=FELT_HIGH,
                shrink=18):
    """Felt-inversion detection for a top-down table.

    Rather than describing every ball colour, measure the one colour that is
    constant - the felt - and treat whatever sits on the table and is NOT felt
    as a ball candidate. One ball-sized disc is then fitted everywhere it can
    go, which is what separates balls that touch.

    Returns (balls, not_felt, table_mask) so callers can inspect the masks.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = img.shape[:2]
    table_mask = table_region(felt_mask(hsv, felt_low, felt_high), H, W, shrink)
    not_felt = not_felt_mask(img, table_mask, felt_low, felt_high)
    if ball_r is None:
        ball_r = estimate_ball_radius(not_felt)
    balls = find_balls(not_felt, ball_r)
    balls.sort(key=lambda b: (b[1], b[0]))
    return balls, not_felt, table_mask


def detect_color(img, hsv_low, hsv_high, min_area=100):
    """Color-based detection. Returns list of (x, y, r)."""
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    # Clean up the mask: remove specks, fill small holes
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        # Optional roundness check: ball contour area should be close to circle area
        circularity = cv2.contourArea(c) / (np.pi * r * r + 1e-6)
        if circularity < 0.6:
            continue
        results.append((int(x), int(y), int(r)))

    # Largest first - usually the actual ball
    results.sort(key=lambda b: b[2], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None,
                    help=f"path to input image (default: first of {', '.join(DEFAULT_INPUTS)})")
    ap.add_argument("--mode", choices=["felt", "hough", "color"], default="felt")
    ap.add_argument("--hsv-low", nargs=3, type=int, default=[5, 120, 120],
                    help="HSV lower bound for color mode (default: orange)")
    ap.add_argument("--hsv-high", nargs=3, type=int, default=[20, 255, 255],
                    help="HSV upper bound for color mode")
    ap.add_argument("--ball-r", type=int, default=None,
                    help="ball radius in px (felt mode); measured from the "
                         "image when not given")
    ap.add_argument("--felt-low", nargs=3, type=int, default=list(FELT_LOW),
                    help=f"felt HSV lower bound, from tuner.py (default {FELT_LOW})")
    ap.add_argument("--felt-high", nargs=3, type=int, default=list(FELT_HIGH),
                    help=f"felt HSV upper bound, from tuner.py (default {FELT_HIGH})")
    ap.add_argument("--debug", action="store_true",
                    help="also save the felt/table masks next to the output")
    ap.add_argument("--out", default=DEFAULT_OUTPUT,
                    help=f"path to output image (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--top", type=int, default=0,
                    help="how many detections to draw, best first (0 = all, the default)")
    args = ap.parse_args()

    # Resolve bare filenames next to this script, so `python server.py` works
    # from any working directory.
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    if args.image is None:
        in_path = find_default_input(here)
        if in_path is None:
            sys.exit(f"No input image found in {here}\n"
                     f"Put one of {', '.join(DEFAULT_INPUTS)} there, "
                     f"or pass a path: python server.py myimage.jpg")
    else:
        in_path = args.image if os.path.isabs(args.image) else os.path.join(here, args.image)
        if not os.path.exists(in_path):
            sys.exit(f"Input image not found: {in_path}")

    img = cv2.imread(in_path)
    if img is None:
        raise SystemExit(f"Could not read image: {in_path}")

    print(f"Input:  {in_path}  ({img.shape[1]}x{img.shape[0]})")

    if args.mode == "felt":
        balls, not_felt, table_mask = detect_felt(
            img, args.ball_r, args.felt_low, args.felt_high)
        if args.debug:
            base = os.path.splitext(out_path)[0]
            cv2.imwrite(f"{base}_notfelt.png", not_felt)
            cv2.imwrite(f"{base}_table.png", table_mask)
            print(f"Debug:  {base}_notfelt.png, {base}_table.png")
    elif args.mode == "hough":
        balls = detect_hough(img)
    else:
        balls = detect_color(img, args.hsv_low, args.hsv_high)

    if not balls:
        print("No ball detected.")
    elif args.top > 0:
        balls = balls[:args.top]

    # Red reads better than green on cyan felt; green suits the other modes.
    ring = (0, 0, 255) if args.mode == "felt" else (0, 255, 0)
    # A full "(x,y) r=" label per ball is unreadable once a table is crowded,
    # so past a handful just number them - the coordinates go to stdout anyway.
    terse = len(balls) > 4

    for i, (x, y, r) in enumerate(balls):
        print(f"Ball {i}: center=({x}, {y})  radius={r}")
        cv2.circle(img, (x, y), r, ring, 2)             # outline
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)   # center dot

        label = str(i) if terse else f"({x},{y}) r={r}"
        # Keep the label on-image when the ball sits near an edge.
        ly = y - r - 8
        if ly < 15:
            ly = min(y + r + 20, img.shape[0] - 5)
        lx = max(5, min(x - (8 if terse else 40), img.shape[1] - (20 if terse else 130)))
        # Dark stroke under the text so it stays readable on any background.
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if not cv2.imwrite(out_path, img):
        raise SystemExit(f"Could not write image: {out_path}")
    print(f"Saved:  {out_path}")


if __name__ == "__main__":
    main()