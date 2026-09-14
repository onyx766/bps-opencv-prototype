"""
BPS prototype - detect a ball's position in a single image.

Three detection modes:
  1. "felt"   - top-down pool table (DEFAULT). Auto-samples the felt colour and
                keeps whatever on the table is NOT felt. No HSV to hand-tune:
                one felt colour is measured from the image, instead of trying to
                describe fifteen different ball colours.
  2. "hough"  - shape-based (finds circles, works for any ball color)
  3. "color"  - HSV color mask + contour (fast and robust if you know the ball color)

Usage:
  python server.py                             # reads test.png/.jpg, writes result.png
  python server.py other.jpg                   # different input image
  python server.py --min-r 15 --max-r 30       # ball radius window, in pixels
  python server.py --debug                     # also dump the felt/table masks
  python server.py --mode hough --top 1        # draw only the strongest circle
  python server.py --mode color --hsv-low 5 120 120 --hsv-high 20 255 255

Output:
  Prints (x, y, radius) for every ball found and saves result.png with the detections drawn.
"""

import argparse
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


def split_blobs(not_felt, r_typ, peak_frac=0.5):
    """Find one centre per ball, splitting balls that touch.

    A distance transform peaks at the centre of each ball; thresholding it
    well below the ball radius separates a cluster into one component per
    ball, where plain contours would report the whole cluster as one lump.
    """
    dist = cv2.distanceTransform(not_felt, cv2.DIST_L2, 5)
    peaks = (dist >= peak_frac * r_typ).astype(np.uint8)
    count, _, _, centroids = cv2.connectedComponentsWithStats(peaks)

    balls = []
    H, W = not_felt.shape[:2]
    for i in range(1, count):                     # 0 is the background
        x, y = (int(round(v)) for v in centroids[i])
        if not (0 <= x < W and 0 <= y < H):
            continue
        r = int(round(dist[y, x])) or int(r_typ)  # distance at the centre ~ radius
        balls.append((x, y, r))
    return balls


def hough_alt(img, table_mask, not_felt, min_r, max_r,
              param1=300, param2=0.5, dp=1.0):
    """Circle detection with HOUGH_GRADIENT_ALT, gated to real table content.

    ALT is the accurate variant of the Hough circle transform: param2 is a
    0-1 "perfectness" ratio rather than an accumulator count. It finds balls
    that are touching, where the blob split stage sees a rack as one lump.

    Two gates keep its extra sensitivity from inventing balls:
      - the centre must sit a ball-radius inside the table (drops pockets,
        rails and cushion shadows)
      - the disc must mostly cover non-felt pixels (drops felt glare)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    inner = cv2.erode(table_mask, np.ones((min_r, min_r), np.uint8))

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=dp,
                               minDist=int(min_r * 1.5), param1=param1,
                               param2=param2, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return []

    balls = []
    for x, y, r in np.round(circles[0]).astype(int):
        if not (0 <= x < W and 0 <= y < H) or not inner[y, x]:
            continue
        disc = not_felt[max(0, y - r):y + r, max(0, x - r):x + r]
        if disc.size == 0 or (disc > 0).mean() < 0.30:
            continue
        balls.append((int(x), int(y), int(r)))
    return balls


def merge_detections(primary, extra, tol):
    """Keep every primary detection, add extras that are not already covered."""
    out = list(primary)
    for x, y, r in extra:
        if all((x - px) ** 2 + (y - py) ** 2 > tol * tol for px, py, _ in out):
            out.append((x, y, r))
    return out


def detect_felt(img, min_r=15, max_r=30, felt_tol=(18, 130, 90), shrink=18,
                shape="hybrid", alt_param2=0.5):
    """Felt-inversion detection for a top-down table.

    Rather than describing every ball colour, measure the one colour that is
    constant - the felt - and treat whatever sits on the table and is NOT felt
    as a ball candidate. The shape stage then splits touching balls, which a
    blob-only approach would merge into one lump.

    Returns (balls, not_felt, table_mask) so callers can inspect the masks.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = img.shape[:2]

    felt = sample_felt(hsv, H, W)
    tol = np.array(felt_tol)
    lo = np.clip(felt - tol, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    hi = np.clip(felt + tol, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    felt_mask = cv2.inRange(hsv, lo, hi)

    table_mask = table_region(felt_mask, H, W, shrink)
    not_felt = cv2.bitwise_and(cv2.bitwise_not(felt_mask), table_mask)
    not_felt = cv2.morphologyEx(not_felt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    not_felt = fill_holes(not_felt)

    # Shape stage. The two methods fail in opposite situations, so by default
    # run both: the blob split is precise on separated balls, ALT is the one
    # that can break a tight rack apart.
    dt_balls = []
    alt_balls = []
    if shape in ("hybrid", "dt"):
        dt_balls = split_blobs(not_felt, r_typ=(min_r + max_r) / 2.0)
        dt_balls = [(x, y, int(min(max(r, min_r), max_r))) for x, y, r in dt_balls]
    if shape in ("hybrid", "alt"):
        alt_balls = hough_alt(img, table_mask, not_felt, min_r, max_r,
                              param2=alt_param2)

    if shape == "dt":
        balls = dt_balls
    elif shape == "alt":
        balls = alt_balls
    else:
        # ALT first: it separates a cluster into individual balls, so its
        # centres are the ones worth keeping where the two disagree.
        balls = merge_detections(alt_balls, dt_balls, tol=min_r)

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
    ap.add_argument("--min-r", type=int, default=15, help="min ball radius, px (felt mode)")
    ap.add_argument("--max-r", type=int, default=30, help="max ball radius, px (felt mode)")
    ap.add_argument("--felt-tol", nargs=3, type=int, default=[18, 130, 90],
                    help="how far from the felt color still counts as felt (H S V)")
    ap.add_argument("--shape", choices=["hybrid", "alt", "dt"], default="hybrid",
                    help="felt-mode shape stage: HOUGH_GRADIENT_ALT, the blob "
                         "split, or both (default)")
    ap.add_argument("--alt-param2", type=float, default=0.5,
                    help="ALT circle 'perfectness', 0-1; lower finds more (default 0.5)")
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
            img, args.min_r, args.max_r, args.felt_tol,
            shape=args.shape, alt_param2=args.alt_param2)
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