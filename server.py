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
from collections import namedtuple

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


#: How LONG a connected region may be, in ball diameters, and still be a pile
#: of balls. The longest thing fifteen touching balls can make is a full rack,
#: which measured 4.7 diameters across on this footage; the arm-and-cue that
#: reaches in over the table measured 13.6 and a cue lying on the cloth 14.7.
#: Seven sits between them with room on both sides.
INTRUSION_SPAN = 7.0

#: A region that REACHES IN over the table boundary is an intrusion once it is
#: this big, in ball areas, and this loosely packed. Both halves are needed. A
#: ball resting against a cushion touches the boundary too - the table mask is
#: eroded inward, so the rail end of the ball sits on its edge - and so does
#: every hairline of glare along the rails, so touching alone proves nothing.
#: Size and packing are what separate them: balls pack densely, filling 0.75 to
#: 0.86 of their own bounding box singly, 0.41 to 0.48 in pairs and 0.59 for a
#: whole rack, while an arm crossing its box on the diagonal filled 0.10.
INTRUSION_EDGE_AREA = 3.0
INTRUSION_FILL = 0.35

#: Regions below this many ball areas are never tested. Nothing that small is
#: an arm, and skipping them keeps the shape work off the fifteen-odd little
#: blobs of rail glare every frame carries.
INTRUSION_MIN_AREA = 2.0

#: Half-width, in ball radii, at which an erased region was solid enough to
#: have HIDDEN a real ball or GROWN a phantom one - measured as the largest
#: disc that fits inside it. A hand measured 2.9 radii and the rack 2.9; a bare
#: cue shaft 0.57 and a hairline of rail glare 0.3. This is not what decides
#: whether a region is erased - a cue is erased either way - it is what decides
#: whether its presence should stop the scoreboard trusting the frame, and a
#: cue shaft too thin to hold a ball has no business stopping anything.
INTRUSION_THICK = 0.8

#: Half-width, in ball radii, below which part of a region is left out when it
#: is MEASURED for the too-long and reaching-in tests. The cushion nose along a
#: rail reads as a thin dark strip that is not felt - 0.15 to 0.32 radii
#: half-width on this footage - and a ball resting against that cushion merges
#: with it into one region 6 to 7.6 diameters long, which both tests condemned:
#: the ball was erased every frame it sat there, so the table read one ball
#: short and a ball knocked off that rail into a pocket was never seen to go.
#: A cue shaft measured 0.57, so 0.4 sheds the strip and still measures a cue.
INTRUSION_SOLID = 0.4

#: ... and the shedding only applies when what is left is BALL-SIZED pieces,
#: in ball areas each, none of them touching the table's edge. Measured: a
#: ball on the strip left 1.1 to 1.7 and two touching balls 2.3 to 2.5, while
#: every hand left a piece of 4.9 to 15.9. A bridge hand whose thin cue tip is
#: shed must still be measured whole - shed, it came in under INTRUSION_SPAN
#: and its fingers were counted as balls.
#:
#: Fingers resting over a cushion are the harder case, because each one IS
#: ball-sized once the strip is shed. What gives them away is where they are:
#: they come in over the rail, so they run right up to the table's edge - 0.05
#: radii from it, all 27 of them over this footage - while a ball resting on
#: the cushion has the strip between it and the edge, 0.44 radii at the least.
INTRUSION_LONE = 3.0


def outer_rim(table, pockets=(), clear=0):
    """The table's outer edge as a thin band, stopping short of every pocket.

    What a hand resting on a rail touches and a ball on the cloth never does -
    except in a pocket's jaw, where a ball on its way down reaches the edge as
    well, so the band leaves a disc of `clear` pixels round each pocket out.
    Taken from the table mask BEFORE the pocket holes are cut into it: the
    holes' own edges are where balls go down, not where hands come in.
    """
    rim = cv2.subtract(table, cv2.erode(table, np.ones((5, 5), np.uint8)))
    for p in pockets or ():
        cv2.circle(rim, (int(p.x), int(p.y)), int(round(clear)), 0, -1)
    return rim


#: One erased region. `bulky` is the flag the game layer reads: something was
#: on the table that could have hidden a ball, so this frame's count is a
#: guess. `box` is (x, y, w, h), for drawing it on the overlay.
Intrusion = namedtuple("Intrusion", "label area span thick fill edge bulky why x y box")


def find_intrusions(not_felt, r, table=None, max_balls=MAX_BALLS,
                    span=INTRUSION_SPAN, edge_area=INTRUSION_EDGE_AREA,
                    fill=INTRUSION_FILL, thick=INTRUSION_THICK,
                    solid=INTRUSION_SOLID, lone=INTRUSION_LONE, rim=None):
    """Which connected regions of the mask are not balls but a player.

    A player's arm over the table is not felt either, and a ball-sized disc
    fits it perfectly at hundreds of positions - one frame of this footage
    produced 58 "balls", forty of them along a forearm. No LOCAL test tells the
    two apart: a forearm has edges, colour variation and a rounded outline, and
    so does a ball wedged in a rack. Asked pixel by pixel, a knuckle is a ball.

    Asked of the WHOLE REGION it is not, and that is the trick here. An arm,
    the hand on the end of it and the cue in that hand are one connected black
    mass, because they touch; so the fingertips that each look exactly like a
    ball are part of a region that, taken together, obviously is not one. Four
    things give it away, any one of which is enough:

      too big     Only sixteen balls exist, so one region holds at most sixteen
                  ball areas however they are piled. The arm measured 24 and,
                  earlier in this footage, 49; the entire fifteen-ball rack, 15.
      too long    A rack is the longest thing balls alone can make - 4.7 ball
                  diameters. The arm-and-cue spanned 13.6.
      reaching in Balls sit ON the table; an arm comes in OVER its edge. A
                  region that touches the table boundary, is bigger than a few
                  balls and is packed far too loosely to be balls came from
                  outside.
      over the rail
                  A hand RESTING on a rail is neither big, long nor loose - two
                  fingers over the cushion read as two tidy balls. But its
                  solid part runs right up to the table's outer edge, which a
                  ball on the cloth never reaches outside a pocket's jaw. Only
                  tested against a `rim` band from outer_rim(), which leaves
                  the pockets out.

    Whichever fires, the answer is the same and it is the point of doing this
    by region: the ENTIRE mass is rejected, every attached fingertip with it.
    Picking off the fingers one at a time cannot work, because one at a time
    they are indistinguishable from balls.

    One exception to measuring the whole region: when shedding everything
    thinner than INTRUSION_SOLID leaves nothing but ball-sized pieces, clear of
    the table's edge, length and packing are measured on those pieces. That is
    a ball joined to the thin cushion strip along a rail, and it measures as
    the one ball it is rather than as a long, loosely packed thing reaching in
    over the edge. A hand leaves a piece far bigger than a ball, and fingers
    over a cushion leave pieces that run up to the edge; both are measured
    whole exactly as before. Without a table mask there is no edge to check,
    and nothing is exempted.

    `table` is the table mask the balls were found inside; without it the
    reaching-in test is skipped and only the size tests are left.

    Returns (labels, [Intrusion, ...]) - the label image so the caller can
    erase by region without labelling the mask twice.
    """
    count, labels, stats, cent = cv2.connectedComponentsWithStats(not_felt)
    ball_area = math.pi * r * r
    # One distance transform for the whole mask: regions are separated by
    # background, so each one's distance-to-outside is the same either way.
    dist = cv2.distanceTransform(not_felt, cv2.DIST_L2, 5)
    border = None
    if table is not None:
        border = cv2.subtract(table, cv2.erode(table, np.ones((5, 5), np.uint8)))
    pad = max(1, int(round(solid * r)))
    shed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))

    found = []
    for i in range(1, count):                     # 0 is the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < INTRUSION_MIN_AREA * ball_area:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        roi = np.where(labels[y0:y0 + h, x0:x0 + w] == i, 255, 0).astype(np.uint8)
        # The solid part: opened with a disc, so strips narrower than it are
        # shed. Padded first, or the crop's own edge would stop the erosion.
        core = cv2.morphologyEx(
            cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0),
            cv2.MORPH_OPEN, shed)
        n, _, pieces, _ = cv2.connectedComponentsWithStats(core)
        ball_sized = (border is not None and n > 1
                      and pieces[1:, cv2.CC_STAT_AREA].max() <= lone * ball_area
                      and not cv2.countNonZero(cv2.bitwise_and(
                          core[pad:pad + h, pad:pad + w],
                          border[y0:y0 + h, x0:x0 + w])))
        over_rail = rim is not None and cv2.countNonZero(cv2.bitwise_and(
            core[pad:pad + h, pad:pad + w], rim[y0:y0 + h, x0:x0 + w])) > 0
        measured = core if ball_sized else roi
        measured_area = cv2.countNonZero(measured)
        cnts, _ = cv2.findContours(measured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        # The MINIMUM-area rectangle, not the upright bounding box: an arm lies
        # across the table on the diagonal, and a rack measured by its upright
        # box is 7.1 diameters on the diagonal rather than the 4.7 it really is
        # - which would condemn the rack and let the arm off on the same number.
        (rw, rh) = cv2.minAreaRect(np.vstack(cnts))[1]
        major = max(rw, rh)
        long_ = major / (2.0 * r)
        packed = measured_area / max(1.0, rw * rh)
        half = float(dist[y0:y0 + h, x0:x0 + w][roi > 0].max()) / float(r)
        touches = (border is not None and
                   cv2.countNonZero(cv2.bitwise_and(roi,
                                                    border[y0:y0 + h,
                                                           x0:x0 + w])) > 0)

        if area > max_balls * ball_area:
            why = f"{area / ball_area:.0f} ball areas in one piece"
        elif long_ > span:
            why = f"{long_:.1f} ball diameters long"
        elif (touches and measured_area >= edge_area * ball_area
              and packed < fill):
            why = "reaches in over the table edge"
        elif over_rail:
            why = "rests over the rail"
        else:
            continue

        found.append(Intrusion(
            label=i, area=round(area / ball_area, 2), span=round(long_, 2),
            thick=round(half, 2), fill=round(packed, 2), edge=bool(touches),
            bulky=half >= thick, why=why,
            x=int(cent[i][0]), y=int(cent[i][1]),
            box=(int(x0), int(y0), int(w), int(h))))
    return labels, found


def drop_intrusions(not_felt, r, table=None, max_balls=MAX_BALLS, rim=None):
    """The mask with every non-ball region erased whole, and what was erased.

    A ball hidden under an arm is lost for those frames, which costs nothing -
    it was not visible. A ball erased because it happened to touch the arm
    costs nothing either, for the same reason, and losing it is the entire
    point: attached to the arm it was going to be counted at the wrong place.
    """
    labels, found = find_intrusions(not_felt, r, table, max_balls, rim=rim)
    if not found:
        return not_felt, found
    out = not_felt.copy()
    for it in found:
        out[labels == it.label] = 0
    return out, found


def drop_oversize_blobs(not_felt, r, max_balls=MAX_BALLS):
    """Back-compatible wrapper: the cleaned mask alone, with no table mask."""
    return drop_intrusions(not_felt, r, None, max_balls)[0]


def find_balls(not_felt, r, coverage=BALL_COVERAGE, separation=BALL_SEPARATION,
               table=None, drop=True):
    """One centre per ball, by asking where a ball-sized disc fits the mask.

    Correlating a filled disc against the mask scores every pixel by the
    fraction of that disc which is non-felt - how well a ball centred there
    explains what was seen. Taking the peaks greedily and suppressing a
    neighbourhood after each separates touching balls: the score stays high
    across a whole cluster, but only one peak per ball survives suppression.

    `drop=False` says the mask has already been through drop_intrusions, which
    is what a caller that wants to SEE the intrusions does - otherwise they are
    found, erased and forgotten in here.
    """
    mask = not_felt
    if drop:
        mask = drop_intrusions(mask, r, table)[0]
    mask = mask.astype(np.float32) / 255.0
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
    balls = find_balls(not_felt, ball_r, table=table_mask)
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