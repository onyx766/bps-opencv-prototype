"""
BPS tuning tool - point it at footage and it makes the pipeline run on that
footage.

What actually needs tuning per venue is the FELT WINDOW. server.py's FELT_LOW /
FELT_HIGH are absolute HSV bounds measured on one blue cloth under one light,
with deliberately narrow saturation and value windows (the docstring there
explains why a median-plus-tolerance window was abandoned: it had to be opened
so wide that dark blue balls fell inside it). Point that window at another table
and it matches no felt at all, table_region() returns the largest blob of
nothing, and every stage downstream is scored against a table that is not
there. Nothing else in the pipeline fails as quietly.

So this tool fits the window to the actual cloth, and judges each candidate by
running the REAL detector over real frames - main.calibrate() and
main.detect_frame(), the same two calls the pipeline makes - rather than by any
separate notion of what felt looks like. A window that swallows the blue balls
scores badly because the balls stop being found, which is the failure mode worth
guarding against and the one a static threshold cannot see.

Everything else it settles is arithmetic or measurement that already existed:
the stride down to ~5 fps, the ball radius, and six pocket coordinates seeded
from the table rectangle.

Pockets stay HUMAN-CONFIRMED by default. detect_pocket.py argues at length that
a pocket is the hardest thing on the table to find automatically, and measuring
it here agrees: what is seeded is not a pocket found in the image but a corner
of the table's own rectangle, and on game.mp4 those corners sit 15 to 70 pixels
from the hand-clicked answer - one to three and a half ball radii, and a
different offset at every pocket, so there is no constant to correct by. The
seeds are worth having because they put a marker near each pocket for the eye to
find, not because they are right. The click window opens on them and they can be
dragged. --auto skips that for headless and Pi runs and says so loudly.

Usage:
    python tune.py clips/                    # a folder of clips
    python tune.py "footage/*.mkv"           # a glob (PowerShell does not expand)
    python tune.py game.mp4 --dry-run        # report only, write nothing
    python tune.py clips/ --auto             # accept seeded pockets, no GUI
    python tune.py clips/ --out rig2.json --config rig2_config.json

Writes bps_config.json (felt_low, felt_high, ball_r, target_fps) and
pockets.json, then verifies by running the detector back over the footage and
printing READY or NOT READY with the numbers behind it.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

import detect_pocket
import main as bps
from server import felt_mask, sample_felt, table_region

#: Frames pulled from across the footage to fit and score against. Balls move
#: and cloth does not, so a median over a spread of frames is a measurement of
#: the table; a single frame is a measurement of one rack.
PROBE_FRAMES = 12

#: The first stretch of a recording is skipped. Camera0001 files carry a
#: systematic ~1.27 s timestamp gap at startup, and the ELP opens on an
#: unsettled exposure - neither is what the cloth looks like during play.
SKIP_HEAD = 0.05

#: Every Nth pixel in each axis when collecting cloth pixels. The window comes
#: from percentiles over millions of pixels; reading all of them buys no
#: accuracy and costs seconds per frame at 1920x1200.
PIXEL_STEP = 4

#: Hue distance from the seed, in OpenCV's 0-179 scale, for a pixel to count as
#: a candidate cloth pixel. The shipped window spans 86-107, so +/-10 about its
#: centre - wide enough for the cloth's own spread, tight enough that the felt
#: and the balls are not one population.
HUE_SPAN = 10

#: Fraction of the seed's saturation and value the FIRST-PASS window reaches
#: down to. That pass only has to find the table well enough to measure inside
#: it, so it is deliberately loose - the window that gets used is cut from the
#: pixels it encloses, not from this.
ROUGH_FLOOR = 0.6

#: Table-mask erosion, matching what main.py passes to calibrate() at every
#: call site. Tuning against a different value would tune against a different
#: table outline than the pipeline sees.
SHRINK = 18

#: Fraction of the frame the eroded table may cover and still be a table. Below
#: the floor the window is matching specks; above the ceiling it is matching
#: most of the room, which is what an over-wide window does.
MIN_COVER, MAX_COVER = 0.15, 0.95

#: Sane ball radius in pixels. estimate_ball_radius() falls back to 20 when it
#: finds nothing to measure, so a plausible-looking number is not proof on its
#: own - but an implausible one is proof of failure.
MIN_BALL_R, MAX_BALL_R = 5, 120

#: Ball counts that a frame of pool can honestly produce. Sixteen is a fresh
#: rack; a single ball late in a game is real; thirty is noise being counted.
MIN_BALLS, MAX_BALLS = 4, 16

#: Percentile pairs tried when cutting the window out of the cloth
#: distribution, tightest last. A wider pair follows the cloth further into its
#: own shadows; a tighter one is better protection against a ball whose hue
#: sits beside the cloth. Which one wins is decided by the detector, not here.
TIGHTNESS = ((0.1, 99.9), (0.5, 99.5), (2.0, 98.0), (5.0, 95.0))


def probe_clips(paths):
    """Per-clip resolution, fps and length, plus whether they can be chained.

    ClipChain exits on a resolution mismatch rather than degrading, because
    pockets are pixel coordinates - so it is worth saying which clip disagrees
    before the run rather than after.
    """
    clips = []
    for path in paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"Could not open video: {path}")
        info = {
            "path": path,
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "bytes": os.path.getsize(path),
        }
        info["seconds"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0.0
        cap.release()
        clips.append(info)
    return clips


def report_clips(clips):
    print("1. FOOTAGE")
    total_s = total_b = 0
    for c in clips:
        print(f"   {os.path.basename(c['path'])}: {c['w']}x{c['h']}  "
              f"{c['fps']:.2f} fps  {c['frames']} frames  "
              f"{c['seconds'] / 60.0:.1f} min  {c['bytes'] / 1e9:.2f} GB")
        total_s += c["seconds"]
        total_b += c["bytes"]
    if len(clips) > 1:
        print(f"   total: {len(clips)} clips, {total_s / 60.0:.1f} min, "
              f"{total_b / 1e9:.2f} GB")

    sizes = {(c["w"], c["h"]) for c in clips}
    if len(sizes) > 1:
        listed = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
        sys.exit(f"   FAIL: clips are not all the same size ({listed}).\n"
                 f"         main.py refuses to chain them, because pocket\n"
                 f"         coordinates are pixels. Tune each size separately.")

    rates = [c["fps"] for c in clips if c["fps"] > 0]
    if rates and max(rates) - min(rates) > 0.5:
        print(f"   WARNING: frame rates differ ({min(rates):.1f}-{max(rates):.1f} "
              f"fps) - the sampling stride is one number, so clocks will drift.")
    return clips[0]["w"], clips[0]["h"], (rates[0] if rates else 0.0)


def stride_for(src_fps, target_fps):
    """The frame-index stride main.py will pick for this source."""
    if src_fps <= 0:
        return 1, 0.0
    stride = max(1, int(round(src_fps / target_fps)))
    return stride, src_fps / stride


def sample_frames(paths, count):
    """`count` sample points spread across the footage, as (index, frame, next).

    Each point carries the frame AND the one straight after it. A felt window
    cut too tight turns glare on the cloth into a ball that blinks; the blink
    is only visible between adjacent frames, so the pair is what makes it
    measurable. `next` is None at the very end of the footage.

    Seeking is by frame index through ClipChain, which maps an index onto the
    clip that holds it - so a spread over several clips is a spread over the
    game, not over the first file.
    """
    cap = bps.open_video(paths)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picked = []

    if total > 0:
        first = int(total * SKIP_HEAD)
        wanted = np.linspace(first, max(first + 1, total - 2), count)
        for idx in sorted({int(i) for i in wanted}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ok_next, follower = cap.read()
            picked.append((idx, frame, follower if ok_next else None))
    else:
        # Some camera-written containers do not carry a frame count. Read
        # forward instead of seeking, which is slower but always true.
        idx, held = 0, None
        while len(picked) < count:
            ok, frame = cap.read()
            if not ok:
                break
            if held is not None:
                picked.append((idx - 1, held, frame))
                held = None
            elif idx % 200 == 0:
                held = frame
            idx += 1

    cap.release()
    if not picked:
        sys.exit("Could not read any frames from the footage.")
    return picked


def felt_seed(frames):
    """Median HSV of the cloth, over the centre patch of every sampled frame.

    sample_felt() is the same patch the pipeline reports, and taking it over a
    spread of frames means a ball parked in the middle of any one of them
    cannot move the answer.
    """
    seeds = []
    for _idx, frame, _next in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = frame.shape[:2]
        seeds.append(sample_felt(hsv, H, W))
    return np.median(np.array(seeds), axis=0)


def felt_candidates(frames, seed):
    """Cloth pixels, measured INSIDE the table rather than across the frame.

    Selecting by hue alone over a whole frame is what a first version of this
    did, and it was wrong in the exact way server.py warns about: every dark
    bluish thing in the room - shadow under a rail, a dim wall, the black 8 -
    shares the cloth's hue, so the percentiles ran down to a saturation floor
    of 39 against a shipped floor of 215, and the detector then found more
    balls than a table can hold.

    So the table is found first with a deliberately loose window, and the
    percentiles are taken only over what is inside it. Cloth is the
    overwhelming majority of that area, which is what makes the percentiles
    mean the cloth and not its neighbours.
    """
    hue = float(seed[0])
    rough_low = (int(max(0, hue - HUE_SPAN)),
                 int(max(0, seed[1] * ROUGH_FLOOR)),
                 int(max(0, seed[2] * ROUGH_FLOOR)))
    rough_high = (int(min(179, hue + HUE_SPAN)), 255, 255)

    chunks = []
    for _idx, frame, _next in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = frame.shape[:2]
        table = table_region(felt_mask(hsv, rough_low, rough_high), H, W, SHRINK)
        inside = table[::PIXEL_STEP, ::PIXEL_STEP] > 0
        if not inside.any():
            continue
        pixels = hsv[::PIXEL_STEP, ::PIXEL_STEP][inside].astype(np.int16)
        gap = np.abs(pixels[:, 0] - hue)
        gap = np.minimum(gap, 180 - gap)           # hue is a circle
        chunks.append(pixels[gap <= HUE_SPAN])

    if not chunks:
        return None, (rough_low, rough_high)
    return np.vstack(chunks), (rough_low, rough_high)


def window_from(candidates, lo_pct, hi_pct):
    """One HSV window cut out of the cloth distribution at these percentiles."""
    lo = np.percentile(candidates, lo_pct, axis=0)
    hi = np.percentile(candidates, hi_pct, axis=0)
    ceilings = (179, 255, 255)
    low, high = [], []
    for i in range(3):
        top = ceilings[i]
        lo_i = int(max(0, min(top, math.floor(lo[i]))))
        hi_i = int(max(0, min(top, math.ceil(hi[i]))))
        # inRange cannot express an empty axis, and cloth flat enough to put
        # every pixel on one value - saturation pinned at 255 under a bright
        # light - produces exactly that. Widen downward at the ceiling, since
        # there is nothing above it to widen into.
        if hi_i <= lo_i:
            if hi_i >= top:
                lo_i, hi_i = max(0, top - 1), top
            else:
                hi_i = lo_i + 1
        low.append(lo_i)
        high.append(hi_i)
    return tuple(low), tuple(high)


def rectangularity(table):
    """Largest table blob's area over its minimum bounding rectangle's.

    A pool table seen from overhead is a rectangle. Whatever else a wrong
    window matches - a wall, a light, scattered speckle - is not, and this is
    the cheapest way to say so without knowing anything about the venue.
    """
    cnts, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    blob = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(blob)
    (_c, (rw, rh), _a) = cv2.minAreaRect(blob)
    box = rw * rh
    return float(area / box) if box > 0 else 0.0


def score_window(frames, low, high):
    """Run the real detector with this window and say how well it worked.

    Returns a dict with the measurements and a score, or a dict carrying only
    `reject` when the window fails a gate outright. Scoring on detections
    rather than on the mask is the point: a window wide enough to swallow a
    dark blue ball looks fine as a mask and loses a ball, and only the
    detector can tell the difference.
    """
    covers, rects, radii, counts, flicker = [], [], [], [], []
    for _idx, frame, follower in frames:
        H, W = frame.shape[:2]
        cal = bps.calibrate(frame, low, high, SHRINK)
        table = cal["table"]
        covers.append(np.count_nonzero(table) / float(H * W))
        rects.append(rectangularity(table))
        radii.append(cal["r"])
        balls, _intrusions = bps.detect_frame(frame, cal)
        counts.append(len(balls))
        if follower is not None:
            after, _ = bps.detect_frame(follower, cal)
            flicker.append(abs(len(after) - len(balls)))

    cover = float(np.median(covers))
    rect = float(np.median(rects))
    radius = int(np.median(radii))
    balls = float(np.median(counts))
    # Counts, not positions: two adjacent frames are 0.2s apart at the rate
    # the rules run at, so a ball mid-shot has moved several radii and would
    # fail a position match. What it has NOT done is appear or disappear.
    churn = float(np.mean(flicker)) if flicker else 0.0
    out = {"low": low, "high": high, "cover": cover, "rect": rect,
           "r": radius, "balls": balls, "churn": churn}

    if not MIN_COVER <= cover <= MAX_COVER:
        out["reject"] = f"table covers {cover * 100:.0f}% of frame"
        return out
    if not MIN_BALL_R <= radius <= MAX_BALL_R:
        out["reject"] = f"ball radius {radius}px"
        return out
    if not MIN_BALLS <= balls <= MAX_BALLS:
        out["reject"] = f"{balls:.0f} balls per frame"
        return out

    # Ball COUNT is a gate and not a score, deliberately. Both failure modes
    # move it: a window so wide it swallows a dark ball loses one, and a
    # window so tight that glare survives invents one. Without an answer key
    # there is nothing here that can tell those apart, so preferring "found
    # more balls" would be as likely to reward phantoms as real detections.
    # Steadiness can tell them apart, because a phantom blinks and a ball
    # does not, so that is what ranks the windows that pass the gates.
    out["score"] = rect / (1.0 + churn)
    return out


def fit_felt(frames):
    """The best-scoring felt window for this footage, and every window tried.

    The window shipped in server.py is scored alongside the fitted ones and
    wins ties, so this can never swap a working setting for one that merely
    measured well - on the footage it was tuned on, it should win outright.
    """
    seed = felt_seed(frames)
    hue = float(seed[0])
    if hue < HUE_SPAN or hue > 179 - HUE_SPAN:
        # cv2.inRange takes a low and a high, so it cannot express a window
        # that wraps past 179 back to 0. Red cloth would need one.
        sys.exit(f"   FAIL: cloth hue is {hue:.0f}, at the end of the 0-179 "
                 f"scale.\n         A window across that wrap cannot be "
                 f"expressed as one HSV range;\n         this needs a code "
                 f"change, not a setting.")

    candidates, _rough = felt_candidates(frames, seed)
    entries = []
    if candidates is None:
        print("   the loose first pass found no table - only the shipped "
              "window can be judged")
    else:
        for lo_pct, hi_pct in TIGHTNESS:
            low, high = window_from(candidates, lo_pct, hi_pct)
            entries.append((f"{lo_pct}-{hi_pct} pct", low, high))
    entries.append(("server.py", tuple(bps.FELT_LOW), tuple(bps.FELT_HIGH)))

    tried = []
    for label, low, high in entries:
        result = score_window(frames, low, high)
        result["label"] = label
        tried.append(result)

    passed = [t for t in tried if "score" in t]
    # Reversed so that on a tie the LAST entry wins: the tighter percentile,
    # and ultimately the shipped window. A tie means the detector could not
    # tell them apart, and the safer choice is the one already in use.
    best = max(reversed(passed), key=lambda t: t["score"]) if passed else None
    return seed, best, tried


def report_felt(seed, best, tried):
    print("\n2. FELT WINDOW")
    print(f"   cloth measured at HSV {[int(v) for v in seed]}")
    for t in tried:
        verdict = (f"score {t['score']:.2f}" if "score" in t
                   else f"rejected: {t['reject']}")
        print(f"   {t['label']:>12}  {tuple(t['low'])}..{tuple(t['high'])}  "
              f"table {t['cover'] * 100:>3.0f}%  rect {t['rect']:.2f}  "
              f"r {t['r']:>3}px  balls {t['balls']:>4.1f}  "
              f"blink {t['churn']:.2f}  {verdict}")

    if best is None:
        print("   FAIL: no window gave a plausible table and ball count.")
        if all("balls per frame" in t.get("reject", "") for t in tried):
            print("         Every window found more balls than a table holds.")
            print("         An overlay burned into the picture reads as balls -")
            print("         tune on the raw recording, not on a result render.")
        return False
    kept = best["label"] == "server.py"
    print(f"   chosen: {tuple(best['low'])}..{tuple(best['high'])} "
          f"({best['label']})  ball radius {best['r']}px, "
          f"{best['balls']:.1f} balls per frame")
    if kept:
        print("   the shipped window already fits this footage - keeping it.")
    return True


def seed_pockets(table, shrink):
    """(six pocket seeds, how many were clamped to the frame edge), or None.

    Not a pocket detector. The table mask is eroded inward by `shrink`, so its
    corners sit inside the real rail - each seed is pushed back out along the
    line from the table centre to land nearer the cushion. The result is close
    enough to nudge and never close enough to trust unseen, which is why the
    click window opens on it.

    A table that runs off the frame is the case that needs the clamp: its
    contour stops at the image border (table_region() zeroes the border on
    purpose), so pushing that corner outward puts the seed off the image
    entirely. A count of how many landed there is worth more than the clamp -
    it means the camera cannot see that part of the table at all.
    """
    H, W = table.shape[:2]
    cnts, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    blob = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(blob)
    centre = rect[0]
    box = cv2.boxPoints(rect)

    def push_out(x, y):
        vx, vy = x - centre[0], y - centre[1]
        span = math.hypot(vx, vy) or 1.0
        return (x + vx / span * shrink, y + vy / span * shrink)

    corners = [push_out(x, y) for x, y in box]

    # The two long edges carry the side pockets. boxPoints returns the corners
    # in order, so edge i runs from corner i to corner i+1.
    edges = [(i, (i + 1) % 4) for i in range(4)]
    lengths = [math.dist(box[a], box[b]) for a, b in edges]
    long_pair = sorted(range(4), key=lambda i: lengths[i])[-2:]
    sides = []
    for i in long_pair:
        a, b = edges[i]
        sides.append(push_out((box[a][0] + box[b][0]) / 2.0,
                              (box[a][1] + box[b][1]) / 2.0))

    seeds, clamped = [], 0
    for x, y in corners + sides:
        cx = int(round(min(max(x, 0), W - 1)))
        cy = int(round(min(max(y, 0), H - 1)))
        if (cx, cy) != (int(round(x)), int(round(y))):
            clamped += 1
        seeds.append((cx, cy))
    return seeds, clamped


def pockets_in_frame(pockets, W, H):
    """Pockets sitting ON the frame edge, which means off the real table.

    A pocket coordinate is the thing every pot claim is measured against. One
    pinned to the image border is not a pocket, it is the point where the
    table left the picture, and a run against it scores pots at a place no
    ball can reach.
    """
    return [p for p in pockets
            if p.x <= 0 or p.y <= 0 or p.x >= W - 1 or p.y >= H - 1]


def settle_pockets(frame, seeds, auto):
    """The six pockets to save: seeded, then confirmed by hand unless --auto."""
    if seeds is None:
        print("   could not seed from the table outline - clicking from scratch")

    if auto:
        if seeds is None:
            print("   FAIL: --auto cannot place pockets without a table outline.")
            return None
        print("   --auto: accepting the seeded pockets WITHOUT confirmation.")
        print("   These are table-rectangle corners, not detected pockets, and")
        print("   measured 15-70px off the hand-clicked answer on known "
              "footage.")
        print("   Check pockets.png before trusting any score from this run.")
        return detect_pocket.name_pockets(seeds)

    print("   opening the click window - drag each marker onto its pocket")
    points = detect_pocket.select_pockets(frame.copy(), seed=seeds)
    if points is None:
        print("   cancelled - nothing saved.")
        return None
    return detect_pocket.name_pockets(points)


def write_config(path, updates):
    """Merge settings into bps_config.json, keeping what is already there.

    Read-modify-write rather than rewrite: the file also carries `game` and
    `mode`, which are decisions about the venue that this tool has no opinion
    about, and a `_comment` that explains the file to the next person.
    """
    existing = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing.update(updates)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
        fh.write("\n")
    return existing


def scan_gaps(path):
    """(frames seen, largest timestamp gap in seconds, where) for one clip.

    main.py samples by frame INDEX, so a recording that dropped a second of
    frames is replayed as though that second never happened - every window in
    logic.py counted in frames (settle, phantom, undo) silently covers more
    real time than it was tuned for. Only the external-HDD capture path has
    tested clean, so this is worth knowing per delivery.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0.0, 0
    previous, worst, worst_at, seen = None, 0.0, 0, 0
    while cap.grab():
        seen += 1
        now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if previous is not None and now > previous:
            gap = now - previous
            if gap > worst:
                worst, worst_at = gap, seen
        previous = now
    cap.release()
    return seen, worst, worst_at


def report_gaps(clips, src_fps):
    print("\n4. TIMESTAMP GAPS")
    expected = 1.0 / src_fps if src_fps > 0 else 0.0
    clean = True
    for c in clips:
        seen, worst, where = scan_gaps(c["path"])
        name = os.path.basename(c["path"])
        if expected and worst > expected * 3:
            clean = False
            print(f"   {name}: largest gap {worst:.3f}s at frame {where} "
                  f"(expected {expected:.3f}s) - DROPPED FRAMES")
        else:
            print(f"   {name}: largest gap {worst:.3f}s over {seen} frames - ok")
    if not clean:
        print("   Frames are sampled by index, so these gaps compress real time")
        print("   in every frame-counted window in logic.py. Ask which storage")
        print("   path recorded this - only the external HDD has tested clean.")


def main():
    ap = argparse.ArgumentParser(
        description="Fit the pipeline to new footage: felt window, ball "
                    "radius, sampling stride and the six pockets.")
    ap.add_argument("video", nargs="*",
                    help="clip(s), a folder or a glob")
    ap.add_argument("--out", default=detect_pocket.DEFAULT_POCKET_FILE,
                    help=f"where to write the pockets "
                         f"(default: {detect_pocket.DEFAULT_POCKET_FILE})")
    ap.add_argument("--config", default=bps.DEFAULT_CONFIG,
                    help=f"config file to update (default: {bps.DEFAULT_CONFIG})")
    ap.add_argument("--target-fps", type=float, default=bps.TARGET_FPS,
                    help=f"rate the rules were tuned at (default {bps.TARGET_FPS})")
    ap.add_argument("--frames", type=int, default=PROBE_FRAMES,
                    help=f"frames sampled across the footage (default {PROBE_FRAMES})")
    ap.add_argument("--auto", action="store_true",
                    help="accept the seeded pockets without the click window")
    ap.add_argument("--repick", action="store_true",
                    help="re-place the pockets even if saved ones still fit")
    ap.add_argument("--no-gap-scan", action="store_true",
                    help="skip the timestamp-gap pass (it reads every frame)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report everything, write nothing")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if not args.video:
        found = bps.find_default_video(here)
        if found is None:
            sys.exit("No video given and none found here - "
                     'pass one: python tune.py "clips/*.mkv"')
        paths = [found]
    else:
        paths = bps.expand_clips(args.video, here)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    cfg_path = args.config if os.path.isabs(args.config) \
        else os.path.join(here, args.config)

    clips = probe_clips(paths)
    W, H, src_fps = report_clips(clips)

    stride, play_fps = stride_for(src_fps, args.target_fps)
    frames = sample_frames(paths, args.frames)
    seed, best, tried = fit_felt(frames)
    if not report_felt(seed, best, tried):
        sys.exit(1)

    print("\n3. SAMPLING")
    print(f"   {src_fps:.2f} fps source / {args.target_fps:.1f} target "
          f"-> stride {stride}, judged at {play_fps:.2f} fps")
    if play_fps > args.target_fps * 1.6 or play_fps < args.target_fps * 0.6:
        print(f"   WARNING: {play_fps:.2f} fps is far from the {args.target_fps:.1f} "
              f"fps every frame-counted\n            constant in logic.py was "
              f"tuned at - those windows will not mean\n            what they "
              f"say at this rate.")

    # Before the click window, not after: every automated check should be
    # finished by the time a person is asked to look at anything, rather than
    # making them sit through a full-file read once they already have.
    if not args.no_gap_scan:
        report_gaps(clips, src_fps)

    print("\n5. POCKETS")
    ref_idx, ref_frame, _after = frames[len(frames) // 2]
    # A fixed camera that only needed its felt re-fitted - a bulb changed, the
    # room got darker - has not moved its pockets, and re-placing six correct
    # coordinates by hand is how a correct calibration gets made worse.
    pockets = None if args.repick else detect_pocket.load_pockets(out_path, W, H)
    seeds, clamped = None, 0

    if pockets:
        print(f"   reusing the {len(pockets)} in {os.path.basename(out_path)}"
              f" - pass --repick to place them again")
    else:
        cal = bps.calibrate(ref_frame, best["low"], best["high"], SHRINK)
        seeded = seed_pockets(cal["table"], SHRINK)
        seeds, clamped = seeded if seeded else (None, 0)
        if seeds:
            print(f"   seeded 6 from the table outline on frame {ref_idx}")
        if clamped:
            print(f"   WARNING: {clamped} of 6 landed on the frame edge - the table")
            print("            runs out of shot there, so those pockets are not in")
            print("            view. Re-aim the camera rather than placing them.")

    if args.dry_run:
        print("   --dry-run: nothing written, no click window.")
        return

    if not pockets:
        pockets = settle_pockets(ref_frame, seeds, args.auto)
        if pockets is None:
            sys.exit(1)
        detect_pocket.save_pockets(out_path, pockets, W, H, paths[0])
        print(f"   saved to {os.path.basename(out_path)}: "
              + ", ".join(f"{p.name}({p.x},{p.y})" for p in pockets))

    preview = os.path.join(here, "pockets.png")
    marked = ref_frame.copy()
    detect_pocket.draw_pockets(marked, pockets, scale=max(0.6, H / 1080.0))
    if cv2.imwrite(preview, marked):
        print(f"   preview: {os.path.basename(preview)}")

    updates = {
        "felt_low": list(best["low"]),
        "felt_high": list(best["high"]),
        "ball_r": int(best["r"]),
        "target_fps": args.target_fps,
    }
    write_config(cfg_path, updates)
    print(f"\n6. WROTE {os.path.basename(cfg_path)}")
    for key, value in updates.items():
        print(f"   {key}: {value}")

    # Verify with the pockets in place: cutting the pocket holes out of the
    # table mask changes what the detector sees, so the numbers above were
    # measured against a table this run will not actually use.
    checks = []
    for _idx, frame, _next in frames:
        cal = bps.calibrate(frame, best["low"], best["high"], SHRINK,
                            ball_r=best["r"], pockets=pockets)
        balls, intrusions = bps.detect_frame(frame, cal)
        checks.append((len(balls), len(intrusions)))
    found = [c[0] for c in checks]
    busy = sum(1 for c in checks if c[1])

    print("\n7. VERIFY")
    print(f"   balls per frame: min {min(found)}, median "
          f"{int(np.median(found))}, max {max(found)} over {len(found)} frames")
    print(f"   frames with something reaching over the table: {busy}/{len(checks)}")

    # Judged on the median and on how OFTEN the count is impossible, never on
    # the worst frame. An arm reaching over the table leaves blobs behind that
    # no felt window can prevent, so a single frame above sixteen says
    # something about that moment; a third of them says something about the
    # calibration.
    over = sum(1 for n in found if n > MAX_BALLS)
    edge = pockets_in_frame(pockets, W, H)
    counts_ok = (MIN_BALLS <= np.median(found) <= MAX_BALLS
                 and over <= len(found) // 3)
    ready = counts_ok and not edge
    if over:
        print(f"   {over}/{len(found)} frames counted more than {MAX_BALLS} "
              f"balls, which a table cannot hold")
    if edge:
        print(f"   {len(edge)} pocket(s) on the frame edge: "
              + ", ".join(p.name for p in edge))

    if ready:
        spec = " ".join(f'"{v}"' if " " in v else v for v in args.video)
        print("\n   READY - run it:")
        print(f"     python main.py {spec}".rstrip())
    else:
        print("\n   NOT READY")
        if not counts_ok:
            print("   The ball count is not plausible for a game of pool.")
        if edge:
            print("   Pockets on the frame edge are where the table left the")
            print("   picture, not where a ball goes down - every pot measured")
            print("   against them would be measured against nothing.")
        print("   Check pockets.png, then re-run without --auto to place them")
        print("   by hand, or re-aim the camera so the whole table is in shot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
