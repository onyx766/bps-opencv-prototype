"""
BPS prototype - detect AND identify balls in a video, drawing ball numbers.

Three pieces, in order:
  1. DETECT   - server.py's felt mode finds where the balls are. The camera is
                fixed, so the felt colour and table outline are measured once
                and reused, which is most of the speed-up over per-frame work.
  2. TRACK    - a nearest-neighbour tracker gives each ball a stable id across
                frames. Needed twice over: bps_ball_id's break-window
                calibration takes track ids, and a per-track vote is what stops
                the drawn numbers flickering.
  3. IDENTIFY - bps_ball_id/rack_color_calibration.py assigns ball NUMBERS by
                relative hue rank under the venue's actual light, rather than by
                absolute colour swatch.

Usage:
    python main.py                        # game.mp4 -> result.mp4, with numbers
    python main.py clip.mp4               # a different video
    python main.py --game 9ball           # 8ball (default) | 9ball | 10ball
    python main.py --show                 # live preview window (q or ESC quits)
    python main.py --stride 5             # process every 5th frame (fast pass)
    python main.py --frames 300           # stop after 300 processed frames
    python main.py --csv balls.csv        # log frame,track,number,x,y,r
    python main.py --no-id                # positions only, skip identification

Press 'q' or ESC to stop early - the video written so far is still playable.
"""

import argparse
import csv
import os
import sys
import time
import cv2
import numpy as np

from server import (fill_holes, hough_alt, merge_detections, sample_felt,
                    split_blobs, table_region)

# bps_ball_id uses flat imports internally, so its folder goes on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "bps_ball_id"))
from rack_color_calibration import (BreakWindowCalibrator,  # noqa: E402
                                    RackColorCalibrator, TrackedBall)

# Tried in order when no video argument is given - first one that exists wins.
DEFAULT_VIDEOS = ["game.mp4", "test.mp4", "input.mp4"]
DEFAULT_OUTPUT = "result.mp4"


def find_default_video(folder):
    """First DEFAULT_VIDEOS candidate in folder, else any .mp4 there."""
    for name in DEFAULT_VIDEOS:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            return os.path.join(folder, name)
    return None


class Tracker:
    """Nearest-neighbour tracker: gives each ball an id that persists.

    Deliberately simple - this footage is 4.9 fps, so a ball in motion can jump
    a long way between frames and no cheap tracker will follow it reliably. It
    holds well while balls are at rest (most frames), which is when
    identification happens, and ids are allowed to churn during a shot.
    """

    def __init__(self, max_dist=110, max_misses=3):
        self.max_dist = max_dist
        self.max_misses = max_misses
        self._next_id = 0
        self.tracks = {}                   # id -> [x, y, r, misses]

    def update(self, balls):
        """Take this frame's (x,y,r) list, return [(track_id, x, y, r), ...]."""
        unmatched = dict(self.tracks)
        out = []

        # Greedy nearest match. Sort by distance so the closest pairs win first,
        # otherwise an early detection can steal a track from a better fit.
        pairs = []
        for di, (x, y, r) in enumerate(balls):
            for tid, (tx, ty, _tr, _m) in unmatched.items():
                d = (x - tx) ** 2 + (y - ty) ** 2
                if d <= self.max_dist ** 2:
                    pairs.append((d, di, tid))
        pairs.sort()

        taken_det, taken_trk = set(), set()
        assign = {}
        for _d, di, tid in pairs:
            if di in taken_det or tid in taken_trk:
                continue
            assign[di] = tid
            taken_det.add(di)
            taken_trk.add(tid)

        for di, (x, y, r) in enumerate(balls):
            tid = assign.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            self.tracks[tid] = [x, y, r, 0]
            out.append((tid, x, y, r))

        # Age out tracks that went unseen; a ball behind a player's arm comes
        # back a frame or two later and should keep its number.
        for tid in list(self.tracks):
            if tid not in taken_trk and tid not in {t for t, *_ in out}:
                self.tracks[tid][3] += 1
                if self.tracks[tid][3] > self.max_misses:
                    del self.tracks[tid]
        return out


class BallRegistry:
    """Carries ball NUMBERS forward by position, one number to one ball.

    The calibration solves all the identities at once, as a set - that is the
    whole point of relative-rank matching, and it is what makes the answer
    unique. Calling identify() per ball per frame throws that away: it picks
    each ball's nearest reference independently, so nothing stops three balls
    all coming back "cue".

    So the numbers are taken from the calibration once, then followed by
    position: each frame every detection is matched to at most one remembered
    ball, nearest first. Uniqueness holds by construction, and it survives the
    tracker losing an id mid-shot.
    """

    def __init__(self, max_move=160, max_misses=25):
        self.max_move = max_move
        self.max_misses = max_misses
        self.known = {}                    # number -> [x, y, misses]

    def seed(self, mapping):
        """mapping: {ball_number: (x, y)} straight from the calibration."""
        for number, (x, y) in mapping.items():
            self.known[number] = [float(x), float(y), 0]

    def assign(self, balls):
        """Return a list of number-or-None, aligned with `balls`."""
        result = [None] * len(balls)
        if not self.known:
            return result

        pairs = []
        for di, (x, y, _r) in enumerate(balls):
            for number, (kx, ky, _m) in self.known.items():
                d2 = (x - kx) ** 2 + (y - ky) ** 2
                if d2 <= self.max_move ** 2:
                    pairs.append((d2, di, number))
        pairs.sort()

        used_det, used_num = set(), set()
        for _d2, di, number in pairs:
            if di in used_det or number in used_num:
                continue
            result[di] = number
            used_det.add(di)
            used_num.add(number)
            x, y, _r = balls[di]
            self.known[number] = [float(x), float(y), 0]

        # A ball that stays unseen was pocketed or is under a player's arm.
        # Drop it eventually so its stale position cannot capture another ball.
        for number in list(self.known):
            if number not in used_num:
                self.known[number][2] += 1
                if self.known[number][2] > self.max_misses:
                    del self.known[number]
        return result


def calibrate(frame, felt_tol, shrink):
    """Measure the felt colour and the table outline once.

    A fixed overhead camera means neither changes between frames, and
    table_region() is the most expensive step in the pipeline - doing it per
    frame would waste most of the runtime.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]
    felt = sample_felt(hsv, H, W)
    tol = np.array(felt_tol)
    lo = np.clip(felt - tol, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    hi = np.clip(felt + tol, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    table = table_region(cv2.inRange(hsv, lo, hi), H, W, shrink)
    return {"lo": lo, "hi": hi, "table": table, "felt": felt}


def detect_frame(frame, cal, min_r, max_r, shape, alt_param2):
    """Per-frame half of the pipeline, using the cached calibration."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    felt_mask = cv2.inRange(hsv, cal["lo"], cal["hi"])

    not_felt = cv2.bitwise_and(cv2.bitwise_not(felt_mask), cal["table"])
    not_felt = cv2.morphologyEx(not_felt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    not_felt = fill_holes(not_felt)

    dt_balls, alt_balls = [], []
    if shape in ("hybrid", "dt"):
        dt_balls = split_blobs(not_felt, r_typ=(min_r + max_r) / 2.0)
        dt_balls = [(x, y, int(min(max(r, min_r), max_r))) for x, y, r in dt_balls]
    if shape in ("hybrid", "alt"):
        alt_balls = hough_alt(frame, cal["table"], not_felt, min_r, max_r,
                              param2=alt_param2)

    if shape == "dt":
        balls = dt_balls
    elif shape == "alt":
        balls = alt_balls
    else:
        balls = merge_detections(alt_balls, dt_balls, tol=min_r)
    balls.sort(key=lambda b: (b[1], b[0]))
    return balls


def draw(frame, items, index, total, fps_now, cal_status):
    """items: [(track_id, x, y, r, number_or_None), ...]"""
    for _tid, x, y, r, number in items:
        # Green once the ball has a number, red while it is still unknown.
        known = number is not None
        cv2.circle(frame, (x, y), r, (0, 200, 0) if known else (0, 0, 255), 2)

        label = "cue" if number == 0 else (str(number) if known else "?")
        scale = 0.5 if label == "cue" else 0.72
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        lx, ly = x - tw // 2, y - r - 8
        if ly - th < 2:                       # ball near the top edge
            ly = y + r + th + 6
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)

    named = sum(1 for it in items if it[4] is not None)
    hud = (f"frame {index}/{total}   balls {len(items)}   ids {named}   "
           f"{fps_now:.1f} fps   |  {cal_status}")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, hud, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help=f"input video (default: first of {', '.join(DEFAULT_VIDEOS)})")
    ap.add_argument("--out", default=DEFAULT_OUTPUT,
                    help=f"annotated output video (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (default 1 = all)")
    ap.add_argument("--start", type=int, default=0, help="first frame to read")
    ap.add_argument("--frames", type=int, default=0,
                    help="stop after N processed frames (0 = whole video)")
    ap.add_argument("--csv", default=None, help="also write ball positions to this CSV")
    ap.add_argument("--min-r", type=int, default=15)
    ap.add_argument("--max-r", type=int, default=30)
    ap.add_argument("--felt-tol", nargs=3, type=int, default=[18, 130, 90])
    ap.add_argument("--shape", choices=["hybrid", "alt", "dt"], default="hybrid")
    ap.add_argument("--alt-param2", type=float, default=0.5)
    ap.add_argument("--recalibrate", type=int, default=0,
                    help="re-measure felt/table every N frames (0 = once, the default)")
    ap.add_argument("--game", default="8ball", choices=["8ball", "9ball", "10ball"],
                    help="which ball set is in play (default 8ball)")
    ap.add_argument("--no-id", action="store_true",
                    help="skip ball identification, draw positions only")
    ap.add_argument("--id-window", type=int, default=40,
                    help="frames to merge for the break-window calibration")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    if args.video is None:
        in_path = find_default_video(here)
        if in_path is None:
            sys.exit(f"No video found in {here}\n"
                     f"Put one of {', '.join(DEFAULT_VIDEOS)} there, "
                     f"or pass a path: python main.py clip.mp4")
    else:
        in_path = args.video if os.path.isabs(args.video) else os.path.join(here, args.video)
        if not os.path.exists(in_path):
            sys.exit(f"Video not found: {in_path}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {in_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input:  {in_path}  ({W}x{H}, {total} frames, {src_fps:.1f} fps)")

    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read the first frame.")
    cal = calibrate(frame, args.felt_tol, shrink=18)
    print(f"Felt:   HSV {cal['felt'].astype(int).tolist()}  "
          f"(table mask covers {100 * (cal['table'] > 0).mean():.1f}% of frame)")

    # Writing at source fps / stride keeps the output playing at real speed.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, src_fps / args.stride), (W, H))
    if not writer.isOpened():
        sys.exit(f"Could not open the video writer for: {out_path}")

    csv_file = csv_writer = None
    if args.csv:
        csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track", "number", "x", "y", "r"])

    # Ball identification (bps_ball_id). The break-window calibrator merges the
    # best view of each tracked ball across several frames - one racked frame is
    # not enough, because a stripe facing away reads as a solid and the cue ball
    # is often not even on the table yet.
    tracker = Tracker()
    registry = BallRegistry()
    identifier = None if args.no_id else RackColorCalibrator(game=args.game)
    window = None if args.no_id else BreakWindowCalibrator(identifier)
    cal_status = "id: off" if args.no_id else "id: calibrating"
    best_cal = None                  # keep the strongest calibration seen so far

    idx = args.start
    processed = 0
    counts = []
    id_counts = []
    t0 = time.perf_counter()
    fps_now = 0.0
    first = True                     # the frame used for calibration is already read
    try:
        while True:
            if not first:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if args.stride > 1 and (idx - args.start) % args.stride:
                    continue
            first = False

            if args.recalibrate and processed and processed % args.recalibrate == 0:
                cal = calibrate(frame, args.felt_tol, shrink=18)

            balls = detect_frame(frame, cal, args.min_r, args.max_r,
                                 args.shape, args.alt_param2)
            counts.append(len(balls))
            tracked = tracker.update(balls)

            items = [(tid, x, y, r, None) for tid, x, y, r in tracked]
            if identifier is not None:
                if window is not None:
                    # Still building the calibration: feed this frame in, and
                    # finalize once enough frames have been merged. Only this
                    # stage needs HSV, so the conversion stops once calibrated.
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    window.observe(hsv, [TrackedBall(tid, x, y, r)
                                         for tid, x, y, r in tracked], frame=idx)
                    if window.frames_used >= args.id_window:
                        res = window.finalize(frame=idx, fps=src_fps)
                        better = res.ok and (
                            best_cal is None
                            or len(res.assignments) > len(best_cal.assignments))
                        if better:
                            # assignments is {sample index -> number}, and each
                            # sample carries the position it was measured at, so
                            # the set solution seeds the registry directly.
                            seed = {num: (res.samples[i].x, res.samples[i].y)
                                    for i, num in res.assignments.items()}
                            if res.cue_index is not None:
                                seed[0] = (res.samples[res.cue_index].x,
                                           res.samples[res.cue_index].y)
                            registry.seed(seed)
                            best_cal = res
                            cal_status = (f"id: {len(seed)} balls, "
                                          f"{res.confidence} conf")
                            print(f"\nIdentified {len(seed)} balls "
                                  f"({res.confidence} confidence, "
                                  f"hue offset {res.hue_offset:+.1f}) from "
                                  f"{window.frames_used} merged frames")
                            print(f"  numbers: {sorted(seed)}", flush=True)
                        elif not res.ok:
                            reason = "; ".join(res.reasons[:1]) or "unusable"
                            print(f"  calibration retry at frame {idx}: {reason}",
                                  flush=True)

                        # A "high" confidence full set is as good as it gets;
                        # anything less is worth another window later in the
                        # video, when the balls are spread and rolling.
                        if res.ok and res.confidence == "high":
                            window = None
                        else:
                            window = BreakWindowCalibrator(identifier)

                numbers = registry.assign(balls)

                # A ball struck hard travels further between frames than the
                # registry will match (this footage is 4.9 fps), so it would
                # stay unknown for good. Fall back to a colour read for those,
                # but only accept a number no other ball already holds - that
                # keeps the one-number-one-ball guarantee.
                if best_cal is not None and any(n is None for n in numbers):
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    taken = {n for n in numbers if n is not None}
                    for i, n in enumerate(numbers):
                        if n is not None:
                            continue
                        x, y, r = balls[i]
                        guess = identifier.identify(hsv, x, y, r)
                        if guess is not None and guess not in taken:
                            numbers[i] = guess
                            taken.add(guess)
                            registry.known[guess] = [float(x), float(y), 0]

                by_pos = {(x, y): n for (x, y, _r), n in zip(balls, numbers)}
                for i, (tid, x, y, r) in enumerate(tracked):
                    items[i] = (tid, x, y, r, by_pos.get((x, y)))

            id_counts.append(sum(1 for it in items if it[4] is not None))
            if csv_writer:
                for tid, x, y, r, number in items:
                    csv_writer.writerow([idx, tid, "" if number is None else number,
                                         x, y, r])

            processed += 1
            elapsed = time.perf_counter() - t0
            fps_now = processed / elapsed if elapsed else 0.0

            writer.write(draw(frame, items, idx, total, fps_now, cal_status))

            if args.show:
                preview = cv2.resize(frame, (W // 2, H // 2))
                cv2.imshow("BPS - video", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Stopped by user.")
                    break

            if processed % 50 == 0:
                print(f"  {processed} frames  |  frame {idx}/{total}  "
                      f"|  {len(balls)} balls, {id_counts[-1]} named  "
                      f"|  {fps_now:.1f} fps", flush=True)

            if args.frames and processed >= args.frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        writer.release()
        if csv_file:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    if counts:
        print(f"\nProcessed {processed} frames in {time.perf_counter() - t0:.1f}s "
              f"({fps_now:.1f} fps)")
        print(f"Balls per frame: min {min(counts)}  max {max(counts)}  "
              f"mean {sum(counts) / len(counts):.1f}")
        if id_counts and not args.no_id:
            print(f"Numbered per frame: min {min(id_counts)}  max {max(id_counts)}  "
                  f"mean {sum(id_counts) / len(id_counts):.1f}")
            if best_cal is not None:
                # numbers() is {ball number -> sample index}, so the keys are
                # the ball numbers.
                got = sorted(best_cal.numbers().keys())
                print(f"Best calibration: {len(got)} balls, "
                      f"{best_cal.confidence} confidence, at frame {best_cal.frame}")
                print(f"Ball numbers resolved: {got}")
    print(f"Saved:  {out_path}")
    if args.csv:
        print(f"Saved:  {args.csv}")


if __name__ == "__main__":
    main()
