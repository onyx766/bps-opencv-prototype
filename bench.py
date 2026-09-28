"""Does the pipeline keep up live on this machine? A yes or a no.

Runs exactly what main.py runs per frame - detect, track, identify, score -
with no window and no video written, and times each stage. Frames the sampler
skips are grabbed, not decoded, just as main.py does.

  A file   is read as fast as possible. YES when a second of video costs less
           than a second of wall time (the real-time factor is 1.0 or more).
           Decoding a saved H.264 file is not the same work as a camera's
           MJPEG, so on the Pi run this on the camera too.
  A camera is read as it delivers. YES when the target rate of frames was
           actually judged, every second, for the whole run.

Run it for a few minutes on the Pi: a Pi that keeps up for thirty seconds and
then throttles hot does not keep up. The temperature and throttle state are
printed at the end when `vcgencmd` is available.

Usage:
    python bench.py game1.mp4                 # 2 minutes of the clip
    python bench.py game1.mp4 --seconds 600   # 10 minutes
    python bench.py --camera 0 --seconds 300  # the table camera, 5 minutes
    python bench.py clips/ --json bench_pi.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import cv2

import detect_pocket
import logic
import main as bps
from server import FELT_HIGH, FELT_LOW
from track_identity import BallClassRegistry

STAGES = ("read", "detect", "track", "identify", "score")


def machine():
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as fh:
            return fh.read().strip("\x00\n ")
    except OSError:
        return f"{platform.system()} {platform.machine()} {platform.processor()}".strip()


def vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", nargs="*", help="clip(s), a folder or a glob")
    ap.add_argument("--camera", type=int, default=None, help="camera index instead")
    ap.add_argument("--seconds", type=float, default=120,
                    help="how much to run: video seconds for a file, wall "
                         "seconds for a camera (default 120)")
    ap.add_argument("--target-fps", type=float, default=bps.TARGET_FPS,
                    help=f"frames judged per second (default {bps.TARGET_FPS:g})")
    ap.add_argument("--pockets", default=None,
                    help="pocket file (default pockets.json, or "
                         "pockets_live.json for a camera)")
    ap.add_argument("--no-id", action="store_true", help="skip identification")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()

    live = args.camera is not None
    if live:
        cap = bps.open_camera(args.camera)
        if cap is None:
            sys.exit(f"Could not open camera {args.camera}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, bps.LIVE_CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, bps.LIVE_CAPTURE_H)
        for _ in range(bps.LIVE_WARMUP):
            cap.read()
        source = f"camera:{args.camera}"
    else:
        if not args.video:
            default = bps.find_default_video(here)
            if default is None:
                sys.exit("No video given and none found here.")
            args.video = [default]
        clips = bps.expand_clips(args.video, here)
        cap = bps.open_video(clips)
        source = clips[0] if len(clips) == 1 else f"{len(clips)} clips"

    ok, frame = cap.read()
    if not ok:
        sys.exit("No frames from the source.")
    H, W = frame.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not 1.0 <= src_fps <= 240.0:
        src_fps = bps.LIVE_FPS if live else 25.0
    stride = max(1, round(src_fps / args.target_fps)) if args.target_fps > 0 else 1
    play_fps = args.target_fps if live else src_fps / stride

    pocket_file = args.pockets or (bps.LIVE_POCKET_FILE if live
                                   else detect_pocket.DEFAULT_POCKET_FILE)
    pockets = detect_pocket.load_pockets(os.path.join(here, pocket_file), W, H)
    cal = bps.calibrate(frame, tuple(FELT_LOW), tuple(FELT_HIGH), shrink=18,
                        pockets=pockets, block=bps.POCKET_BLOCK)
    tracker = bps.Tracker()
    registry = None if args.no_id else BallClassRegistry()
    session = logic.GameSession(pockets, cal["r"], fps=src_fps, sample_fps=play_fps,
                                first_rack_confirmed=True, started=True)

    print(f"Machine: {machine()}")
    print(f"Python {platform.python_version()}, OpenCV {cv2.__version__}")
    print(f"Source:  {source}  {W}x{H}  {src_fps:.1f} fps")
    print(f"Judging: {play_fps:.1f} fps "
          + ("by the clock" if live else f"(1 in {stride} frames)")
          + f", pockets {'loaded' if pockets else 'NOT SET - pots not exercised'}")
    print(f"Running {args.seconds:g} s ...", flush=True)

    times = {s: [] for s in STAGES}
    grab_ms = []
    idx = processed = 0
    per_second = {}
    t0 = time.perf_counter()
    taken_at = t0
    period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0
    end_frame = int(args.seconds * src_fps)
    while True:
        now = time.perf_counter()
        if live and now - t0 >= args.seconds:
            break
        if not live and idx >= end_frame:
            break
        if idx:
            skip = (now - taken_at < period) if live else (idx % stride != 0)
            a = time.perf_counter()
            if skip:
                ok = cap.grab()
                grab_ms.append((time.perf_counter() - a) * 1000)
                idx += 1
                if not ok and not live:
                    break
                continue
            ok, frame = cap.read()
            times["read"].append((time.perf_counter() - a) * 1000)
            if not ok:
                if live:
                    continue
                break
        taken_at = time.perf_counter()

        a = time.perf_counter()
        balls, intrusions = bps.detect_frame(frame, cal)
        b = time.perf_counter()
        tracked = tracker.update(balls)
        c = time.perf_counter()
        labels = {}
        if registry is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            labels = registry.update(tracked, hsv, frame=idx)
        d = time.perf_counter()
        session.update(tracked, labels, idx,
                       intruding=any(it.bulky for it in intrusions))
        e = time.perf_counter()
        for name, ms in zip(STAGES[1:], (b - a, c - b, d - c, e - d)):
            times[name].append(ms * 1000)

        processed += 1
        idx += 1
        sec = int(time.perf_counter() - t0)
        per_second[sec] = per_second.get(sec, 0) + 1
        if processed % 50 == 0:
            wall = time.perf_counter() - t0
            print(f"  {processed} judged, {wall:.0f}s wall, "
                  f"{idx / src_fps / wall:.2f}x real time" if not live else
                  f"  {processed} judged, {wall:.0f}s wall, "
                  f"{processed / wall:.1f} fps", flush=True)
    wall = time.perf_counter() - t0
    cap.release()

    video_s = idx / src_fps
    realtime = video_s / wall if wall else 0.0
    judged_fps = processed / wall if wall else 0.0
    full = [n for s, n in per_second.items() if 0 < s < int(wall)]
    slow_seconds = sum(1 for n in full if n < 0.95 * args.target_fps)
    if live:
        keeps_up = judged_fps >= 0.95 * args.target_fps and slow_seconds == 0
    else:
        keeps_up = realtime >= 1.0
    total_ms = [sum(v) for v in zip(*(times[s] for s in STAGES[1:]))]

    print(f"\n{processed} frames judged in {wall:.1f}s  ({judged_fps:.1f} fps"
          + (f", target {args.target_fps:g}" if live else "") + ")")
    if not live:
        print(f"{video_s:.1f}s of video in {wall:.1f}s = {realtime:.2f}x real time")
    else:
        print(f"seconds that judged fewer than {0.95 * args.target_fps:.1f} "
              f"frames: {slow_seconds} of {len(full)}")
    print(f"\n{'stage':<10}{'mean ms':>9}{'p95 ms':>9}")
    for name in STAGES:
        v = times[name]
        if v:
            print(f"{name:<10}{sum(v) / len(v):>9.1f}{pct(v, 0.95):>9.1f}")
    if grab_ms:
        print(f"{'skip/grab':<10}{sum(grab_ms) / len(grab_ms):>9.1f}{pct(grab_ms, 0.95):>9.1f}"
              f"   x{stride - 1 if not live else round(len(grab_ms) / max(1, processed))} "
              f"per judged frame")
    if total_ms:
        print(f"{'pipeline':<10}{sum(total_ms) / len(total_ms):>9.1f}"
              f"{pct(total_ms, 0.95):>9.1f}   budget {1000 / play_fps:.0f} ms")

    temp, throttled = vcgencmd("measure_temp"), vcgencmd("get_throttled")
    if temp:
        print(f"\nPi: {temp}  {throttled}"
              + ("  (THROTTLED - cooling needed)" if throttled
                 and throttled.split("=")[-1] not in ("0x0", "0") else ""))

    print(f"\nKeeps up live: {'YES' if keeps_up else 'NO'}")
    if not keeps_up:
        print("  Options: a lower --target-fps if the rules still hold at it, a "
              "smaller camera resolution, or cropping to the table before detect.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"machine": machine(), "source": source, "size": [W, H],
                       "src_fps": src_fps, "target_fps": args.target_fps,
                       "judged": processed, "wall_s": round(wall, 2),
                       "judged_fps": round(judged_fps, 2),
                       "realtime": None if live else round(realtime, 3),
                       "stage_mean_ms": {s: round(sum(v) / len(v), 2)
                                         for s, v in times.items() if v},
                       "pipeline_p95_ms": round(pct(total_ms, 0.95), 2),
                       "pi_temp": temp, "pi_throttled": throttled,
                       "keeps_up": keeps_up}, fh, indent=2)
        print(f"Saved:  {args.json}")
    sys.exit(0 if keeps_up else 1)


if __name__ == "__main__":
    main()
