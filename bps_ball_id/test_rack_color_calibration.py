"""
BPS Pro — test harness for rack-time colour calibration.

Renders SYNTHETIC racks directly in HSV (no cv2, no footage needed) under a
range of nasty venue conditions, then checks that every ball gets the right
number. Conditions mimic Blue Dolphin: blue cloth, dim warm light, glare spot,
per-ball shadow, camera hue drift.

Run:  python3 test_rack_color_calibration.py
"""

from __future__ import annotations

import numpy as np

from rack_color_calibration import (
    BallPalette,
    BreakWindowCalibrator,
    RackColorCalibrator,
    TrackedBall,
)

RNG = np.random.default_rng(7)

# Ground-truth "real world" hues, deliberately NOT identical to the pack seeds
# so the test cannot pass by matching the swatch.
TRUE_HUE = {
    1: 27, 2: 108, 3: 178, 4: 138, 5: 11, 6: 62, 7: 172,
    9: 27, 10: 108, 11: 178, 12: 138, 13: 11, 14: 62, 15: 172,
}
DARK_VAL = {4: 70, 7: 78, 12: 74, 15: 80}


def render_rack(numbers, hue_drift=0.0, dim=1.0, shadow=0.0, glare=True, size=520,
                hide_bands=(), no_cue=False, pitch=90, jitter=0):
    """Return (hsv_frame, positions, truth) with one ball per number."""
    palette = BallPalette.load()
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    # blue Olhausen cloth
    hsv[:, :, 0] = int((112 + hue_drift) % 180)
    hsv[:, :, 1] = 150
    hsv[:, :, 2] = int(120 * dim)
    hsv[:, :, 0] = (hsv[:, :, 0] + RNG.integers(-2, 3, hsv.shape[:2])) % 180

    positions, truth = [], {}
    r = 20
    cols = 5
    hide_bands = set(hide_bands)
    jx = (lambda: int(RNG.integers(-jitter, jitter + 1))) if jitter else (lambda: 0)
    for i, num in enumerate(numbers):
        cx = 70 + (i % cols) * pitch + jx()
        cy = 70 + (i // cols) * pitch + jx()
        spec = palette.spec(num)
        ys, xs = np.mgrid[cy - r - 2: cy + r + 3, cx - r - 2: cx + r + 3]
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        ball = d <= r

        # per-ball shadow: this is what kills absolute hue windows
        shade = 1.0 - shadow * RNG.random()
        if num == 8:
            hue, sat, val = 0, 20, int(40 * dim * shade)
        else:
            hue = (TRUE_HUE[num] + hue_drift) % 180
            base_v = DARK_VAL.get(num, 205)
            sat = 215 if num not in DARK_VAL else 190
            val = int(base_v * dim * shade)

        sl = (slice(cy - r - 2, cy + r + 3), slice(cx - r - 2, cx + r + 3))
        region = hsv[sl]
        region[..., 0] = np.where(ball, int(hue), region[..., 0])
        region[..., 1] = np.where(ball, sat, region[..., 1])
        region[..., 2] = np.where(ball, val, region[..., 2])

        if spec.kind == "stripe" and num not in hide_bands:
            band = ball & (d > r * 0.62)
            region[..., 1] = np.where(band, 12, region[..., 1])
            region[..., 2] = np.where(band, int(235 * dim * shade), region[..., 2])

        if glare:
            spot = d <= r * 0.28
            region[..., 1] = np.where(spot, 5, region[..., 1])
            region[..., 2] = np.where(spot, min(255, int(250 * dim)), region[..., 2])

        hsv[sl] = region
        positions.append((cx, cy, r))
        truth[len(positions) - 1] = num

    # cue ball, always last
    if no_cue:
        noise = RNG.integers(-3, 4, hsv.shape, dtype=np.int16)
        hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        hsv[:, :, 0] %= 180
        return hsv, positions, truth
    cx, cy = 70 + (len(numbers) % cols) * pitch, 70 + (len(numbers) // cols) * pitch
    ys, xs = np.mgrid[cy - r - 2: cy + r + 3, cx - r - 2: cx + r + 3]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    ball = d <= r
    sl = (slice(cy - r - 2, cy + r + 3), slice(cx - r - 2, cx + r + 3))
    region = hsv[sl]
    region[..., 1] = np.where(ball, 10, region[..., 1])
    region[..., 2] = np.where(ball, int(240 * dim), region[..., 2])
    hsv[sl] = region
    positions.append((cx, cy, r))

    noise = RNG.integers(-3, 4, hsv.shape, dtype=np.int16)
    hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    hsv[:, :, 0] %= 180
    return hsv, positions, truth


CONDITIONS = [
    ("bright neutral",      dict(hue_drift=0,   dim=1.0,  shadow=0.0)),
    ("dim venue",           dict(hue_drift=0,   dim=0.72, shadow=0.10)),
    ("warm cast +12",       dict(hue_drift=12,  dim=0.85, shadow=0.15)),
    ("cool cast -10",       dict(hue_drift=-10, dim=0.90, shadow=0.15)),
    ("dim + heavy shadow",  dict(hue_drift=8,   dim=0.65, shadow=0.28)),
]


def break_window_case() -> int:
    """
    Racked frame vs break window, same rack, same light.

    The racked frame is rendered the way Mike's 2026-09-09 rack shot actually
    came out: no cue ball in shot, balls touching, and three stripes with their
    band turned away. The break window is the same rack a second later -- cue
    ball on the table, balls apart, and rolling, so a different pair of stripes
    hides its band in each frame.
    """
    palette = BallPalette.load()
    numbers = palette.object_balls("8ball")
    stripes = [n for n in numbers if palette.spec(n).kind == "stripe"]
    failures = 0
    print("=== Break window vs racked frame (8-Ball) ===")

    # 1. the racked frame, as shot
    hsv, positions, truth = render_rack(
        numbers, dim=0.8, shadow=0.15,
        hide_bands={9, 13, 15}, no_cue=True, pitch=44,
    )
    racked = RackColorCalibrator(game="8ball").calibrate(hsv, positions, frame=10)
    racked_ok = sum(1 for i in truth if racked.assignments.get(i) == truth[i])
    print(f"  racked frame         {racked_ok}/{len(truth)} ids  conf={racked.confidence}")

    # 2. the break window: six frames, band visibility rotating
    cal = RackColorCalibrator(game="8ball")
    win = BreakWindowCalibrator(cal)
    for k in range(6):
        hidden = {stripes[k % len(stripes)], stripes[(k + 3) % len(stripes)]}
        hsv, positions, truth = render_rack(
            numbers, dim=0.8, shadow=0.15, hide_bands=hidden, pitch=90, jitter=3,
        )
        tracks = [
            TrackedBall(track_id=i, x=x, y=y, r=r)
            for i, (x, y, r) in enumerate(positions)
        ]
        win.observe(hsv, tracks, frame=20 + k)
    merged = win.finalize(frame=26)
    order = win.track_order if merged.ok else []
    win_ok = sum(
        1 for i, num in merged.assignments.items()
        if truth.get(order[i]) == num
    )
    print(
        f"  break window         {win_ok}/{len(truth)} ids  conf={merged.confidence}"
        f"  ({win.frames_used}/{win.frames_seen} frames merged)"
    )

    if win_ok <= racked_ok:
        print("      FAIL: break window did not beat the racked frame")
        failures += 1
    if win_ok < len(truth):
        print(f"      FAIL: break window missed {len(truth) - win_ok} balls")
        failures += 1
    print()
    return failures


def run() -> int:
    palette = BallPalette.load()
    failures = 0
    print(f"pack v{palette.version} ({palette.league_year})\n")

    for game in ("8ball", "9ball", "10ball"):
        numbers = palette.object_balls(game)
        print(f"=== {palette.label(game)} — {len(numbers)} object balls ===")
        for label, kw in CONDITIONS:
            hsv, positions, truth = render_rack(numbers, **kw)
            cal = RackColorCalibrator(game=game)
            res = cal.calibrate(hsv, positions, frame=100)
            got = res.assignments
            wrong = [
                (truth[i], got.get(i)) for i in truth if got.get(i) != truth[i]
            ]
            n_ok = len(truth) - len(wrong)
            status = "PASS" if not wrong else "FAIL"
            if wrong:
                failures += 1
            print(
                f"  {label:20s} {status}  {n_ok}/{len(truth)} ids  "
                f"offset {res.hue_offset:+.1f}  conf={res.confidence}"
                + (f"  wrong={wrong}" if wrong else "")
            )
            # the two hard pairs, called out explicitly
            for pair in ((3, 5), (4, 8), (11, 13), (7, 15)):
                if all(p in numbers for p in pair):
                    ok = all(
                        got.get(next(i for i in truth if truth[i] == p)) == p
                        for p in pair
                    )
                    if not ok:
                        print(f"      HARD PAIR {pair} MISSED")

        # mid-game identify() after calibration
        hsv, positions, truth = render_rack(numbers, hue_drift=9, dim=0.7, shadow=0.2)
        cal = RackColorCalibrator(game=game)
        cal.calibrate(hsv, positions, frame=1)
        mid_wrong = [
            (truth[i], cal.identify(hsv, *positions[i]))
            for i in truth
            if cal.identify(hsv, *positions[i]) != truth[i]
        ]
        print(
            f"  identify() mid-game  {'PASS' if not mid_wrong else 'FAIL'}  "
            f"{len(truth) - len(mid_wrong)}/{len(truth)}"
            + (f"  wrong={mid_wrong}" if mid_wrong else "")
        )
        if mid_wrong:
            failures += 1

        # fallback: occluded rack (too few balls) must reuse, not crash
        cal2 = RackColorCalibrator(game=game)
        cal2.calibrate(hsv, positions, frame=1)
        occluded = cal2.calibrate(hsv, positions[:3], frame=2)
        assert occluded.confidence == "reused", occluded
        assert occluded.reused_from_frame == 1
        # cold start with no history must fail cleanly
        cold = RackColorCalibrator(game=game).calibrate(hsv, positions[:2], frame=3)
        assert cold.ok is False and cold.confidence == "low"
        print("  fallback + cold start  PASS\n")

    failures += break_window_case()

    print("FAILURES:", failures)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
