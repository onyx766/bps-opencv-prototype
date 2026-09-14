"""
BPS prototype - find the balls in a video and say what each one is: CUE, 8,
STRIPE or SOLID.

Three pieces, in order, after a one-time pocket click:
  0. POCKETS  - detect_pocket.py shows the first frame and takes six clicks, one
                per pocket, then saves them to pockets.json. The camera is fixed
                so this is paid once: later runs load the file and go straight
                into the video. Clicking beats finding them automatically - a
                pocket is a dark hole, and so is every shadow under a rail.
  1. DETECT   - server.py's felt mode finds where the balls are, and rejects
                the player: an arm, the hand on it and the cue it holds are one
                connected non-felt mass, and a mass too big, too long or
                reaching in over the table edge is erased WHOLE - every
                fingertip that looked exactly like a ball going with it,
                because one at a time they are indistinguishable from balls.
                What is left over is handed to logic.py as a fact about the
                frame rather than thrown away: a hand is on the table, so this
                frame's ball count is not worth counting. The camera is
                fixed, so the table outline and the ball radius are measured
                once and reused, which is most of the speed-up over per-frame
                work. Every ball gets the SAME radius: at a fixed height they
                really are the same size, so a per-ball radius guess is noise,
                and a radius that is off drags the colour sample onto the cloth.
  2. TRACK    - a nearest-neighbour tracker gives each ball a stable id across
                frames, which is what lets the verdict be voted over time
                rather than recomputed from scratch every frame.
  3. IDENTIFY - ball_class.py counts the white pixels inside each ball, and
                track_identity.py votes that over recent frames. One frame is
                not enough: stripes and solids meet around 20% white, so a
                single reading flickers. Over several frames a rolling stripe
                shows its band and a solid never does.
  4. SCORE    - logic.py turns "a ball vanished in a pocket" into eight-ball:
                groups, turns, fouls and the win. A ball that vanishes anywhere
                else was occluded, which is the one distinction the clicked
                pockets buy.

What you see: a scoreboard strip above the video, then a ring per ball coloured
by class with the class written beside it. Ring THICKNESS is confidence - thick
once the vote has settled, thin while it is still being taken.

Usage:
    python main.py                        # game.mp4 -> result.mp4, with labels
    python main.py clip.mp4               # a different video
    python main.py --show                 # live preview window (q or ESC quits)
    python main.py --stride 5             # process every 5th frame (fast pass)
    python main.py --frames 300           # stop after 300 processed frames
    python main.py --csv balls.csv        # log frame,track,class,...
    python main.py --no-id                # positions only, skip identification
    python main.py --repick-pockets       # click the six pockets again
    python main.py --no-pockets           # run without them (no scoring)
    python main.py --player-a Ann --player-b Bo

Press 'q' or ESC to stop early - the video written so far is still playable.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

import detect_pocket
import logic
from server import (FELT_HIGH, FELT_LOW, drop_intrusions, estimate_ball_radius,
                    felt_mask, find_balls, not_felt_mask, sample_felt,
                    table_region)
from track_identity import UNKNOWN_BGR, BallClassRegistry

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

    def __init__(self, max_dist=140, max_misses=6):
        self.max_dist = max_dist
        self.max_misses = max_misses
        self._next_id = 0
        self.tracks = {}                   # id -> [x, y, r, misses, vx, vy]

    def update(self, balls):
        """Take this frame's (x,y,r) list, return [(track_id, x, y, r), ...]."""
        unmatched = dict(self.tracks)
        out = []

        # Greedy nearest match against each track's PREDICTED position. A
        # rolling ball is matched where it is heading, not where it was, which
        # is what keeps a track id alive through a shot at this frame rate.
        pairs = []
        for di, (x, y, r) in enumerate(balls):
            for tid, (tx, ty, _tr, _m, vx, vy) in unmatched.items():
                d = min((x - (tx + vx)) ** 2 + (y - (ty + vy)) ** 2,
                        (x - tx) ** 2 + (y - ty) ** 2)
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
                vx = vy = 0.0
            else:
                px, py = self.tracks[tid][0], self.tracks[tid][1]
                vx, vy = x - px, y - py
            self.tracks[tid] = [x, y, r, 0, vx, vy]
            out.append((tid, x, y, r))

        # Age out tracks that went unseen; a ball behind a player's arm comes
        # back a frame or two later and should keep its number.
        for tid in list(self.tracks):
            if tid not in taken_trk and tid not in {t for t, *_ in out}:
                self.tracks[tid][3] += 1
                if self.tracks[tid][3] > self.max_misses:
                    del self.tracks[tid]
        return out


#: Radius of the hole punched in the table mask at each clicked pocket, in ball
#: radii. A pocket is a dark hole, so it is not felt, so a ball-sized disc fits
#: it perfectly - a pocket that falls inside the table mask reads as a permanent
#: ball which blinks in and out with the threshold, and every blink looks
#: exactly like a pot. Cutting the six known pockets out removes that whole
#: class of phantom. Kept just wide enough to cover the mouth: a ball hanging in
#: the jaws inside the blind spot would read as potted, so the blind spot should
#: not reach out onto the felt where a ball can come to rest.
POCKET_BLOCK = 1.6


def calibrate(frame, felt_low, felt_high, shrink, ball_r=None, pockets=None,
              block=POCKET_BLOCK):
    """Measure the table outline and the ball radius once.

    A fixed overhead camera means neither changes between frames, and
    table_region() is the most expensive step in the pipeline - doing it per
    frame would waste most of the runtime. The felt colour itself is not
    sampled: the bounds come from tuner.py (server.FELT_LOW/HIGH), which
    separates the blue balls from the cloth far better than a tolerance band
    around the median could.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]
    table = table_region(felt_mask(hsv, felt_low, felt_high), H, W, shrink)
    not_felt = not_felt_mask(frame, table, felt_low, felt_high)
    measured = sample_felt(hsv, H, W)        # reported only, not used to match
    r = ball_r if ball_r else estimate_ball_radius(not_felt)

    # After the radius is measured, not before: the hole is sized in ball radii,
    # and the pocket blobs are the wrong shape to pass the radius estimator's
    # squareness test anyway, so they do not bias it.
    for p in pockets or []:
        cv2.circle(table, (p.x, p.y), int(round(block * r)), 0, -1)

    return {"low": felt_low, "high": felt_high, "table": table,
            "felt": measured, "r": r}


def detect_frame(frame, cal):
    """Per-frame half of the pipeline, using the cached calibration.

    Returns (balls, intrusions). The intrusions - arms, hands, cues, erased
    from the mask whole before any ball is looked for - are handed back rather
    than discarded because their PRESENCE is worth more than their removal: a
    frame with a hand in it is a frame whose ball count means nothing, and the
    game layer would rather wait for the next one than count through it.
    """
    not_felt = not_felt_mask(frame, cal["table"], cal["low"], cal["high"])
    clean, intrusions = drop_intrusions(not_felt, cal["r"], cal["table"])
    balls = find_balls(clean, cal["r"], drop=False)
    balls.sort(key=lambda b: (b[1], b[0]))
    return balls, intrusions


#: Ring THICKNESS per confidence: a thicker ring is a stronger claim. Ring
#: COLOUR is the class, so the overlay says what the ball is and how sure we
#: are in two separate channels instead of overloading one.
CONF_WEIGHT = {
    "voted":   3,      # the vote has settled
    "pending": 1,      # too few frames seen to trust it yet
}


#: Directions a label may be pushed, straight up first then fanning out to
#: either side in step, so a crowded cluster opens symmetrically.
_FAN = [(math.sin(a), -math.cos(a)) for a in
        [i * math.pi / 6 for i in (0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6)]]


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_overlay(img, items, scale=1.0):
    """Draw every ball's ring, then place the labels so they do not collide.

    Rings go down first so no label chip can bury one.

    Placement matters more than it sounds: racked, fifteen balls sit inside a
    triangle barely wider than a single label, so the obvious "just above the
    ball" position buries all fifteen under each other - which is exactly what
    the first version of this did. Each label is instead tried at a ring of
    offsets working outward and takes the first that is clear, with a leader
    line back to its ball once it has moved far enough to be ambiguous.
    """
    H, W = img.shape[:2]
    thick = max(1, int(round(scale * 1.6)))
    pad = max(2, int(scale * 4))

    for _tid, x, y, r, label in items:
        bgr = UNKNOWN_BGR if label is None else tuple(label.bgr)
        weight = 1 if label is None else CONF_WEIGHT.get(label.confidence, 1)
        cv2.circle(img, (x, y), r, bgr, max(1, int(round(weight * scale))))

    # Biggest ball first: the labels most likely to be read get the best spots.
    taken = []
    for _tid, x, y, r, label in sorted(items, key=lambda it: -it[3]):
        bgr = UNKNOWN_BGR if label is None else tuple(label.bgr)
        text = "?" if label is None else label.text
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         scale, thick)
        bw, bh = tw + 2 * pad, th + base + 2 * pad

        spot = None
        for step in (1.0, 1.9, 2.9, 4.1, 5.5):
            for dx, dy in _FAN:
                cx = x + dx * (r + step * bh * 0.8)
                cy = y + dy * (r + step * bh * 0.8)
                box = (int(cx - bw / 2), int(cy - bh / 2),
                       int(cx + bw / 2), int(cy + bh / 2))
                if box[0] < 2 or box[1] < 2 or box[2] > W - 2 or box[3] > H - 2:
                    continue
                if any(_overlaps(box, t) for t in taken):
                    continue
                spot = box
                break
            if spot is not None:
                break
        if spot is None:                  # nowhere clear; stack it above anyway
            spot = (int(x - bw / 2), int(y - r - bh), int(x + bw / 2), int(y - r))
        taken.append(spot)

        # A leader line once the label has been pushed clear of its own ball,
        # so a fanned-out cluster stays readable as fifteen labelled balls.
        mx, my = (spot[0] + spot[2]) // 2, (spot[1] + spot[3]) // 2
        if (mx - x) ** 2 + (my - y) ** 2 > (r + bh) ** 2:
            cv2.line(img, (x, y), (mx, my), bgr, max(1, int(scale)), cv2.LINE_AA)

        # A filled chip behind the text: outlined text alone disappears into the
        # cloth, and the chip is what makes a label readable over a ball.
        cv2.rectangle(img, (spot[0], spot[1]), (spot[2], spot[3]), (0, 0, 0), -1)
        cv2.rectangle(img, (spot[0], spot[1]), (spot[2], spot[3]), bgr,
                      max(1, int(scale)))
        # The ring carries the true colour; the text is that hue washed towards
        # white, because "BLACK 8" and "MAROON SOLID" are unreadable in their own.
        tint = tuple(min(255, int(c * 0.45 + 140)) for c in bgr)
        cv2.putText(img, text, (spot[0] + pad, spot[3] - pad - base),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, tint, thick, cv2.LINE_AA)


#: Scoreboard palette, BGR. A near-black board with a single green accent: the
#: board has to read as chrome, not compete with the table under it.
BOARD_BG = (16, 22, 18)
BOARD_GREEN = (128, 222, 74)
BOARD_WHITE = (245, 245, 245)
BOARD_GREY = (150, 158, 152)
BOARD_DIM = (96, 104, 98)
BOARD_RED = (90, 90, 235)

#: Board height as a fraction of the frame. The board is stacked ABOVE the
#: video rather than drawn over it - the top rail and its three pockets are the
#: part of the table an overlaid bar would cover.
BOARD_FRACTION = 0.16

#: Separator in the stat line. OpenCV 5's built-in font has the middle dot; on
#: an older build that draws as a box, so it is one constant to change.
DOT = " · "


def _board_text(img, text, x, y, size, colour, thick=1, anchor="left"):
    """Put text with y as its vertical CENTRE and x as the chosen anchor.

    Every row of the board is centred against a design line, so measuring the
    text and placing it from the middle beats tracking baselines by hand.
    """
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    if anchor == "center":
        x -= tw / 2.0
    elif anchor == "right":
        x -= tw
    cv2.putText(img, text, (int(x), int(y + th / 2.0)),
                cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)
    return tw


def scoreboard(W, height, session, index, total, fps_now):
    """The strip above the video: who is on what, and what just happened.

    The active player is marked twice - an accent bar on their side and their
    name in the accent colour - because on a glance at a still frame the score
    alone does not say whose turn it is, and that is the first thing a viewer
    of a pool match wants to know.
    """
    game = session.game
    board = np.full((height, W, 3), BOARD_BG, np.uint8)
    s = height / 150.0                       # the layout is designed at 150 px
    edge = int(46 * s)

    # Accent bar on the shooter's side.
    if not game.over:
        bx = edge - int(18 * s) if game.turn == 0 else W - edge + int(14 * s)
        cv2.rectangle(board, (bx, int(0.18 * height)),
                      (bx + max(2, int(4 * s)), int(0.82 * height)),
                      BOARD_GREEN, -1)

    _board_text(board, "8-BALL", W / 2, 0.22 * height, 0.52 * s,
                BOARD_GREEN, 1, "center")

    for i, anchor, nx, sx in ((0, "left", edge, W * 0.385),
                              (1, "right", W - edge, W * 0.615)):
        live = (not game.over) and game.turn == i
        _board_text(board, game.players[i], nx, 0.46 * height, 0.66 * s,
                    BOARD_GREEN if live else BOARD_WHITE,
                    max(1, int(round(2 * s))), anchor)
        sub = game.group_name(i)
        if game.wins[i]:
            sub += DOT + f"{game.wins[i]} RACK" + ("S" if game.wins[i] > 1 else "")
        if game.fouls[i]:
            sub += DOT + f"{game.fouls[i]} FOUL" + ("S" if game.fouls[i] > 1 else "")
        _board_text(board, sub, nx, 0.64 * height, 0.42 * s, BOARD_DIM, 1, anchor)
        _board_text(board, str(game.score(i)), sx, 0.46 * height, 1.55 * s,
                    BOARD_WHITE, max(1, int(round(2 * s))), "center")

    _board_text(board, f"{game.score(0)} - {game.score(1)}", W / 2,
                0.46 * height, 0.80 * s, BOARD_GREY, max(1, int(round(1.5 * s))),
                "center")
    _board_text(board,
                DOT.join([f"RACK {game.rack}", f"SHOT {game.shot}",
                          f"INNING {game.inning}", session.clock,
                          f"BALLS {session.balls}"]),
                W / 2, 0.66 * height, 0.42 * s, BOARD_DIM, 1, "center")

    # The status line is the board's only moving part, so it carries the colour:
    # green for a result, red for a foul, grey for the run of play.
    status = session.status
    colour = (BOARD_GREEN if game.over else
              BOARD_RED if status.startswith("FOUL") else BOARD_GREY)
    _board_text(board, status, W / 2, 0.82 * height, 0.52 * s, colour,
                max(1, int(round(1.2 * s))), "center")

    foot = f"FRAME {index}/{total}{DOT}{fps_now:.1f} FPS"
    # Say when the count being shown is a held one. BALLS on the stat line is
    # read as fact, and during an intrusion it is not one - the arm has taken
    # whatever it was lying across out of the mask with it.
    if session.busy:
        foot += DOT + "HAND ON THE TABLE"
    _board_text(board, foot, edge, 0.93 * height, 0.38 * s,
                BOARD_RED if session.busy else BOARD_DIM, 1, "left")
    if game.ball_in_hand and not game.over:
        _board_text(board, "BALL IN HAND", W - edge, 0.93 * height, 0.38 * s,
                    BOARD_RED, 1, "right")

    # A hairline where the board meets the video, so the two read as separate
    # panels rather than as an overlay that failed to cover the top rail.
    board[-max(1, int(round(2 * s))):, :] = BOARD_GREEN
    return board


def draw_win(frame, session, scale):
    """Celebrate the rack over the video: a band, a name, and some movement.

    Movement because a still caption is easy to miss on a fast-moving overlay -
    the text grows in, breathes while it holds, and fades rather than vanishing.
    """
    name, progress = session.celebration
    game = session.game
    H, W = frame.shape[:2]
    grow = min(1.0, progress / 0.12) ** 0.5           # eased entry
    fade = 1.0 if progress < 0.82 else max(0.0, (1.0 - progress) / 0.18)
    pulse = 1.0 + 0.025 * math.sin(progress * 26.0)
    size = scale * 2.4 * (0.55 + 0.45 * grow) * pulse

    band = int(H * 0.20)
    top = (H - band) // 2
    shade = frame.copy()
    cv2.rectangle(shade, (0, top), (W, top + band), (8, 12, 9), -1)
    cv2.addWeighted(shade, 0.62 * fade, frame, 1.0 - 0.62 * fade, 0, frame)
    line = max(1, int(round(2 * scale)))
    cv2.line(frame, (0, top), (W, top), BOARD_GREEN, line)
    cv2.line(frame, (0, top + band), (W, top + band), BOARD_GREEN, line)

    text = f"{name.upper()} WINS"
    thick = max(2, int(round(3.5 * scale)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    # Faded text means blending toward the band, not toward black.
    tint = tuple(int(20 + (c - 20) * fade) for c in BOARD_GREEN)
    cv2.putText(frame, text, ((W - tw) // 2, int(top + band * 0.46 + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, size, tint, thick, cv2.LINE_AA)

    # What the rack finished at, so the celebration is also the final score -
    # the board behind it resets as soon as the next triangle is set.
    sub = DOT.join([f"RACK {game.rack}",
                    f"{game.score(0)} - {game.score(1)}",
                    game.status.split(" - ")[-1]])
    _board_text(frame, sub, W / 2, top + band * 0.82, 0.66 * scale,
                tuple(int(20 + (c - 20) * fade) for c in BOARD_GREY),
                max(1, int(round(1.3 * scale))), "center")
    return frame


#: Intrusions are drawn in the board's warning red, the same hue a foul uses,
#: because they mean the same thing to a viewer: what you are looking at is not
#: being counted.
INTRUSION_BGR = BOARD_RED


def draw_intrusions(img, intrusions, scale=1.0):
    """Outline what was erased from the mask, and say why.

    Worth the pixels because this is the one part of the pipeline that throws
    away things that look exactly like balls. Drawn, it is obvious that the arm
    went and the balls stayed; undrawn, a run where the detector quietly ate a
    corner of the table looks identical to a run where it worked.
    """
    for it in intrusions or []:
        x, y, w, h = it.box
        thick = max(1, int(round(scale * (2 if it.bulky else 1))))
        cv2.rectangle(img, (x, y), (x + w, y + h), INTRUSION_BGR, thick)
        if not it.bulky:
            continue          # a cue shaft or a hairline of glare: no caption
        text = f"NOT BALLS - {it.why}".upper()
        cv2.putText(img, text, (x, max(int(14 * scale), y - int(8 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, max(int(14 * scale), y - int(8 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, INTRUSION_BGR,
                    thick, cv2.LINE_AA)


def draw(frame, items, index, total, fps_now, scale=1.0, pockets=None,
         session=None, board_h=0, intrusions=None):
    """items: [(track_id, x, y, r, Label_or_None), ...]"""
    # Pockets first: they are fixed scenery, and a ball's ring must sit on top
    # of a pocket marker when the two overlap, not under it.
    detect_pocket.draw_pockets(frame, pockets, scale)
    draw_intrusions(frame, intrusions, scale)
    draw_overlay(frame, items, scale)
    if session is not None and session.celebration:
        draw_win(frame, session, scale)
    if session is None or board_h <= 0:
        return frame
    return np.vstack([
        scoreboard(frame.shape[1], board_h, session, index, total, fps_now),
        frame])


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
    ap.add_argument("--ball-r", type=int, default=None,
                    help="ball radius in px; measured from the first frame "
                         "when not given")
    ap.add_argument("--felt-low", nargs=3, type=int, default=list(FELT_LOW),
                    help=f"felt HSV lower bound, from tuner.py (default {FELT_LOW})")
    ap.add_argument("--felt-high", nargs=3, type=int, default=list(FELT_HIGH),
                    help=f"felt HSV upper bound, from tuner.py (default {FELT_HIGH})")
    ap.add_argument("--recalibrate", type=int, default=0,
                    help="re-measure felt/table every N frames (0 = once, the default)")
    ap.add_argument("--no-id", action="store_true",
                    help="skip ball identification, draw positions only")
    ap.add_argument("--pockets", default=detect_pocket.DEFAULT_POCKET_FILE,
                    help="pocket calibration file; clicked on the first frame "
                         f"when missing (default: {detect_pocket.DEFAULT_POCKET_FILE})")
    ap.add_argument("--repick-pockets", action="store_true",
                    help="click the six pockets again, replacing the saved ones")
    ap.add_argument("--no-pockets", action="store_true",
                    help="skip pocket calibration entirely")
    ap.add_argument("--player-a", default="Player A", help="left-hand name")
    ap.add_argument("--player-b", default="Player B", help="right-hand name")
    ap.add_argument("--pot-radius", type=float, default=logic.POCKET_MOUTH,
                    help="how close to a pocket a ball's last sighting must be "
                         f"to count as potted, in ball radii "
                         f"(default {logic.POCKET_MOUTH})")
    ap.add_argument("--pocket-block", type=float, default=POCKET_BLOCK,
                    help="radius of the blind spot cut out of the table mask at "
                         f"each pocket, in ball radii (default {POCKET_BLOCK}); "
                         "0 puts the pocket mouths back in as ball candidates")
    ap.add_argument("--no-hand-gate", action="store_true",
                    help="keep erasing arms and cues from the mask, but do not "
                         "let their presence hold back the count - judge every "
                         "settling the moment the balls stop, hand or no hand")
    ap.add_argument("--events", default=None,
                    help="write the identity, pot and game logs to this JSON file")
    ap.add_argument("--label-scale", type=float, default=1.0,
                    help="multiplier on the overlay text size; the base size "
                         "already tracks the frame height (default 1.0)")
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

    # Pockets before anything else, so the ten-second click happens while
    # nothing is half-written: a cancel here leaves no output file behind.
    pockets = None
    if not args.no_pockets:
        pockets_path = (args.pockets if os.path.isabs(args.pockets)
                        else os.path.join(here, args.pockets))
        pockets = detect_pocket.calibrate(frame, pockets_path, video=in_path,
                                          force=args.repick_pockets)
        if pockets is None:
            sys.exit("Pocket selection cancelled - nothing was processed.\n"
                     "Run again, or use --no-pockets to skip it.")

    cal = calibrate(frame, tuple(args.felt_low), tuple(args.felt_high), shrink=18,
                    ball_r=args.ball_r, pockets=pockets, block=args.pocket_block)
    print(f"Felt:   window {tuple(args.felt_low)}..{tuple(args.felt_high)}  "
          f"(measured median HSV {cal['felt'].astype(int).tolist()})  "
          f"(table mask covers {100 * (cal['table'] > 0).mean():.1f}% of frame)")
    print(f"Ball:   radius {cal['r']} px"
          f"{' (given)' if args.ball_r else ' (measured)'}")

    # Overlay text is sized from the frame height, not fixed in pixels: a label
    # that reads well on 1080p is invisible on 4K and unreadable once a 1080p
    # video is played back in a half-width window.
    label_scale = max(0.35, (H / 1080.0) * 0.95 * args.label_scale)

    # The scoreboard is a strip stacked on top of the video, so the written
    # frames are taller than the source ones.
    board_h = max(96, int(round(H * BOARD_FRACTION)))

    # Frames arrive at src_fps / stride, and the game clock counts video time,
    # not processing time.
    play_fps = max(1.0, src_fps / args.stride)
    session = logic.GameSession(pockets, cal["r"], fps=src_fps,
                                players=(args.player_a, args.player_b),
                                mouth=args.pot_radius)

    # Writing at source fps / stride keeps the output playing at real speed.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             play_fps, (W, H + board_h))
    if not writer.isOpened():
        sys.exit(f"Could not open the video writer for: {out_path}")

    csv_file = csv_writer = None
    if args.csv:
        csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track", "class", "confidence", "x", "y", "r"])

    tracker = Tracker()
    registry = None if args.no_id else BallClassRegistry()

    idx = args.start
    processed = 0
    logged = 0
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
                cal = calibrate(frame, tuple(args.felt_low),
                                tuple(args.felt_high), shrink=18,
                                ball_r=args.ball_r, pockets=pockets,
                                block=args.pocket_block)

            balls, intrusions = detect_frame(frame, cal)
            counts.append(len(balls))
            tracked = tracker.update(balls)
            # Only the BULKY ones stop the clock: a region big enough to have
            # hidden a ball or grown a phantom. A cue shaft lying on the cloth
            # is erased too, but it is too thin to have covered anything, and
            # treating it as a reason to distrust the frame would stall the
            # scoreboard through every shot that was still being lined up.
            hand = (any(it.bulky for it in intrusions)
                    and not args.no_hand_gate)

            items = [(tid, x, y, r, None) for tid, x, y, r in tracked]
            labels = {}
            if registry is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                labels = registry.update(tracked, hsv, frame=idx)
                items = [(tid, x, y, r, labels.get(tid))
                         for tid, x, y, r in tracked]
            id_counts.append(sum(1 for it in items if it[4] is not None))

            for pot in session.update(tracked, labels, idx, intruding=hand):
                print(f"  POT   frame {idx}  {pot.cls.upper()} in {pot.pocket} "
                      f"(track {pot.track}, {pot.dist:.0f} px out, "
                      f"closing {pot.closed:.0f} px/frame)", flush=True)
            # Anything the rules decided this frame - a foul, a group, a win.
            while logged < len(session.game.log):
                print(f"  GAME  frame {idx}  {session.game.log[logged]['text']}",
                      flush=True)
                logged += 1
            if csv_writer:
                for tid, x, y, r, label in items:
                    csv_writer.writerow([
                        idx, tid,
                        "" if label is None else label.cls,
                        "" if label is None else label.confidence,
                        x, y, r])

            processed += 1
            elapsed = time.perf_counter() - t0
            fps_now = processed / elapsed if elapsed else 0.0

            out = draw(frame, items, idx, total, fps_now, label_scale,
                       pockets, session, board_h, intrusions)
            writer.write(out)

            if args.show:
                preview = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2))
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
        if id_counts and registry is not None:
            print(f"Labelled per frame: min {min(id_counts)}  max {max(id_counts)}  "
                  f"mean {sum(id_counts) / len(id_counts):.1f}")
            summary = registry.summary()
            print(f"Live tracks at the end: {summary['tracked']}  "
                  f"{summary['by_class']}")
            print(f"  uniqueness demotions: {len(registry.events)}")

        game = session.game
        print(f"\nGame:   {game.players[0]} {game.score(0)} - {game.score(1)} "
              f"{game.players[1]}   "
              f"({game.group_name(0)} vs {game.group_name(1)})")
        print(f"        {game.shot} shots, {game.inning} innings, "
              f"{len(session.all_pots)} pots, "
              f"{game.fouls[0]}+{game.fouls[1]} fouls")
        print(f"        {len(session.episodes)} episodes of motion, "
              f"{len(session.pots.rejected)} claims refused")
        waited = [e for e in session.episodes if e.get("held")]
        print(f"        hands/cues on the table in {session.busy_frames} of "
              f"{processed} frames "
              f"({100.0 * session.busy_frames / max(1, processed):.0f}%), "
              f"{len(waited)} settlings held for a clear table"
              + (f", {len(session.waits)} judged without one"
                 if session.waits else ""))
        # The self-check: a ball left the table and nothing was credited for it.
        # Each of these is a pot that went unscored, so print where to look.
        if session.unexplained:
            print(f"        UNEXPLAINED: {len(session.unexplained)} settlings "
                  f"lost a ball with nothing credited, at frames "
                  f"{[e['frame'] for e in session.unexplained][:12]}")
        print(f"        {game.status}")

    if args.events:
        ev_path = (args.events if os.path.isabs(args.events)
                   else os.path.join(here, args.events))
        with open(ev_path, "w") as fh:
            json.dump({"identity": [] if registry is None else registry.events,
                       "pots": [p._asdict() for p in session.all_pots],
                       "episodes": session.episodes,
                       "unexplained": session.unexplained,
                       "rejected": {
                           "claims": [p._asdict() for p in session.ignored],
                           "why": session.pots.rejected},
                       "intrusions": {"busy_frames": session.busy_frames,
                                      "frames": processed,
                                      "gave_up_waiting": session.waits},
                       "game": session.game.log,
                       "summary": session.game.summary()}, fh, indent=2)
        print(f"Saved:  {ev_path}")
    print(f"Saved:  {out_path}")
    if args.csv:
        print(f"Saved:  {args.csv}")


if __name__ == "__main__":
    main()
