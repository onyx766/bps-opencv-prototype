"""
BPS prototype - find the balls, score the game, and put the client's HUD on it.

Five pieces, in order, after a one-time pocket click:
  0. POCKETS  - detect_pocket.py shows the first frame and takes six clicks, one
                per pocket, then saves them to pockets.json.
  1. DETECT   - server.py's felt mode finds where the balls are, and rejects
                the player: an arm, the hand on it and the cue it holds are one
                connected non-felt mass, erased WHOLE.
  2. TRACK    - a nearest-neighbour tracker gives each ball a stable id.
  3. IDENTIFY - ball_class.py and track_identity.py vote CUE / 8 / STRIPE /
                SOLID over recent frames.
  4. SCORE    - logic.py turns what was seen into evidence, and games.py turns
                the evidence into the client's locked rules. rules.py cites every
                row of BPS_Scoring_TestSuite_v1.3.csv and says where it lives.
  5. HUD      - this file. Detection, tracking and identification above are
                unchanged; everything the client's suite asks a player to SEE or
                TAP is drawn and handled here.

THE HUD, AND WHY IT LOOKS THE WAY IT DOES

    scoreboard    Names by table side (M-15), Skill Level only where it applies
                  (H-03, H-04), groups or points, a foul COUNTER and never a
                  deduction (F-01), time-outs left (H-05), the breaker-anchored
                  inning (H-01). Calls under the confidence bar carry an amber
                  dot (M-14). SCORE UNCONFIRMED is advisory and never blocks
                  (M-03). Occlusion is a small grey dot, not a banner (M-05).
    button bar    Exactly the taps that are legal right now, straight from
                  GameSession.commands(): MARK DEFENCE only between shots
                  (H-02), END TIME OUT only while one runs (H-05), UNDO gone once
                  the score is final (M-03), CALL FOUL greyed once the next
                  stroke begins (F-10).
    prompts       The questions the camera must not answer alone: confirm
                  groups with two thumbnails (M-06), a wedged pair (W-11), the
                  first rack on a new table (M-09), a league illegal break
                  (M-10), a ball to replace (M-02).
    on the table  The head-string zone for a restricted ball in hand (F-02),
                  where a moved ball goes back (M-07), frozen balls (F-08), the
                  time-out clock (H-05), W-02's undo window.
    review        The last ten seconds, side by side with camera 2 (M-13). BPS
                  shows the evidence; the players decide, with UNDO.

Usage:
    python main.py                        # game.mp4 -> result.mp4
    python main.py --show                 # live, with a clickable HUD
    python main.py --game 9ball           # nine-ball rules
    python main.py --mode league --sl1 4 --sl2 6
    python main.py --mode practice        # analytics only (M-11)
    python main.py --taps "120=timeout,420=end_timeout,600=review"
    python main.py --stride 5 --frames 300
    python main.py --repick-pockets | --no-pockets | --no-id

Keys in --show:  d defence  t time out  e end time out  x cancel time out
                 u undo  k confirm final  o reopen  f call foul  s stalemate
                 r review  l call loss (unmarked 8)  z declare frozen
                 g swap groups  p push-out (Masters)
                 1/2/3 answer a prompt  space pause  q/ESC quit

League manual options:
    python main.py --lag-winner 2         # who won the lag (breaks, top inning)
    python main.py --format masters       # race to 7, no time-outs, push-outs
    python main.py --break-assigns off    # the CSV's W-08: break never assigns
    python main.py --innings breaker      # the CSV's H-01 inning count
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import detect_pocket
import games
import logic
import rules
import skill_level
from server import (FELT_HIGH, FELT_LOW, drop_intrusions, estimate_ball_radius,
                    felt_mask, find_balls, not_felt_mask, outer_rim,
                    sample_felt, table_region)
from track_identity import UNKNOWN_BGR, BallClassRegistry

# Tried in order when no video argument is given - first one that exists wins.
DEFAULT_VIDEOS = ["game.mp4", "test.mp4", "input.mp4"]
DEFAULT_OUTPUT = "result.mp4"

#: M-09 needs to know whether a table has ever had its first rack confirmed.
#: That is a fact about the TABLE, not about a run, so it outlives the process.
TABLE_MEMORY = "bps_table.json"

WINDOW = "BPS - HUD"
#: The live preview is shown at half size; clicks are mapped back through this.
PREVIEW_SCALE = 0.5


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
#: it perfectly - a pocket inside the table mask reads as a permanent ball that
#: blinks with the threshold, and every blink looks exactly like a pot.
POCKET_BLOCK = 1.6

#: How far round each pocket, in ball radii, the table's outer edge does NOT
#: count as a hand coming in over the rail (server.outer_rim). A hand resting on
#: the left rail at 4:17 in this footage read as two balls; its fingers touched
#: the edge in every frame, while every ball resting on a cushion stayed 0.65
#: radii or more clear of it. The exception is a pocket's jaw: the red solid
#: going into the bottom-left pocket touched the edge 3.1 radii from the pocket.
#: The pocket mouth used for pot claims covers that with room to spare.
RIM_POCKET_CLEAR = logic.POCKET_MOUTH


def calibrate(frame, felt_low, felt_high, shrink, ball_r=None, pockets=None,
              block=POCKET_BLOCK):
    """Measure the table outline and the ball radius once.

    A fixed overhead camera means neither changes between frames, and
    table_region() is the most expensive step in the pipeline.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]
    table = table_region(felt_mask(hsv, felt_low, felt_high), H, W, shrink)
    not_felt = not_felt_mask(frame, table, felt_low, felt_high)
    measured = sample_felt(hsv, H, W)        # reported only, not used to match
    r = ball_r if ball_r else estimate_ball_radius(not_felt)

    # Before the pocket holes are cut: their edges are where balls go down.
    # Without pockets nothing protects a ball in a jaw, so there is no rim test.
    rim = outer_rim(table, pockets, RIM_POCKET_CLEAR * r) if pockets else None
    for p in pockets or []:
        cv2.circle(table, (p.x, p.y), int(round(block * r)), 0, -1)

    return {"low": felt_low, "high": felt_high, "table": table,
            "felt": measured, "r": r, "rim": rim}


def detect_frame(frame, cal):
    """Per-frame half of the pipeline, using the cached calibration.

    Returns (balls, intrusions). The intrusions are handed back rather than
    discarded because their PRESENCE is worth more than their removal: a frame
    with a hand in it is a frame whose ball count means nothing.
    """
    not_felt = not_felt_mask(frame, cal["table"], cal["low"], cal["high"])
    clean, intrusions = drop_intrusions(not_felt, cal["r"], cal["table"],
                                        rim=cal.get("rim"))
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

    Racked, fifteen balls sit inside a triangle barely wider than a single
    label, so each label is tried at a ring of offsets working outward and takes
    the first that is clear, with a leader line back to its ball.
    """
    H, W = img.shape[:2]
    thick = max(1, int(round(scale * 1.6)))
    pad = max(2, int(scale * 4))

    for _tid, x, y, r, label in items:
        bgr = UNKNOWN_BGR if label is None else tuple(label.bgr)
        weight = 1 if label is None else CONF_WEIGHT.get(label.confidence, 1)
        cv2.circle(img, (x, y), r, bgr, max(1, int(round(weight * scale))))

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
        if spot is None:
            spot = (int(x - bw / 2), int(y - r - bh), int(x + bw / 2), int(y - r))
        taken.append(spot)

        mx, my = (spot[0] + spot[2]) // 2, (spot[1] + spot[3]) // 2
        if (mx - x) ** 2 + (my - y) ** 2 > (r + bh) ** 2:
            cv2.line(img, (x, y), (mx, my), bgr, max(1, int(scale)), cv2.LINE_AA)

        cv2.rectangle(img, (spot[0], spot[1]), (spot[2], spot[3]), (0, 0, 0), -1)
        cv2.rectangle(img, (spot[0], spot[1]), (spot[2], spot[3]), bgr,
                      max(1, int(scale)))
        tint = tuple(min(255, int(c * 0.45 + 140)) for c in bgr)
        cv2.putText(img, text, (spot[0] + pad, spot[3] - pad - base),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, tint, thick, cv2.LINE_AA)


# ---- palette ----------------------------------------------------------------

#: Scoreboard palette, BGR. A near-black board with a single green accent: the
#: board has to read as chrome, not compete with the table under it. Amber is
#: for "look at this, nothing is wrong yet" - review flags, provisional calls,
#: an unconfirmed score - and red is kept for fouls, so the two never blur.
BOARD_BG = (16, 22, 18)
BOARD_GREEN = (128, 222, 74)
BOARD_WHITE = (245, 245, 245)
BOARD_GREY = (150, 158, 152)
BOARD_DIM = (96, 104, 98)
BOARD_RED = (90, 90, 235)
BOARD_AMBER = (60, 190, 245)
BAR_BG = (22, 30, 25)
BUTTON_BG = (46, 56, 49)
BUTTON_OFF = (30, 38, 33)
FROZEN_BGR = (230, 210, 120)

#: Board height as a fraction of the frame. Stacked ABOVE the video rather than
#: drawn over it - the top rail and its three pockets are what an overlaid bar
#: would cover.
BOARD_FRACTION = 0.16

#: The button strip under the board. Big enough to hit with a thumb on a
#: tablet-sized preview, which is the smallest screen this HUD is aimed at.
BAR_FRACTION = 0.05

#: Separator in the stat line.
DOT = " · "

#: Every player-facing string drawn this run that broke H-04. Collected rather
#: than raised, so a bad string shows up in the run summary instead of crashing
#: a render halfway through a match.
COPY_OFFENDERS = set()


def _board_text(img, text, x, y, size, colour, thick=1, anchor="left"):
    """Put text with y as its vertical CENTRE and x as the chosen anchor.

    Every HUD string passes through here, which makes this the one place H-04
    can be checked on what players actually see.
    """
    if not skill_level.clean(text):
        COPY_OFFENDERS.add(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    if anchor == "center":
        x -= tw / 2.0
    elif anchor == "right":
        x -= tw
    cv2.putText(img, text, (int(x), int(y + th / 2.0)),
                cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)
    return tw


def _fit(text, size, thick, width):
    """The largest text size, up to `size`, at which `text` fits in `width`."""
    while size > 0.2:
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
        if tw <= width:
            break
        size *= 0.9
    return size


def _shade(img, box, colour, alpha):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    block = np.full_like(roi, colour)
    cv2.addWeighted(block, alpha, roi, 1.0 - alpha, 0, roi)


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "S")


# ---- the scoreboard ---------------------------------------------------------

def scoreboard(W, height, session, index, total, fps_now):
    """The strip above the video: who is on what, and what just happened.

    The active player is marked twice - an accent bar on their side and their
    name in the accent colour - because on a glance at a still frame the score
    alone does not say whose turn it is.
    """
    game = session.game
    board = np.full((height, W, 3), BOARD_BG, np.uint8)
    s = height / 150.0
    edge = int(46 * s)
    practice = game.mode == games.PRACTICE
    nine = game.discipline == games.NINE_BALL
    race = game.race()

    if not game.over and not practice:
        bx = edge - int(18 * s) if game.turn == 0 else W - edge + int(14 * s)
        cv2.rectangle(board, (bx, int(0.18 * height)),
                      (bx + max(2, int(4 * s)), int(0.82 * height)),
                      BOARD_GREEN, -1)

    title = DOT.join(["9-BALL" if nine else "8-BALL", game.mode.upper()])
    if race:
        title += DOT + f"RACE {race[0]}-{race[1]}"
        if game.chart.provisional:
            title += " (PROVISIONAL CHART)"
    _board_text(board, title, W / 2, 0.14 * height, 0.46 * s, BOARD_GREEN, 1,
                "center")

    for i, anchor, nx, sx in ((0, "left", edge, W * 0.385),
                              (1, "right", W - edge, W * 0.615)):
        live = (not game.over) and (not practice) and game.turn == i
        name = game.players[i]
        level = game.level_text(i)          # empty unless H-03 applies
        if level:
            name = f"{name}  {level}" if i == 0 else f"{level}  {name}"
        _board_text(board, name, nx, 0.40 * height, 0.62 * s,
                    BOARD_GREEN if live else BOARD_WHITE,
                    max(1, int(round(2 * s))), anchor)

        if practice:
            a = game.analytics
            sub = DOT.join([f"TSR {a.tsr(i):.0%}", f"CRR {a.crr(i):.0%}",
                            f"SDR {a.sdr(i):.0%}"])
            big = str(a.made[i])
        else:
            parts = []
            if nine:
                parts.append(f"{game.score(i)}/{race[i]} PTS" if race
                             else f"{game.score(i)} PTS")
            else:
                group = game.group_name(i)
                if game.group[i] and not game.groups_confirmed:
                    group += "?"            # M-06: provisional until the tap
                parts.append(group)
            if game.racks_won[i]:
                parts.append(_plural(game.racks_won[i], "RACK"))
            parts.append(_plural(game.fouls[i], "FOUL"))     # F-01: a counter
            parts.append(f"T/O {game.timeouts.remaining(i)}")
            if game.defence_marking:
                parts.append(f"{game.defences[i]} DEF")
            sub = DOT.join(parts)
            big = str(game.score(i))
        _board_text(board, sub, nx, 0.58 * height, 0.40 * s, BOARD_DIM, 1, anchor)
        _board_text(board, big, sx, 0.42 * height, 1.45 * s,
                    BOARD_WHITE, max(1, int(round(2 * s))), "center")

    centre = "PRACTICE" if practice else f"{game.score(0)} - {game.score(1)}"
    _board_text(board, centre, W / 2, 0.42 * height, 0.74 * s, BOARD_GREY,
                max(1, int(round(1.5 * s))), "center")

    stats = [f"RACK {game.rack}"]
    if not practice:
        stats.append(f"INNING {game.inning}")               # H-01
    stats += [f"SHOT {game.shot}", session.clock, f"BALLS {session.balls}"]
    _board_text(board, DOT.join(stats), W / 2, 0.60 * height, 0.40 * s,
                BOARD_DIM, 1, "center")

    # The status line is the board's only moving part, so it carries the
    # colour: green for a result, red for a foul, amber for something waiting
    # on a person, grey for the run of play.
    status = session.status
    if game.over:
        colour = BOARD_GREEN
    elif status.startswith("FOUL") or status.startswith("ILLEGAL"):
        colour = BOARD_RED
    elif status.startswith(("TIME OUT", "WAITING", "REVIEW", "UNDONE")):
        colour = BOARD_AMBER
    else:
        colour = BOARD_GREY
    tw = _board_text(board, status, W / 2, 0.78 * height, 0.50 * s, colour,
                     max(1, int(round(1.2 * s))), "center")
    # M-14: a call made under the confidence bar carries a visible dot.
    conf = game.status_confidence
    if conf is not None and conf < rules.CONTACT_CONFIDENCE:
        cv2.circle(board, (int(W / 2 - tw / 2 - 14 * s), int(0.78 * height)),
                   max(3, int(5 * s)), BOARD_AMBER, -1, cv2.LINE_AA)

    foot = f"FRAME {index}/{total}{DOT}{fps_now:.1f} FPS"
    fx = edge + _board_text(board, foot, edge, 0.93 * height, 0.36 * s,
                            BOARD_DIM, 1, "left")
    # M-05: occlusion is normal play. A small grey dot and a quiet word, in the
    # same dim colour as the frame counter - never red, never a banner.
    if session.busy:
        cx = int(fx + 16 * s)
        cv2.circle(board, (cx, int(0.93 * height)), max(2, int(4 * s)),
                   BOARD_GREY, -1, cv2.LINE_AA)
        _board_text(board, "VIEW PAUSED", cx + 10 * s, 0.93 * height, 0.34 * s,
                    BOARD_DIM, 1, "left")

    flags = []
    if game.ball_in_hand and not game.over:
        flags.append(("BALL IN HAND - HEAD STRING"
                      if game.ball_in_hand == games.BEHIND_HEAD_STRING
                      else "BALL IN HAND", BOARD_RED))
    if (not nine and not practice and game.struck and game.open_table
            and not game.over):
        flags.append(("OPEN TABLE", BOARD_GREEN))                # W-08
    if game.record.unconfirmed:
        flags.append(("SCORE UNCONFIRMED", BOARD_AMBER))         # M-03
    if game.record.final:
        flags.append(("FINAL SCORE", BOARD_GREEN))
    flagged = len(game.record.flagged())
    if flagged:
        flags.append((f"{flagged} FLAGGED", BOARD_AMBER))        # F-09, M-14
    if session.frozen and not session.shots.moving:
        flags.append((f"{len(session.frozen)} FROZEN", FROZEN_BGR))  # F-08
    x = W - edge
    for text, col in reversed(flags):
        x -= _board_text(board, text, x, 0.93 * height, 0.36 * s, col, 1,
                         "right") + 18 * s

    board[-max(1, int(round(2 * s))):, :] = BOARD_GREEN
    return board


# ---- the button bar ---------------------------------------------------------

def button_bar(W, height, session):
    """Exactly the taps that are legal right now, and where they are.

    What is on this bar is decided by GameSession.commands(), not here - H-02,
    H-05, M-03 and F-10 are all rules about when a button EXISTS, and keeping
    that decision next to the rules is what stops the HUD offering a tap the
    rules would refuse.
    """
    bar = np.full((height, W, 3), BAR_BG, np.uint8)
    commands = session.commands()
    pad = max(4, int(height * 0.14))
    n = max(1, len(commands))
    bw = (W - pad * (n + 1)) / float(n)
    rects = []
    s = height / 54.0
    for k, (cid, label, enabled) in enumerate(commands):
        x0 = int(pad + k * (bw + pad))
        x1 = int(x0 + bw)
        y0, y1 = pad, height - pad
        # The transient buttons - defence and a running time-out - are the ones
        # the client wants noticed while they exist, so they carry the accent.
        accent = enabled and cid in ("defence", "end_timeout")
        fill = BOARD_GREEN if accent else BUTTON_BG if enabled else BUTTON_OFF
        cv2.rectangle(bar, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(bar, (x0, y0), (x1, y1),
                      BOARD_GREEN if enabled else BOARD_DIM, 1)
        colour = BOARD_BG if accent else BOARD_WHITE if enabled else BOARD_DIM
        size = _fit(label, 0.52 * s, 1, bw - 2 * pad)
        _board_text(bar, label, (x0 + x1) / 2.0, (y0 + y1) / 2.0, size, colour,
                    max(1, int(round(1.2 * s))), "center")
        if enabled:
            rects.append((x0, y0, x1, y1, "command", {"name": cid}))
    return bar, rects


# ---- prompts ----------------------------------------------------------------

OPTION_TEXT = {"stripe": "STRIPES", "solid": "SOLIDS",
               "pocketed": "DROP THEM IN", "play-on": "PLAY ON",
               "yes": "CONFIRM RACK", "done": "DONE", "re-rack": "RE-RACK"}


def group_thumbs(frame, items, size):
    """M-06's two thumbnails: one stripe and one solid, cut from the table.

    Real balls from this table under this light, not icons - the question is
    "which of THESE is yours", and a clip-art stripe does not answer it. A voted
    label is preferred so the thumbnail is itself not the shaky call.
    """
    out = {}
    for cls in (games.STRIPE, games.SOLID):
        pool = [it for it in items if it[4] is not None and it[4].cls == cls]
        pool.sort(key=lambda it: it[4].confidence != "voted")
        if not pool:
            continue
        _tid, x, y, r, _lb = pool[0]
        h = max(4, int(r * 1.6))
        crop = frame[max(0, y - h):y + h, max(0, x - h):x + h]
        if crop.size:
            out[cls] = cv2.resize(crop, (size, size),
                                  interpolation=cv2.INTER_CUBIC)
    return out


def draw_prompt(frame, session, scale, thumbs):
    """The one open question, with a button per answer."""
    p = session.prompt
    if p is None:
        return []
    H, W = frame.shape[:2]
    with_thumbs = p.kind == "groups" and bool(thumbs)
    pw = int(min(W * 0.66, max(W * 0.42, 820 * scale)))
    ph = int((250 if with_thumbs else 150) * scale)
    x0, y0 = (W - pw) // 2, H - ph - int(30 * scale)
    _shade(frame, (x0, y0, x0 + pw, y0 + ph), (10, 14, 11), 0.88)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), BOARD_GREEN,
                  max(1, int(round(2 * scale))))
    title_size = _fit(p.text, 0.74 * scale, 2, pw - 60 * scale)
    _board_text(frame, p.text, x0 + pw / 2.0, y0 + 30 * scale, title_size,
                BOARD_WHITE, max(1, int(round(1.6 * scale))), "center")
    _board_text(frame, p.rule, x0 + pw - 10 * scale, y0 + 14 * scale,
                0.36 * scale, BOARD_DIM, 1, "right")

    n = len(p.options)
    gap = int(16 * scale)
    bw = (pw - gap * (n + 1)) // max(1, n)
    by0, by1 = int(y0 + 56 * scale), y0 + ph - gap
    rects = []
    for k, opt in enumerate(p.options):
        bx0 = x0 + gap + k * (bw + gap)
        bx1 = bx0 + bw
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), BUTTON_BG, -1)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), BOARD_GREEN,
                      max(1, int(scale)))
        label = f"{k + 1}  {OPTION_TEXT.get(opt, str(opt).upper())}"
        thumb = thumbs.get(opt) if with_thumbs else None
        if thumb is not None:
            t = min(by1 - by0 - int(46 * scale), bw - 2 * gap)
            tx, ty = (bx0 + bx1) // 2 - t // 2, by0 + int(8 * scale)
            if t > 8 and ty + t <= H and tx >= 0:
                frame[ty:ty + t, tx:tx + t] = cv2.resize(thumb, (t, t))
                cv2.rectangle(frame, (tx, ty), (tx + t, ty + t), BOARD_GREY, 1)
            _board_text(frame, label, (bx0 + bx1) / 2.0, by1 - 20 * scale,
                        0.6 * scale, BOARD_WHITE, max(1, int(round(1.4 * scale))),
                        "center")
        else:
            _board_text(frame, label, (bx0 + bx1) / 2.0, (by0 + by1) / 2.0,
                        _fit(label, 0.62 * scale, 1, bw - 2 * gap), BOARD_WHITE,
                        max(1, int(round(1.4 * scale))), "center")
        rects.append((bx0, by0, bx1, by1, "answer",
                      {"kind": p.kind, "choice": opt}))
    return rects


# ---- marks on the table -----------------------------------------------------

def _dashed_circle(img, centre, radius, colour, thick, dashes=18):
    for k in range(dashes):
        a0 = 360.0 * k / dashes
        cv2.ellipse(img, centre, (radius, radius), 0, a0,
                    a0 + 360.0 / dashes * 0.55, colour, thick, cv2.LINE_AA)


def draw_head_string(frame, session, scale):
    """F-02: shade the kitchen while a restricted ball in hand is owed.

    Green once the cue ball is placed legally, red while it is not, amber while
    there is nothing to judge yet. If no rack has been seen, the system does not
    know which end is the head, and says so on the zone rather than drawing a
    confident line that might be in the wrong half.
    """
    game = session.game
    if game.ball_in_hand != games.BEHIND_HEAD_STRING or game.over:
        return
    zone = session.table.head_zone()
    if zone is None:
        return
    ok = session.placement
    colour = BOARD_GREEN if ok else BOARD_RED if ok is False else BOARD_AMBER
    _shade(frame, zone, colour, 0.16)
    t = session.table
    x0, y0, x1, y1 = zone
    thick = max(2, int(round(3 * scale)))
    if t.long_axis == "x":
        lx = x1 if t.head_at_low else x0
        cv2.line(frame, (lx, y0), (lx, y1), colour, thick, cv2.LINE_AA)
    else:
        ly = y1 if t.head_at_low else y0
        cv2.line(frame, (x0, ly), (x1, ly), colour, thick, cv2.LINE_AA)
    text = ("CUE BALL BEHIND THE HEAD STRING" if ok else
            "MOVE THE CUE BALL BEHIND THE LINE" if ok is False else
            "BALL IN HAND - BEHIND THE HEAD STRING")
    if not t.head_known:
        text += " (HEAD END NOT YET SEEN)"
    _board_text(frame, text, (x0 + x1) / 2.0, y0 + 40 * scale, 0.62 * scale,
                colour, max(1, int(round(1.6 * scale))), "center")


def draw_marks(frame, session, items, scale):
    """F-08's frozen balls and M-07's put-it-back spot."""
    if not session.shots.moving:
        frozen = set(session.frozen)
        for tid, x, y, r, _lb in items:
            if tid in frozen:
                _dashed_circle(frame, (x, y), int(r * 1.45), FROZEN_BGR,
                               max(1, int(round(1.5 * scale))), dashes=10)
    at = session.replace_at
    if at:
        x, y = int(at[0]), int(at[1])
        r = int(session.ball_r)
        _dashed_circle(frame, (x, y), r, BOARD_AMBER,
                       max(2, int(round(2 * scale))))
        cv2.drawMarker(frame, (x, y), BOARD_AMBER, cv2.MARKER_CROSS,
                       max(6, r // 2), max(1, int(scale)), cv2.LINE_AA)
        _board_text(frame, "PUT THE BALL BACK HERE", x, y - r - 18 * scale,
                    0.56 * scale, BOARD_AMBER, max(1, int(round(1.4 * scale))),
                    "center")


def draw_timeout(frame, session, scale):
    """H-05's visible clock. Advisory: it reads over, it does not stop play."""
    t = session.game.timeouts
    if t.active is None:
        return
    H, W = frame.shape[:2]
    left = rules.TIMEOUT_SECONDS - t.elapsed(session.frame, session.fps)
    over = left < 0
    secs = int(abs(left))
    clock = f"{'+' if over else ''}{secs // 60}:{secs % 60:02d}"
    bw, bh = int(300 * scale), int(118 * scale)
    x0, y0 = W - bw - int(24 * scale), int(24 * scale)
    colour = BOARD_AMBER if over else BOARD_GREEN
    _shade(frame, (x0, y0, x0 + bw, y0 + bh), (10, 14, 11), 0.85)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), colour,
                  max(1, int(round(2 * scale))))
    player = t.active["player"]
    _board_text(frame, f"TIME OUT{DOT}{session.game.name(player)}",
                x0 + bw / 2.0, y0 + 20 * scale, 0.48 * scale, BOARD_GREY, 1,
                "center")
    _board_text(frame, clock, x0 + bw / 2.0, y0 + 58 * scale, 1.25 * scale,
                colour, max(2, int(round(2.5 * scale))), "center")
    sub = (f"{t.remaining(player)} LEFT BEFORE THIS ONE" if t.active["chargeable"]
           else "NOT CHARGED - RACK NOT BROKEN")
    _board_text(frame, sub, x0 + bw / 2.0, y0 + 94 * scale, 0.40 * scale,
                BOARD_DIM, 1, "center")
    done = max(0.0, min(1.0, 1.0 - left / rules.TIMEOUT_SECONDS))
    cv2.rectangle(frame, (x0, y0 + bh - 4), (x0 + int(bw * done), y0 + bh),
                  colour, -1)


def draw_undo_banner(frame, session, scale):
    """W-02's ten seconds, as a banner that is itself the undo button."""
    left = session.undo_banner
    if left is None:
        return []
    H, W = frame.shape[:2]
    bw, bh = int(min(W * 0.72, 980 * scale)), int(58 * scale)
    x0, y0 = (W - bw) // 2, int(20 * scale)
    _shade(frame, (x0, y0, x0 + bw, y0 + bh), (10, 14, 11), 0.9)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), BOARD_AMBER,
                  max(1, int(round(2 * scale))))
    secs = left * rules.EIGHT_EARLY_UNDO_SECONDS
    text = f"{session.game.status}{DOT}TAP TO UNDO ({secs:.0f}s)"
    _board_text(frame, text, x0 + bw / 2.0, y0 + bh / 2.0 - 3 * scale,
                _fit(text, 0.62 * scale, 1, bw - 30 * scale), BOARD_AMBER,
                max(1, int(round(1.5 * scale))), "center")
    cv2.rectangle(frame, (x0, y0 + bh - 5), (x0 + int(bw * left), y0 + bh),
                  BOARD_AMBER, -1)
    return [(x0, y0, x0 + bw, y0 + bh, "command", {"name": "undo"})]


def draw_win(frame, session, scale):
    """Celebrate the rack over the video: a band, a name, and some movement."""
    name, progress = session.celebration
    game = session.game
    H, W = frame.shape[:2]
    grow = min(1.0, progress / 0.12) ** 0.5
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
    if game.match_winner is not None:
        text = f"{game.name(game.match_winner)} WINS THE MATCH"
    thick = max(2, int(round(3.5 * scale)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    tint = tuple(int(20 + (c - 20) * fade) for c in BOARD_GREEN)
    cv2.putText(frame, text, ((W - tw) // 2, int(top + band * 0.46 + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, size, tint, thick, cv2.LINE_AA)

    sub = DOT.join([f"RACK {game.rack}", f"{game.score(0)} - {game.score(1)}",
                    game.status.split(" - ")[-1]])
    _board_text(frame, sub, W / 2, top + band * 0.82, 0.66 * scale,
                tuple(int(20 + (c - 20) * fade) for c in BOARD_GREY),
                max(1, int(round(1.3 * scale))), "center")
    return frame


def draw_intrusions(img, intrusions, scale=1.0, debug=False):
    """Outline what was erased from the mask - only when asked to.

    The original prototype captioned every arm "NOT BALLS" in warning red. That
    is a debugging view, and M-05 is explicit that occlusion is normal play and
    must not look like an error, so it is off by default and the board carries a
    quiet dot instead. --debug-intrusions brings it back.
    """
    if not debug:
        return
    for it in intrusions or []:
        x, y, w, h = it.box
        thick = max(1, int(round(scale * (2 if it.bulky else 1))))
        cv2.rectangle(img, (x, y), (x + w, y + h), BOARD_RED, thick)
        if not it.bulky:
            continue
        text = f"NOT BALLS - {it.why}".upper()
        org = (x, max(int(14 * scale), y - int(8 * scale)))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    BOARD_RED, thick, cv2.LINE_AA)


# ---- M-13: review -----------------------------------------------------------

class Review:
    """M-13: the last ten seconds, side by side, and BPS does not overrule.

    The client asks for both cameras. This prototype has one, so the second
    panel says so plainly rather than being quietly left out - the layout is
    the product, and a second feed drops into it without a redesign.

    The clip is frozen at the moment REVIEW is tapped, so play carrying on at
    the table does not scroll the evidence away while two people are arguing
    over it.
    """

    def __init__(self, fps, width, seconds=rules.REVIEW_SECONDS, auto_close=False):
        self.fps = max(1.0, fps)
        self.buffer = deque(maxlen=max(1, int(round(seconds * self.fps))))
        self.width = max(160, width // 4)
        self.auto_close = auto_close
        self.clip = None
        self.pos = 0

    @property
    def active(self):
        return self.clip is not None

    def push(self, frame):
        h = int(frame.shape[0] * self.width / frame.shape[1])
        self.buffer.append(cv2.resize(frame, (self.width, h),
                                      interpolation=cv2.INTER_AREA))

    def open(self):
        self.clip = list(self.buffer) or None
        self.pos = 0

    def close(self):
        self.clip = None

    def draw(self, frame, scale):
        if not self.active:
            return []
        H, W = frame.shape[:2]
        _shade(frame, (0, 0, W, H), (8, 12, 9), 0.86)
        gap = int(W * 0.02)
        pw = (W - 3 * gap) // 2
        ph = int(pw * H / W)
        top = (H - ph) // 2
        at = min(self.pos, len(self.clip) - 1)
        frame[top:top + ph, gap:gap + pw] = cv2.resize(self.clip[at], (pw, ph))
        cv2.rectangle(frame, (gap, top), (gap + pw, top + ph), BOARD_GREEN, 2)

        rx = 2 * gap + pw
        cv2.rectangle(frame, (rx, top), (rx + pw, top + ph), (30, 38, 33), -1)
        cv2.rectangle(frame, (rx, top), (rx + pw, top + ph), BOARD_DIM, 2)
        _board_text(frame, "CAMERA 2", rx + pw / 2.0, top + ph * 0.45,
                    0.9 * scale, BOARD_DIM, max(1, int(round(2 * scale))),
                    "center")
        _board_text(frame, "NOT CONNECTED", rx + pw / 2.0, top + ph * 0.56,
                    0.6 * scale, BOARD_DIM, 1, "center")

        ago = (len(self.clip) - 1 - at) / self.fps
        _board_text(frame, f"CAMERA 1{DOT}-{ago:.1f}s", gap, top - 22 * scale,
                    0.6 * scale, BOARD_GREEN, max(1, int(round(1.4 * scale))))
        _board_text(frame, "REVIEW - LAST 10 SECONDS", W / 2.0, 40 * scale,
                    0.9 * scale, BOARD_WHITE, max(1, int(round(2 * scale))),
                    "center")
        cv2.rectangle(frame, (gap, top + ph + 6),
                      (gap + int(pw * (at + 1) / len(self.clip)), top + ph + 12),
                      BOARD_GREEN, -1)
        _board_text(frame, "BPS SHOWS THE EVIDENCE - PLAYERS DECIDE. "
                           "CORRECT THE SCORE WITH UNDO.",
                    W / 2.0, top + ph + 48 * scale, 0.56 * scale, BOARD_GREY, 1,
                    "center")

        bw, bh = int(260 * scale), int(52 * scale)
        bx, by = (W - bw) // 2, H - bh - int(26 * scale)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), BUTTON_BG, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), BOARD_GREEN, 1)
        _board_text(frame, "CLOSE REVIEW", bx + bw / 2.0, by + bh / 2.0,
                    0.6 * scale, BOARD_WHITE, max(1, int(round(1.4 * scale))),
                    "center")

        self.pos += 1
        if self.pos >= len(self.clip):
            if self.auto_close:
                self.close()
            else:
                self.pos = 0
        return [(bx, by, bx + bw, by + bh, "close_review", {})]


# ---- input ------------------------------------------------------------------

KEYMAP = {"d": "defence", "t": "timeout", "e": "end_timeout",
          "x": "cancel_timeout", "u": "undo", "k": "confirm", "o": "reopen",
          "f": "foul", "s": "stalemate", "r": "review", "l": "call_loss",
          "z": "frozen", "g": "swap_groups", "p": "push_out"}


def key_tap(key, session):
    """A keypress as a tap, or None."""
    if key == 8:                               # backspace: the back button
        return ("command", {"name": "undo"})
    ch = chr(key) if 0 <= key < 128 else ""
    if ch in KEYMAP:
        return ("command", {"name": KEYMAP[ch]})
    prompt = session.prompt
    if ch.isdigit() and prompt is not None:
        n = int(ch) - 1
        if 0 <= n < len(prompt.options):
            return ("answer", {"kind": prompt.kind,
                               "choice": prompt.options[n]})
    return None


def on_mouse(event, x, y, _flags, ui):
    if event != cv2.EVENT_LBUTTONUP:
        return
    X, Y = x / PREVIEW_SCALE, y / PREVIEW_SCALE
    for x0, y0, x1, y1, kind, kw in reversed(ui["rects"]):
        if x0 <= X <= x1 and y0 <= Y <= y1:
            ui["taps"].append((kind, kw))
            return


def parse_taps(spec):
    """--taps "FRAME=COMMAND[/ARG[/ARG]],..." -> sorted [(frame, tap)].

    Scripted taps are how a rendered video demonstrates the HUD without anyone
    at a keyboard: "300=timeout,420=end_timeout" shows H-05 end to end, and
    "600=answer/groups/stripe" answers M-06's prompt.
    """
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        frame, _, command = part.partition("=")
        bits = command.split("/")
        name = bits[0]
        if name == "answer":
            tap = ("answer", {"kind": bits[1] if len(bits) > 1 else None,
                              "choice": bits[2] if len(bits) > 2 else None})
        elif name == "call_pocket":
            tap = ("command", {"name": name,
                               "pocket": bits[1] if len(bits) > 1 else None})
        else:
            tap = ("command", {"name": name})
        out.append((int(frame), tap))
    return sorted(out, key=lambda item: item[0])


def run_tap(session, review, tap):
    """Apply one tap. Returns True when it did something."""
    kind, kw = tap
    if kind == "answer":
        return session.answer(kw.get("kind"), kw.get("choice"))
    if kind == "close_review":
        review.close()
        return True
    name = kw.get("name")
    if name == "review" and review.active:
        review.close()
        return True
    ok = session.command(name, **{k: v for k, v in kw.items() if k != "name"})
    if name == "review" and ok:
        review.open()
    return ok


def describe_tap(tap):
    kind, kw = tap
    if kind == "answer":
        return f"ANSWER {kw.get('kind')} -> {kw.get('choice')}"
    return str(kw.get("name", kind)).upper()


# ---- composition ------------------------------------------------------------

def render(raw, items, intrusions, index, total, fps_now, scale, pockets,
           session, board_h, bar_h, review, debug_intrusions=False):
    """One output frame: scoreboard, button bar, then the table and its marks.

    Returns (image, rects) where rects are every clickable area in OUTPUT
    coordinates, topmost last.
    """
    frame = raw.copy()
    thumbs = {}
    if session.prompt is not None and session.prompt.kind == "groups":
        thumbs = group_thumbs(raw, items, int(110 * scale))

    detect_pocket.draw_pockets(frame, pockets, scale)
    draw_intrusions(frame, intrusions, scale, debug_intrusions)
    draw_head_string(frame, session, scale)
    draw_overlay(frame, items, scale)
    draw_marks(frame, session, items, scale)
    if session.celebration:
        draw_win(frame, session, scale)
    draw_timeout(frame, session, scale)
    on_video = draw_undo_banner(frame, session, scale)
    if review.active:
        on_video = review.draw(frame, scale)
    else:
        on_video += draw_prompt(frame, session, scale, thumbs)

    W = frame.shape[1]
    board = scoreboard(W, board_h, session, index, total, fps_now)
    bar, on_bar = button_bar(W, bar_h, session)
    out = np.vstack([board, bar, frame])
    rects = ([(x0, y0 + board_h, x1, y1 + board_h, k, kw)
              for x0, y0, x1, y1, k, kw in on_bar] +
             [(x0, y0 + board_h + bar_h, x1, y1 + board_h + bar_h, k, kw)
              for x0, y0, x1, y1, k, kw in on_video])
    return out, rects


def table_key(pockets_path, W, H):
    return f"{os.path.basename(pockets_path)}@{W}x{H}"


def load_table_memory(path, key):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get(key, {})
    except (OSError, ValueError):
        return {}


def save_table_memory(path, key, entry):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data[key] = entry
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help=f"input video (default: first of {', '.join(DEFAULT_VIDEOS)})")
    ap.add_argument("--out", default=DEFAULT_OUTPUT,
                    help=f"annotated output video (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--show", action="store_true",
                    help="live window with a clickable HUD")
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
    # M-15: players are Player 1 and Player 2 by table side; names are optional.
    ap.add_argument("--player-1", "--player-a", dest="player_1",
                    default="Player 1", help="left-hand side (default Player 1)")
    ap.add_argument("--player-2", "--player-b", dest="player_2",
                    default="Player 2", help="right-hand side (default Player 2)")
    ap.add_argument("--game", choices=(games.EIGHT_BALL, games.NINE_BALL),
                    default=games.EIGHT_BALL, help="discipline (default 8ball)")
    ap.add_argument("--mode", choices=games.MODES, default=games.CASUAL,
                    help="league enforces, casual advises, practice only "
                         "measures (default casual)")
    ap.add_argument("--sl1", type=int, default=None,
                    help="Player 1's Skill Level, 1-7. Applied only in league "
                         "mode with defence marking on")
    ap.add_argument("--sl2", type=int, default=None,
                    help="Player 2's Skill Level, 1-7")
    ap.add_argument("--defence-marking", choices=("auto", "on", "off"),
                    default="auto",
                    help="the MARK DEFENCE button; auto = on for league and "
                         "practice (default auto)")
    ap.add_argument("--pocket-marking", choices=("auto", "on", "off"),
                    default="auto",
                    help="8-ball called pocket; auto = on for league only")
    ap.add_argument("--format", dest="fmt", choices=skill_level.FORMATS,
                    default=skill_level.OPEN,
                    help="league format - time-outs, races, Masters rules "
                         "(default open)")
    ap.add_argument("--break-assigns", choices=("on", "off"), default="on",
                    help="one category pocketed on the break takes that group "
                         "(on, the manual); off keeps the table open after "
                         "every break")
    ap.add_argument("--innings", choices=(games.SCORESHEET_INNINGS,
                                          games.BREAKER_INNINGS),
                    default=games.SCORESHEET_INNINGS,
                    help="scoresheet: the lag loser closes each inning; "
                         "breaker: the inning ticks when the breaker returns")
    ap.add_argument("--lag-winner", type=int, choices=(1, 2), default=1,
                    help="which player won the lag: breaks first and is the "
                         "top of every inning (default 1)")
    ap.add_argument("--taps", default=None,
                    help='scripted HUD taps, e.g. "300=timeout,420=end_timeout,'
                         '600=answer/groups/stripe"')
    ap.add_argument("--record", default=None,
                    help="match record JSON (default: <out>_record.json)")
    ap.add_argument("--pot-radius", type=float, default=logic.POCKET_MOUTH,
                    help="how close to a pocket a ball's last sighting must be "
                         f"to count as potted, in ball radii "
                         f"(default {logic.POCKET_MOUTH})")
    ap.add_argument("--pocket-block", type=float, default=POCKET_BLOCK,
                    help="radius of the blind spot cut out of the table mask at "
                         f"each pocket, in ball radii (default {POCKET_BLOCK})")
    ap.add_argument("--no-hand-gate", action="store_true",
                    help="judge every settling the moment the balls stop, hand "
                         "or no hand")
    ap.add_argument("--debug-intrusions", action="store_true",
                    help="outline and caption erased arms and cues (off by "
                         "default: occlusion is normal play)")
    ap.add_argument("--events", default=None,
                    help="write the identity, pot and game logs to this JSON file")
    ap.add_argument("--label-scale", type=float, default=1.0,
                    help="multiplier on the overlay text size (default 1.0)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    record_path = args.record or os.path.splitext(out_path)[0] + "_record.json"
    if not os.path.isabs(record_path):
        record_path = os.path.join(here, record_path)

    print(rules.banner())
    skill_level.assert_clean()

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

    pockets = None
    pockets_path = (args.pockets if os.path.isabs(args.pockets)
                    else os.path.join(here, args.pockets))
    if not args.no_pockets:
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

    label_scale = max(0.35, (H / 1080.0) * 0.95 * args.label_scale)
    board_h = max(96, int(round(H * BOARD_FRACTION)))
    bar_h = max(40, int(round(H * BAR_FRACTION)))
    play_fps = max(1.0, src_fps / args.stride)

    defence = (args.defence_marking == "on" or
               (args.defence_marking == "auto"
                and args.mode in (games.LEAGUE, games.PRACTICE)))
    pocket_marking = (None if args.pocket_marking == "auto"
                      else args.pocket_marking == "on")

    # M-09: the first rack on a table is confirmed by a tap. Only a live HUD can
    # take one, so a rendered run treats the table as already known and says so.
    memory_path = os.path.join(here, TABLE_MEMORY)
    key = table_key(pockets_path, W, H)
    memory = load_table_memory(memory_path, key)
    first_rack_ok = (not args.show) or bool(memory.get("first_rack_confirmed"))

    session = logic.GameSession(pockets, cal["r"], fps=src_fps,
                                players=(args.player_1, args.player_2),
                                mouth=args.pot_radius, discipline=args.game,
                                mode=args.mode, levels=(args.sl1, args.sl2),
                                defence_marking=defence,
                                pocket_marking=pocket_marking,
                                first_rack_confirmed=first_rack_ok,
                                sample_fps=play_fps, fmt=args.fmt,
                                break_assigns=args.break_assigns == "on",
                                innings_rule=args.innings,
                                lag_winner=args.lag_winner - 1)
    game = session.game
    print(f"Game:   {args.game}  {args.mode}  {args.fmt} format  defence marking "
          f"{'on' if defence else 'off'}  pocket marking "
          f"{'on' if game.pocket_marking else 'off'}")
    if (args.sl1 or args.sl2) and not game.handicapped:
        print("        Skill Levels given but not applied - they only count in "
              "league mode with defence marking on (H-03)")
    elif game.handicapped:
        print(f"        {skill_level.describe(args.sl1)} vs "
              f"{skill_level.describe(args.sl2)}, race {game.race()}"
              + ("  (PROVISIONAL CHART - replace from the client's)"
                 if game.chart.provisional else ""))
    if not args.show:
        print("        rendered run: no one can tap, so M-09's first-rack "
              "confirmation is skipped and prompts are logged, not answered")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             play_fps, (W, H + board_h + bar_h))
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
    review = Review(play_fps, W, auto_close=not args.show)
    scripted = parse_taps(args.taps)
    ui = {"rects": [], "taps": []}
    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse, ui)
        print("HUD:    click the buttons, or keys: "
              + "  ".join(f"{k} {v}" for k, v in KEYMAP.items())
              + "  1-3 answer  space pause  q quit")

    idx = args.start
    processed = 0
    logged = 0
    counts = []
    id_counts = []
    t0 = time.perf_counter()
    fps_now = 0.0
    first = True
    paused = False
    current = None
    try:
        while True:
            advanced = False
            if not paused or current is None:
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
                if csv_writer:
                    for tid, x, y, r, label in items:
                        csv_writer.writerow([
                            idx, tid,
                            "" if label is None else label.cls,
                            "" if label is None else label.confidence,
                            x, y, r])

                review.push(frame)
                current = (frame, items, intrusions)
                processed += 1
                elapsed = time.perf_counter() - t0
                fps_now = processed / elapsed if elapsed else 0.0
                advanced = True

            while scripted and scripted[0][0] <= idx:
                _at, tap = scripted.pop(0)
                done = run_tap(session, review, tap)
                print(f"  TAP   frame {idx}  {describe_tap(tap)}"
                      f"{'' if done else '  (not available now)'}", flush=True)
            while ui["taps"]:
                tap = ui["taps"].pop(0)
                done = run_tap(session, review, tap)
                print(f"  TAP   frame {idx}  {describe_tap(tap)}"
                      f"{'' if done else '  (not available now)'}", flush=True)

            events = game.record.events
            while logged < len(events):
                e = events[logged]
                dot = " *" if e.uncertain else ""
                print(f"  GAME  frame {e.frame}  [{e.rule or '-'}] {e.text}{dot}",
                      flush=True)
                logged += 1

            out, ui["rects"] = render(current[0], current[1], current[2], idx,
                                      total, fps_now, label_scale, pockets,
                                      session, board_h, bar_h, review,
                                      args.debug_intrusions)
            if advanced:
                writer.write(out)

            if args.show:
                preview = cv2.resize(out, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                                     interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, preview)
                key_code = cv2.waitKey(30 if paused else 1)
                if key_code != -1:
                    k = key_code & 0xFF
                    if k in (ord("q"), 27):
                        print("Stopped by user.")
                        break
                    if k == ord(" "):
                        paused = not paused
                    else:
                        tap = key_tap(k, session)
                        if tap:
                            ui["taps"].append(tap)

            if advanced and processed % 50 == 0:
                print(f"  {processed} frames  |  frame {idx}/{total}  "
                      f"|  {counts[-1]} balls, "
                      f"{id_counts[-1]} named  |  {fps_now:.1f} fps", flush=True)

            if advanced and args.frames and processed >= args.frames:
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

    if args.show and session.first_rack_confirmed and pockets is not None:
        save_table_memory(memory_path, key, {"first_rack_confirmed": True})

    if counts:
        print(f"\nProcessed {processed} frames in {time.perf_counter() - t0:.1f}s "
              f"({fps_now:.1f} fps)")
        print(f"Balls per frame: min {min(counts)}  max {max(counts)}  "
              f"mean {sum(counts) / len(counts):.1f}")
        if id_counts and registry is not None:
            print(f"Labelled per frame: min {min(id_counts)}  max {max(id_counts)}  "
                  f"mean {sum(id_counts) / len(id_counts):.1f}")

        summary = session.summary()
        print(f"\nGame:   {game.players[0]} {game.score(0)} - {game.score(1)} "
              f"{game.players[1]}   ({args.game}, {args.mode})")
        if args.game == games.EIGHT_BALL:
            print(f"        {game.group_name(0)} vs {game.group_name(1)}")
        else:
            print(f"        {game.dead} dead balls this rack, invariant "
                  f"{'ok' if summary['invariant_ok'] else 'BROKEN'}")
        print(f"        {game.rack} rack(s), racks won {game.racks_won}, "
              f"{game.shot} shots, inning {game.inning}")
        print(f"        fouls {game.fouls}, defences {game.defences}, "
              f"time-outs {game.timeouts.used}")
        print(f"        {summary['pots']} pots, {session.phantoms} taken back "
              f"(M-04/W-10), {session.replaced} moved by hand (M-07), "
              f"{len(game.record.flagged())} flagged for review")
        for i in (0, 1):
            a = game.analytics.summary(i)
            print(f"        {game.players[i]}: TSR {a['TSR']:.0%}  "
                  f"CRR {a['CRR']:.0%}  SDR {a['SDR']:.0%}  ({a['shots']} shots)")
        print(f"        hands/cues on the table in {session.busy_frames} of "
              f"{processed} frames, {len(session.episodes)} episodes of motion")
        if session.unexplained:
            print(f"        UNEXPLAINED: {len(session.unexplained)} settlings "
                  f"lost a ball with nothing credited, at frames "
                  f"{[e['frame'] for e in session.unexplained][:12]}")
        if session.prompts:
            print(f"        unanswered prompts: "
                  f"{[p.text for p in session.prompts]}")
        if game.record.unconfirmed:
            print("        SCORE UNCONFIRMED - a correction was left open")
        print(f"        {game.status}")
        if COPY_OFFENDERS:
            print(f"        H-04 VIOLATION - drawn on the HUD: "
                  f"{sorted(COPY_OFFENDERS)}")

    game.record.save(record_path, summary=session.summary())
    print(f"Saved:  {record_path}")

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
                       "game": game.record.as_list(),
                       "summary": session.summary()}, fh, indent=2)
        print(f"Saved:  {ev_path}")
    print(f"Saved:  {out_path}")
    if args.csv:
        print(f"Saved:  {args.csv}")


if __name__ == "__main__":
    main()
