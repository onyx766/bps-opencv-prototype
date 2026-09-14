"""
BPS prototype - eight-ball, as rules over what the camera can actually see.

The vision layer knows three things: a tracked ball vanished next to a pocket,
the balls are moving or they are not, and roughly what each ball is (cue, 8,
stripe, solid). Everything a scoreboard wants - whose turn it is, who is on
stripes, whether that was a foul - is inferred from those three. So this file
is a state machine over POT EVENTS and SHOT BOUNDARIES, not over frames.

Three pieces, feeding each other:

  1. PotDetector    A track that disappears inside a pocket's mouth and does not
                    come back was potted. A track that disappears out in open
                    table was occluded - an arm, a cue, a bad frame - and is
                    ignored. That one distinction is the whole detector, and the
                    pockets clicked in detect_pocket.py are what make it possible.

  2. ShotSegmenter  A shot runs from the CUE BALL being struck to every ball
                    settling again. Only the cue ball may start one, because
                    everything else that moves over a pool table is a player.
                    Rules have to be applied per shot and not per pot, because
                    "potted my ball but scratched" is a single outcome that does
                    not exist until the whole shot is known.

  3. EightBallGame  The rules themselves, driven by shot outcomes. Pure state -
                    no OpenCV, no frames - so it can be tested and read on its own.

RULES IMPLEMENTED
    break, table open until someone legally pots (bar rules: the break assigns
    too - see ASSIGN_ON_BREAK), group assignment on the first legal pot, potting your own
    group continues your turn, missing passes it, scratch is a foul, potting an
    opponent's ball is a foul (bar rules - WPA calls this a miss, not a foul,
    but it is the rule the scoreboard was asked for), opponent's balls still
    count for the opponent once they are down, the 8 is legal only after your
    group is clear, the 8 potted early or with a scratch loses, the 8 on the
    break re-racks, ball in hand after a foul.

RULES NOT IMPLEMENTED, because nothing in the video can see them
    calling the pocket, first-contact fouls (hitting the wrong ball first), the
    no-rail-after-contact foul, jumped balls, push-out, and the three-consecutive-
    foul loss. A pot is judged by what went down, which is all the pixels offer.
    Every one of these would need either a caller or contact detection, and
    guessing at them would produce confident wrong verdicts rather than none.
"""

from collections import Counter, deque, namedtuple

CUE, EIGHT, STRIPE, SOLID = "cue", "eight", "stripe", "solid"
UNKNOWN = "unknown"

#: Balls in a group, and the other group.
GROUP_SIZE = 7
OTHER = {STRIPE: SOLID, SOLID: STRIPE}

#: Does a ball potted on the BREAK win its group? Bar rules say yes - first ball
#: down picks your side - and that is the table this footage was shot on. WPA
#: tournament rules say no: the break never assigns, the table stays open, and
#: the breaker chooses on their next legal pot. Set False for that. Either way a
#: break that drops BOTH a stripe and a solid leaves the table open, because
#: nothing in the shot says which group the breaker meant to take.
ASSIGN_ON_BREAK = True

#: How close to a pocket a track's last known position has to be to count as
#: potted, in ball radii, FOR A BALL AT REST. A ball never disappears at the
#: pocket lip, it disappears at the edge of the eroded table mask, which sits an
#: erosion width plus a ball radius further out. A real pot into the bottom-left
#: pocket of this footage had its last sighting 84 px from the clicked centre -
#: 4.2 radii - and a 3.0 cut threw it away. 5.0 clears that with margin.
#:
#: A MOVING ball gets its own speed added on top, because the reach is really a
#: question about the next frame: a ball still 135 px from the pocket but
#: closing 118 px per frame is in the pocket before it can be seen again, and
#: that exact ball - a cue ball scratching into the top middle - was thrown away
#: by a fixed cut. Speed is only ever added to a ball that is moving INTO the
#: pocket, so nothing at rest gets a wider mouth, which is where phantoms live.
#: Widening this is only safe because three other gates now stand behind it: the
#: shot has to be in play, the track has to be old enough to be a ball, and the
#: table must not have lost a crowd of tracks at once.
POCKET_MOUTH = 5.0

#: Frames a track must stay missing before its pot is believed. A ball behind a
#: player's arm comes back within a frame or two.
POT_CONFIRM = 3

#: Frames a track must have been seen for before its disappearance can be a
#: pot. A real ball is tracked for as long as it is on the table; a fragment of
#: an arm or a cue crossing the felt lives a frame or two and then breaks up,
#: and if it breaks up next to a pocket it looks exactly like a ball going down.
POT_MIN_AGE = 4

#: How much closer to the pocket a ball's last step must take it, in ball radii,
#: before the disappearance counts as a pot. A ball that goes down is travelling
#: into the pocket when it is last seen. A ball whose track simply broke - and
#: at 4.9 fps a struck ball crosses far more than the tracker's search radius,
#: so tracks break constantly - was going wherever it was going, and on the
#: break fifteen of them break at once, near every pocket on the table.
POT_APPROACH_R = 0.5

#: Tracks that may vanish together before the whole batch is called occlusion
#: rather than potting. A player leaning over the table merges with the balls
#: near them into one blob too big to be balls, which server.drop_oversize_blobs
#: erases whole - measured on this footage, sixteen balls became seven in a
#: single frame. Two balls can drop on one shot; nine cannot.
MASS_VANISH = 3

#: A ball is "moving" if a track shifted more than this fraction of a ball
#: radius since the previous frame. Detection jitter is a pixel or two.
MOTION_R = 0.35

#: ... but the CUE ball has to shift this much further to start a shot. A shot
#: begins when the cue ball is STRUCK, and nothing else on the table can begin
#: one: an arm reaching across, a cue laid over the felt, or a ball re-detected
#: half a width off all move something, and all of them used to start a shot,
#: end it, and hand the turn to the other player a second later.
#:
#: The cut is measured, and the two cases are an order of magnitude apart at
#: this frame rate. A player carrying the cue ball back to the table by hand
#: moved it 19 and 20 px on consecutive frames; a struck cue ball moved 131,
#: 210, 213, 120 and 106. Three radii (60 px here) sits in the gap, which
#: matters because a hand-placed ball used to open a shot, and a shot that opens
#: while a player is standing over the table is where phantom pots come from.
CUE_MOTION_R = 3.0

#: Frames of stillness that end a shot. Must be longer than POT_CONFIRM, or a
#: ball potted late in the shot would be confirmed after the shot it belonged
#: to had already been judged, and would be credited to the next one.
SETTLE_FRAMES = POT_CONFIRM + 2

Pot = namedtuple("Pot", "frame track cls pocket dist closed age")


def _median(values):
    """Middle value - the ball count's defence against a one-frame miscount."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0


class PotDetector:
    """Vanished tracks -> pot events, using the clicked pocket positions.

    Three things have to hold before a disappearance is believed to be a pot:
    it happened inside a pocket's mouth, the track stayed gone, and the track
    was old enough to have been a ball at all. The third is what keeps arms and
    cue sticks out of the score - they break into short-lived fragments, and a
    fragment that dies near a pocket is otherwise indistinguishable from a ball
    going down.
    """

    def __init__(self, pockets, ball_r, mouth=POCKET_MOUTH, confirm=POT_CONFIRM,
                 min_age=POT_MIN_AGE, mass=MASS_VANISH,
                 approach=POT_APPROACH_R):
        self.pockets = list(pockets or [])
        self.reach = mouth * max(1, ball_r)
        self.confirm = confirm
        self.min_age = min_age
        self.mass = mass
        self.approach = approach * max(1, ball_r)
        self.last = {}          # track id -> (x, y, cls)
        self.prev = {}          # ... and where it was the frame before that
        self.age = Counter()    # track id -> frames seen
        self.missing = Counter()
        self.rejected = []      # near-misses, for the run summary

    def update(self, tracked, labels, frame):
        """Returns the pots confirmed on this frame (usually none)."""
        live = set()
        for tid, x, y, _r in tracked:
            live.add(tid)
            label = (labels or {}).get(tid)
            # Keep the LAST KNOWN class: a ball on its way into a pocket is
            # blurred and half-occluded, and the frame it vanishes on is the
            # least reliable one to ask.
            cls = label.cls if label is not None else self.last.get(tid, (0, 0, UNKNOWN))[2]
            if tid in self.last:
                self.prev[tid] = self.last[tid][:2]
            self.last[tid] = (x, y, cls)
            self.age[tid] += 1
            self.missing.pop(tid, None)

        gone = []
        for tid in [t for t in self.last if t not in live]:
            self.missing[tid] += 1
            if self.missing[tid] < self.confirm:
                continue
            x, y, cls = self.last.pop(tid)
            age = self.age.pop(tid, 0)
            was = self.prev.pop(tid, None)
            del self.missing[tid]
            gone.append((tid, x, y, cls, age, was))

        pots = []
        for tid, x, y, cls, age, was in gone:
            pocket, dist = nearest_pocket(x, y, self.pockets)
            if pocket is None:
                continue
            closed = (nearest_pocket(was[0], was[1], [pocket])[1] - dist
                      if was else 0.0)
            # One frame's worth of travel on top of the resting mouth - see
            # POCKET_MOUTH. Only a ball closing on the pocket earns it.
            if dist > self.reach + max(0.0, closed):
                continue                      # vanished in open table: occluded
            why = ("too short-lived to be a ball" if age < self.min_age else
                   f"{len(gone)} tracks vanished together" if len(gone) > self.mass
                   else f"not moving into the pocket (closed {closed:.0f} px)"
                   if closed < self.approach else None)
            if why:
                self.rejected.append({"frame": frame, "track": tid, "cls": cls,
                                      "pocket": pocket.name, "age": age,
                                      "why": why})
                continue
            pots.append(Pot(frame, tid, cls, pocket.name, round(dist, 1),
                            round(closed, 1), age))
        return pots


def nearest_pocket(x, y, pockets):
    """(pocket, distance) closest to a point, or (None, inf).

    Kept here as well as in detect_pocket so the rules layer does not have to
    import the calibration UI to answer "which pocket".
    """
    if not pockets:
        return None, float("inf")
    best = min(pockets, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
    return best, ((best.x - x) ** 2 + (best.y - y) ** 2) ** 0.5


class ShotSegmenter:
    """Moving / settled, from how far the matched tracks shifted per frame.

    A shot STARTS when the cue ball moves and at no other time. That asymmetry
    is the whole point: every other moving thing on this table - an arm, a cue
    stick laid across the felt, a ball re-detected a width off, a pocket blob
    blinking - is not a shot, and treating it as one hands the turn back and
    forth several times a second.

    A shot ENDS when EVERYTHING is still, cue ball included, because the shot is
    not over while an object ball is still rolling toward a pocket.

    Only tracks seen in BOTH frames are measured. Ids that appear or vanish say
    nothing about motion - that is occlusion, not movement.
    """

    def __init__(self, ball_r, move_px=None, cue_move_px=None,
                 settle=SETTLE_FRAMES):
        self.move = move_px if move_px else max(3.0, MOTION_R * ball_r)
        self.cue_move = cue_move_px if cue_move_px else max(6.0,
                                                            CUE_MOTION_R * ball_r)
        self.settle = settle
        self.prev = {}
        self.moving = False
        self.still = 0
        self.cue_seen = False          # has the cue ball ever been identified

    def _shift(self, tid, now):
        if tid is None or tid not in now or tid not in self.prev:
            return 0.0
        return ((now[tid][0] - self.prev[tid][0]) ** 2 +
                (now[tid][1] - self.prev[tid][1]) ** 2) ** 0.5

    def update(self, tracked, labels=None, any_ball_starts=False):
        """Returns "start", "end" or None.

        `any_ball_starts` is the fallback for a run with identification off:
        with no cue ball to watch, any motion has to do, which is the old
        behaviour and the noisy one.
        """
        now = {tid: (x, y) for tid, x, y, _r in tracked}
        cue = next((tid for tid, lb in (labels or {}).items()
                    if lb is not None and lb.cls == CUE and tid in now), None)
        self.cue_seen = self.cue_seen or cue is not None

        cue_shift = self._shift(cue, now)
        # default=0: with no id in common there is nothing to measure - every
        # ball is new, or they all vanished at once behind a player leaning in.
        shifted = max((self._shift(t, now) for t in now if t in self.prev),
                      default=0.0)
        self.prev = now

        if not self.moving:
            struck = (cue_shift > self.cue_move or
                      (any_ball_starts and shifted > self.move))
            if struck:
                self.moving, self.still = True, 0
                return "start"
            return None

        if shifted > self.move:
            self.still = 0
            return None
        self.still += 1
        if self.still >= self.settle:
            self.moving = False
            return "end"
        return None


class EightBallGame:
    """Two players, one rack, the rules listed at the top of this file.

    Scores are DERIVED from what is down rather than incremented per pot, and
    that is deliberate: balls potted on the break belong to nobody until the
    groups are assigned, and then they retroactively belong to whoever got that
    group. A running counter cannot express that; counting the table can.
    """

    def __init__(self, players=("Player A", "Player B")):
        self.players = [str(p) for p in players]
        self.turn = 0
        self.group = [None, None]          # index -> STRIPE / SOLID / None
        self.potted = Counter()            # class -> how many are down
        self.fouls = [0, 0]
        self.ball_in_hand = False
        self.shot = 0
        self.inning = 1
        self.broke = False
        self.over = False
        self.winner = None
        self.status = "BREAK"
        self.log = []

    # ---- what the scoreboard reads ------------------------------------------

    @property
    def open_table(self):
        return self.group[0] is None

    def score(self, i):
        """Balls of this player's group that are down (0 while the table is open)."""
        return self.potted[self.group[i]] if self.group[i] else 0

    def remaining(self, i):
        return GROUP_SIZE - self.score(i) if self.group[i] else GROUP_SIZE

    def on_the_eight(self, i):
        """True once this player's group is clear and only the 8 is left."""
        return self.group[i] is not None and self.score(i) >= GROUP_SIZE

    def group_name(self, i):
        if self.group[i] is None:
            return "OPEN"
        return "STRIPES" if self.group[i] == STRIPE else "SOLIDS"

    # ---- driven by ShotSegmenter --------------------------------------------

    def shot_started(self, frame):
        if self.over:
            return
        self.shot += 1
        self.ball_in_hand = False          # taken, whether or not it was owed

    def shot_ended(self, pots, frame):
        """Judge one shot. `pots` is the classes potted during it, in order."""
        if self.over:
            return
        shooter = self.turn
        scratch = CUE in pots
        objects = [c for c in pots if c in (STRIPE, SOLID)]
        eight = EIGHT in pots
        was_on_eight = self.on_the_eight(shooter)   # BEFORE this shot's pots

        for cls in pots:
            self._rack_down(cls, frame)

        if not self.broke:
            self._judge_break(shooter, pots, objects, eight, scratch, frame)
            return

        # The 8 ends the rack whichever way it goes, so it is judged first and
        # nothing after it matters.
        if eight:
            if scratch:
                self._lose(shooter, "SCRATCHED ON THE 8", frame)
            elif self.open_table or not was_on_eight:
                self._lose(shooter, "8 POTTED EARLY", frame)
            else:
                self._win(shooter, "RAN THE TABLE", frame)
            return

        if self.open_table:
            if scratch:
                self._foul(shooter, "SCRATCH", frame)
            elif objects:
                self._assign(shooter, objects[0], frame)
            elif pots:
                self._note(frame, "pot", f"{self._name(shooter)} POTTED A BALL")
            else:
                self._pass(shooter, "MISS", frame)
            return

        mine = sum(1 for c in objects if c == self.group[shooter])
        theirs = len(objects) - mine
        if scratch:
            self._foul(shooter, "SCRATCH", frame)
        elif theirs:
            self._foul(shooter, "WRONG GROUP POTTED", frame)
        elif mine or UNKNOWN in pots:
            verb = "ON THE 8" if self.on_the_eight(shooter) else "CONTINUES"
            self._note(frame, "pot", f"{self._name(shooter)} {verb}")
        else:
            self._pass(shooter, "MISS", frame)

    # ---- rule helpers -------------------------------------------------------

    def _judge_break(self, shooter, pots, objects, eight, scratch, frame):
        """The break is its own case: no group is won on it, and the 8 re-racks."""
        self.broke = True
        if eight:
            self.rerack(frame)
        elif scratch:
            self._foul(shooter, "SCRATCH ON THE BREAK", frame)
        elif objects and ASSIGN_ON_BREAK and len(set(objects)) == 1:
            self._assign(shooter, objects[0], frame)
        elif objects:
            self._note(frame, "break",
                       f"{self._name(shooter)} BROKE - TABLE OPEN")
        else:
            self._pass(shooter, "DRY BREAK", frame)

    def _rack_down(self, cls, frame):
        """Put a ball down, refusing counts the rack cannot hold.

        Class votes flicker, so an eighth stripe is a misread rather than a
        ball. Clamping keeps the scoreboard honest instead of showing 8/7.
        """
        if cls == CUE:
            return                           # the cue ball comes back out
        cap = 1 if cls == EIGHT else GROUP_SIZE
        if cls in (STRIPE, SOLID, EIGHT) and self.potted[cls] >= cap:
            self._note(frame, "impossible", f"IGNORED AN EXTRA {cls.upper()}")
            return
        self.potted[cls] += 1

    def _assign(self, shooter, cls, frame):
        self.group[shooter] = cls
        self.group[1 - shooter] = OTHER[cls]
        self._note(frame, "assign",
                   f"{self._name(shooter)} TAKES {self.group_name(shooter)}")

    def _foul(self, shooter, why, frame):
        self.fouls[shooter] += 1
        self.ball_in_hand = True
        self._swap(shooter)
        self._note(frame, "foul", f"FOUL - {why} - BALL IN HAND")

    def _pass(self, shooter, why, frame):
        self._swap(shooter)
        self._note(frame, "turn", f"{why} - {self._name(self.turn)} TO SHOOT")

    def _swap(self, shooter):
        self.turn = 1 - shooter
        # An inning is both players having shot, so it turns over when play
        # comes back round to whoever broke.
        if self.turn == 0:
            self.inning += 1

    def _win(self, shooter, why, frame):
        self.over, self.winner = True, shooter
        self._note(frame, "win", f"{self._name(shooter)} WINS - {why}")

    def _lose(self, shooter, why, frame):
        self.over, self.winner = True, 1 - shooter
        self.fouls[shooter] += 1
        self._note(frame, "win", f"{self._name(1 - shooter)} WINS - {why}")

    def rerack(self, frame=0):
        """8 on the break: the rack is void, the same player breaks again."""
        self.group = [None, None]
        self.potted.clear()
        self.broke = False
        self._note(frame, "rerack", "8 ON THE BREAK - RE-RACK")

    def _name(self, i):
        return self.players[i].upper()

    def _note(self, frame, kind, text):
        self.status = text
        self.log.append({"frame": frame, "shot": self.shot, "type": kind,
                         "turn": self.turn, "text": text})

    def summary(self):
        return {
            "players": self.players,
            "groups": [self.group_name(0), self.group_name(1)],
            "score": [self.score(0), self.score(1)],
            "fouls": list(self.fouls),
            "shots": self.shot,
            "innings": self.inning,
            "winner": None if self.winner is None else self.players[self.winner],
            "status": self.status,
        }


class GameSession:
    """What main.py holds: the three pieces above, wired together.

    Feed it every frame's tracks and labels; read the scoreboard off `game`.
    """

    def __init__(self, pockets, ball_r, fps=25.0, players=("Player A", "Player B"),
                 mouth=POCKET_MOUTH, settle=SETTLE_FRAMES, identified=True):
        self.pots = PotDetector(pockets, ball_r, mouth=mouth)
        self.shots = ShotSegmenter(ball_r, settle=settle)
        self.game = EightBallGame(players)
        self.fps = fps if fps and fps > 0 else 25.0
        self.identified = identified
        self.candidates = []       # pots claimed since this shot started
        self.recent = deque(maxlen=settle)     # ball count over the last frames
        # The ball count is only trustworthy at ONE moment: the instant a shot
        # settles. The balls have stopped and the player is still standing back
        # watching them. A count taken any later is inflated by whoever has
        # leaned over the table to line up the next shot - measured on this
        # footage at twenty-two balls on a sixteen-ball table - so each shot is
        # judged against where the PREVIOUS shot left the table, not against
        # what the table looks like while someone is standing in it.
        self.settled_balls = None
        self.before = None
        self.balls = 0
        self.frame = 0
        self.has_pockets = bool(pockets)
        self.all_pots = []
        self.ignored = []          # claims that did not survive a gate

    def update(self, tracked, labels, frame):
        """Returns the pots credited on this frame, for logging/overlay."""
        self.frame, self.balls = frame, len(tracked)
        self.recent.append(len(tracked))

        event = self.shots.update(tracked, labels,
                                  any_ball_starts=not self.identified)
        if event == "start":
            self.game.shot_started(frame)
            self.candidates = []
            self.before = (self.settled_balls if self.settled_balls is not None
                           else _median(self.recent))

        # A ball cannot go down while nothing is moving, so a "pot" found on a
        # still table is an occlusion or a mask flicker - most often a player
        # standing over a pocket between shots. Requiring a shot to be in play
        # is what stops the score climbing on its own. `end` still counts: the
        # frame a shot settles on belongs to that shot.
        claims = self.pots.update(tracked, labels, frame)
        if claims and (self.shots.moving or event == "end"):
            self.candidates.extend(claims)
        elif claims:
            self.ignored.extend(claims)

        credited = []
        if event == "end":
            credited = self._settle_up(frame)
            broke = self.game.broke
            self.game.shot_ended([p.cls for p in credited], frame)
            self.candidates = []
            # A re-rack (the 8 on the break) puts fifteen balls back on the
            # table, so nothing counted before it means anything afterwards.
            if broke and not self.game.broke:
                self.settled_balls = None

        # Before the first shot there is no previous settling to compare with,
        # so the racked table standing still is the reference.
        if self.settled_balls is None and not self.shots.moving:
            self.settled_balls = _median(self.recent)
        return credited

    def _settle_up(self, frame):
        """Decide which of this shot's claims were real, by counting the table.

        A disappearing track is weak evidence. At 4.9 fps a struck ball crosses
        far more than the tracker's search radius, so its id dies mid-flight and
        is reborn elsewhere - and if it died near a pocket it claims a pot that
        never happened. What is NOT weak evidence is how many balls are on the
        table once everything has stopped: a pot means one fewer, permanently,
        and a broken track means the same number.

        So the claims are only ever as good as the drop in the settled count,
        and the best-explained ones get the benefit of it. The count has to be
        read against what the table SHOULD hold, which is not the same as what
        it held last time: a potted object ball stays down, a potted cue ball is
        fished out and put back.
        """
        after = _median(self.recent)
        before = self.before if self.before is not None else after
        dropped = max(0, before - after)

        # The cue ball first, then whichever claim the geometry explains best -
        # far out but closing fast beats near but barely moving. The cue goes
        # first because a missed scratch is the costliest verdict on the table:
        # it hands the shooter a score and the turn, when the rules owe both to
        # the opponent, and the cue ball is the one ball the classifier is
        # actually good at.
        ranked = sorted(self.candidates,
                        key=lambda p: (p.cls != CUE, p.dist - p.closed))
        credited, refused = ranked[:dropped], ranked[dropped:]
        for p in refused:
            self.ignored.append(p)
            self.pots.rejected.append({
                "frame": frame, "track": p.track, "cls": p.cls,
                "pocket": p.pocket, "age": p.age,
                "why": f"table still holds {after} balls (was {before})"})
        self.all_pots.extend(credited)

        # The reference for the NEXT shot is what the table will hold when it
        # is taken, and a scratch is the one pot that comes back: the player
        # fishes the cue ball out and puts it down again. Counting it as gone
        # leaves the reference one short, and the next real pot then looks like
        # no change at all - which is exactly how a solid potted straight after
        # a scratch was read as a miss.
        self.settled_balls = after + sum(1 for p in credited if p.cls == CUE)
        return credited

    @property
    def clock(self):
        """Video position as MM:SS - the scoreboard's game clock."""
        secs = int(self.frame / self.fps)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    @property
    def status(self):
        if not self.has_pockets:
            return "NO POCKETS - POTS NOT TRACKED"
        if self.game.over:
            return self.game.status
        if self.shots.moving:
            return "SHOT IN PLAY"
        # Says why nothing is happening: with no cue ball found, no shot can
        # start, so the turn never changes and the score never moves.
        if self.identified and not self.shots.cue_seen:
            return "NO CUE BALL FOUND - WAITING"
        return self.game.status
