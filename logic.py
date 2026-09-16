"""BPS - the bridge between what the camera sees and the rules the client locked.

The vision layer knows three things: a tracked ball vanished next to a pocket,
the balls are moving or they are not, and roughly what each ball is (cue, 8,
stripe, solid). Everything a scoreboard wants - whose turn it is, who is on
stripes, whether that was a foul - is inferred from those three. So this file
is a state machine over POT EVENTS and SHOT BOUNDARIES, not over frames.

Four pieces, feeding each other:

  1. PotDetector    A track that disappears inside a pocket's mouth and does not
                    come back CLAIMS a pot. A track that disappears out in open
                    table was occluded - an arm, a cue, a bad frame - and claims
                    nothing. Claims are not verdicts: a track can switch ids in a
                    collision and die beside a pocket with its ball still rolling.

                    What decides "pocketed or not" is the SETTLED TABLE: the
                    balls of each class before the shot against the balls of
                    each class once everything has stopped. A ball present
                    before and absent after went down, whatever happened to the
                    ids that carried it mid-shot; a ball that rattled out of the
                    jaw is back on the settled table and was never potted (W-10).
                    Claims only say WHICH POCKET - see GameSession._settle_up.

  2. ShotSegmenter  A shot runs from the CUE BALL being struck to every ball
                    settling again. Rules have to be applied per shot and not
                    per pot, because "potted my ball but scratched" is a single
                    outcome that does not exist until the whole shot is known.

  3. TableGeometry  The six clicked pockets, read as a table: where the rails
                    are (F-04's cushion band, F-08's frozen balls) and where the
                    head string runs (F-02's restricted ball in hand).

  4. GameSession    Assembles a ShotFacts from all of the above and hands it to
                    games.py, which holds the rules themselves.

Running underneath all four: whether the table is BUSY. A hand or a cue over
the cloth is detected in server.py and erased from the mask whole, fingertips
and all, but no single-frame filter can be perfect and none has to be. A hand
appears only while someone is shooting and then leaves; a ball sits still for
minutes. So the useful thing to know is not "which of these blobs is a finger"
but "a hand is on the table right now, do not draw conclusions from this
frame" - and the system simply waits. That is also M-05: occlusion is normal
play, not an error state, and it gets a quiet indicator rather than a banner.

WHAT THIS FILE DECIDES AND WHAT IT REFUSES TO

Everything here produces EVIDENCE, and the rules in games.py decide what the
evidence means. The division matters because the client's suite is mostly a set
of instructions about confidence: F-03 is a foul at 85% and a review below it,
F-09 says an undetermined contact is no foul at all, and M-14 says every
auto-call carries its number so the shaky ones can be marked. A layer that
collapsed "probably" into "yes" would make all three unimplementable.

So the three fields in ShotFacts that the camera cannot always establish -
rail contact, object-ball contact, and which ball was hit first - are TRISTATE.
None means "could not tell", travels into the rules as itself, and comes out as
a review with the clip kept. At 4.9 fps first contact is very often None, and
that is the correct answer rather than a limitation to paper over.

RULES IMPLEMENTED HERE (the rest are in games.py, cited by row in rules.py)
    F-04 rail contact, F-05 object-ball contact, F-08 frozen balls, M-02
    hanging-ball timing, M-04 phantom pockets, M-05 occlusion, M-06 group
    confirmation, M-07 replaced balls, M-08 a missing cue ball, M-09 rack
    detection, W-10 a ball bouncing back out, W-11 a wedged pair.
"""

import math
from collections import Counter, deque, namedtuple

import games
import rules
from games import CUE, EIGHT, SOLID, STRIPE, UNKNOWN, make_game

#: Balls in a group, and the other group. Re-exported: callers used to import
#: these from here and there is no reason to break them.
GROUP_SIZE = games.GROUP_SIZE
OTHER = games.OTHER
EightBallGame = games.EightBallGame
NineBallGame = games.NineBallGame

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
POCKET_MOUTH = 5.0

#: ... and how far out a vanishing is still worth REMEMBERING, in ball radii. A
#: ball struck hard enough is reborn as a new track mid-flight, and a track one
#: frame old has no previous position to earn the speed allowance above: at 5:32
#: in this footage a stripe was last seen 163 px (8 radii) short of the top-left
#: pocket it went into, and was never claimed. Claims out here are only a
#: fallback - used when the table lost more balls than the near claims explain.
POT_FAR_MOUTH = 10.0

#: Frames a track must stay missing before its pot is believed. A ball behind a
#: player's arm comes back within a frame or two.
POT_CONFIRM = 3

#: Frames a track wants to have been seen for before its disappearance is
#: convincing. A ball sitting on the table is tracked for hundreds of frames; a
#: fragment of an arm crossing the felt lives one or two and then breaks up, and
#: if it breaks up next to a pocket it looks like a ball going down.
#:
#: This is a PREFERENCE, not a veto. A ball struck hard outruns the tracker, so
#: the track that dies in the pocket can be a newborn one frame old. Age ranks
#: claims against each other; it never refuses one outright.
POT_MIN_AGE = 4

#: How much closer to the pocket a ball's last step must take it, in ball radii,
#: before the disappearance counts as a pot. A ball that goes down is travelling
#: into the pocket when it is last seen.
POT_APPROACH_R = 0.5

#: Tracks that may vanish together before the whole batch is called occlusion
#: rather than potting. A player leaning over the table merges with the balls
#: near them into one blob too big to be balls, which server.drop_intrusions
#: erases whole - measured on this footage, sixteen balls became seven in a
#: single frame. Two balls can drop on one shot; nine cannot.
MASS_VANISH = 3

#: A ball is "moving" if a track shifted more than this fraction of a ball
#: radius since the previous frame. Detection jitter is a pixel or two.
MOTION_R = 0.35

#: How far a ball has to be from where it was, comparing one settling of the
#: table with the next, to count as having MOVED. In ball radii.
MOVED_R = 1.5

#: And how many balls must have moved between two settlings for the motion in
#: between to have been a shot rather than a player.
#:
#: Comparing SETTLED tables separates them, and it needs no threshold on speed
#: at all. A shot rearranges the table: the cue ball ends up somewhere else and
#: so does whatever it hit, or something is missing because it went down. A
#: player reaching over rearranges nothing. A player placing the cue ball in
#: hand moves exactly one ball - which is why one is not enough, and which is
#: also how M-07 spots a ball that was moved by hand.
#:
#: Measured over 1000 frames of this footage: every real shot moved between 2
#: and 11 balls, and every episode that was not a shot moved 0 or 1.
SHOT_MOVED = 2

#: Most balls one shot can take off the table. A break pots two or three at the
#: outside; a table that loses more than this in one settling is being CLEARED,
#: which is what happens at the end of every rack in this footage - the players
#: pick the balls up and rack them. Measured: the biggest real drop in this
#: match is two, and the clear-up at the end of rack 2 dropped seven in one
#: settling, which was judged a shot and ended the rack on a stale claim.
MAX_SHOT_POTS = 4

#: ... except while the rack is still standing, when the break has to scatter
#: it. A tight rack is where detected centres are least trustworthy, so it is
#: also where the most movement should be demanded.
SHOT_MOVED_BREAK = 5

#: Balls that have to be on the table for it to count as still racked. Not 16:
#: the detector loses one or two inside the triangle, where balls touch.
FULL_RACK = 14

#: ... and for a nine-ball diamond, which is nine balls with the same losses.
FULL_RACK_NINE = 8

#: How close two balls must be, in ball radii between centres, to be touching.
TOUCHING_R = 2.3

#: Touching balls that mean the table has been RACKED rather than merely
#: crowded. Measured: the two racks in this footage read 14, 15, 15, 17 and 19
#: balls in contact. A player leaning over the table breaks into a knot of
#: ball-sized blobs, and those read 8 to 10.
RACK_TOUCHING = 13
RACK_TOUCHING_NINE = 7

#: ... and it has to STAY racked. A triangle sits waiting while someone chalks
#: up and lines up the break.
#:
#: This is also why the rack test cannot wait for a CLEAR table the way the ball
#: count does. Measured at the second rack in this footage: the triangle read as
#: racked for 130 consecutive frames and not one of them was free of a hand or a
#: cue - people stand at the table the whole time they rack up and line up the
#: break. Requiring a clear frame means that rack is never seen, and the next
#: game is scored into the finished one. Persistence does the job instead: a
#: triangle still being built is not thirteen balls in contact for 12 frames.
RACK_FRAMES = 12

#: Frames of stillness that end a shot. Must be longer than POT_CONFIRM, or a
#: ball potted late in the shot would be confirmed after the shot it belonged
#: to had already been judged.
SETTLE_FRAMES = POT_CONFIRM + 2

#: Frames a settling may have waited for the table to clear and still have its
#: census read. A settling held while hands cleared the table is a count nobody
#: should score from - at 4:17 in this footage one held nine frames credited a
#: stripe that undid a correct call - but a hold of a frame or two is the
#: ordinary business of a player straightening up over the shot they just
#: played, and refusing those threw away both of the pots at 7:07 and 7:29.
CENSUS_MAX_HOLD = 3

#: Frames the table must stay clear of hands and cues before its ball count is
#: believed again.
INTRUSION_CLEAR = 2

#: ... and the longest a settled table will be held waiting for that. A verdict
#: that never arrives is worse than one taken over a hand.
INTRUSION_WAIT = 30

#: Seconds the winner's name is celebrated over the video after a rack ends.
WIN_SECONDS = 4.0

#: Width of the cushion band, in ball radii, measured in from the rail line
#: through the pockets. A ball whose CENTRE enters this band has touched the
#: cushion, which is what F-04 needs and what F-08 marks as frozen.
#:
#: Two radii, not one: the rail line runs through the clicked pocket centres,
#: which sit slightly outside the cushion face, so a ball resting against the
#: cushion has its centre about two radii inside that line rather than one.
RAIL_BAND_R = 2.0

#: Frames per second below which "no ball reached a rail" is not believed.
#: Seeing a ball IN the cushion band is proof of contact at any frame rate.
#: Not seeing one is not proof of anything at 4.9 fps: a ball can come off a
#: cushion between two frames and never be photographed near it. F-04 is an
#: auto-called foul, so its absence has to be established, and below this rate
#: it cannot be - the answer is "could not tell", and F-09's principle applies.
RAIL_MIN_FPS = 15.0

#: The head string sits a quarter of the table's length from the head rail.
#: That is the table's own geometry rather than a tuned number, and F-02 needs
#: it drawn as well as tested.
HEAD_STRING_FRACTION = 0.25

#: Frames without any track labelled CUE, on a clear settled table, before the
#: cue ball is called missing (M-08). Generous: the classifier drops the cue for
#: a frame or two regularly, and a wait state that flickers is worse than none.
CUE_MISSING_FRAMES = 12

#: How close two balls' centres must both be to a pocket to look like the
#: wedged pair of W-11, in ball radii from the clicked pocket centre.
WEDGE_REACH_R = 3.0

#: The classes the settled census counts. The 8 is last, so that when balls go
#: missing an ordinary ball is named before the rack-ending one is.
CENSUS_CLASSES = (CUE, STRIPE, SOLID, EIGHT)

#: How sure a pot is that the settled table proved but no pocket claim backs.
#: The count and the census agree a ball of this class went down, so it is
#: scored - but nothing the camera saw at a mouth says which pocket took it,
#: and that is under M-14's bar so the call carries its dot.
CENSUS_POT_CONFIDENCE = 0.65

#: How far off a ball's last direction of travel a pocket may lie, in degrees,
#: and still be the pocket it was heading for when its track was lost.
HEADING_DEG = 25.0

#: Where a credited pot's evidence came from.
#:   CLAIM     a track vanished at a mouth and the settled count confirmed it -
#:             used only when the settled census cannot be read.
#:   CENSUS    the settled census says this class went down, and a pocket claim
#:             says where.
#:   INFERRED  the settled census says this class went down, and the pocket is
#:             only the ball's last heading, or None.
CLAIM, CENSUS, INFERRED = "claim", "census", "inferred"

#: `busy` is the one field the vision layer contributes directly: a hand or a
#: cue was over the table while this track was missing, so its disappearance
#: has an innocent explanation that has nothing to do with a pocket.
#: `x`, `y` are where the track was last seen - so a claim for a ball that is in
#: fact still sitting there can be told from one that went down.
#: `source` is one of CLAIM, CENSUS, INFERRED above.
Pot = namedtuple("Pot", "frame track cls pocket dist closed age busy x y source",
                 defaults=(None, None, CLAIM))

#: Every track that went missing during a shot, near a pocket or not. Nothing
#: is decided from these; they are what "which pocket" is inferred from when the
#: settled table proves a ball went down and no claim was made at a mouth.
#: `was` is where the track stood the frame before, which gives its heading.
Vanish = namedtuple("Vanish", "frame track cls x y was age busy")

#: A prompt the HUD must show and a player must answer. `options` are the taps
#: that resolve it; `rule` cites the row that demanded it.
Prompt = namedtuple("Prompt", "kind rule text options data")


def _median(values):
    """Middle value - the ball count's defence against a one-frame miscount."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0


def _line_distance(x, y, a, b):
    """Distance from (x, y) to the line through points a and b."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if not length:
        return ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
    return abs(dy * (x - ax) - dx * (y - ay)) / length


class PotDetector:
    """Vanished tracks -> pot CLAIMS, using the clicked pocket positions.

    Claims, not verdicts. Two things are refused outright here: a track that
    vanished too far from any pocket to have gone down it, and a track that died
    in a crowd, because a crowd of tracks dying together is a person leaning
    over the table. Everything else becomes a claim, and how convincing it is -
    an established track, closing on the mouth - only ranks it.

    That division matters. The evidence a single track leaves behind is weak in
    both directions: an arm fragment beside a pocket looks like a pot, and a
    ball struck hard enough to outrun the tracker leaves a one-frame-old track
    that looks like an arm fragment. Neither can be settled here. What settles
    it is whether the table actually lost a ball, which only GameSession can
    see, once everything stops rolling - and even then W-10 says the ball has to
    still be in the pocket after the table settles, not merely to have crossed
    the mouth.
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
        self.far_reach = POT_FAR_MOUTH * max(1, ball_r)
        self.far = []           # fallback claims beyond the mouth, see POT_FAR_MOUTH
        self.last = {}          # track id -> (x, y, cls)
        self.prev = {}          # ... and where it was the frame before that
        self.age = Counter()    # track id -> frames seen
        self.missing = Counter()
        self.busy = {}          # ... and whether a hand was over the table then
        self.rejected = []      # near-misses, for the run summary
        self.vanished = []      # every confirmed disappearance, see Vanish

    def update(self, tracked, labels, frame, busy=False):
        """Returns the pots confirmed on this frame (usually none).

        `busy` says a hand or a cue is on the table right now. It does not
        refuse anything - a ball really does go down while the cue that struck
        it is still over the cloth - it is remembered per track and ranked on,
        so that a finger blinking out beside a pocket loses to any cleaner claim
        for the same missing ball.
        """
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
            self.busy.pop(tid, None)

        gone = []
        for tid in [t for t in self.last if t not in live]:
            self.missing[tid] += 1
            # Sticky across the whole absence, not just the frame it is
            # confirmed on: what matters is whether anything was leaning over
            # the table while the ball was unaccounted for.
            self.busy[tid] = self.busy.get(tid, False) or busy
            if self.missing[tid] < self.confirm:
                continue
            x, y, cls = self.last.pop(tid)
            age = self.age.pop(tid, 0)
            was = self.prev.pop(tid, None)
            del self.missing[tid]
            gone.append((tid, x, y, cls, age, was, self.busy.pop(tid, False)))

        crowd = len(gone) > self.mass
        self.vanished.extend(Vanish(frame, tid, cls, int(x), int(y), was, age,
                                    hand or crowd)
                             for tid, x, y, cls, age, was, hand in gone)

        pots = []
        for tid, x, y, cls, age, was, hand in gone:
            pocket, dist = nearest_pocket(x, y, self.pockets)
            if pocket is None:
                continue
            closed = (nearest_pocket(was[0], was[1], [pocket])[1] - dist
                      if was else 0.0)
            # One frame's worth of travel on top of the resting mouth - see
            # POCKET_MOUTH. Only a ball closing on the pocket earns it.
            near = dist <= self.reach + max(0.0, closed)
            if not near and dist > self.far_reach:
                continue                      # vanished in open table: occluded
            # The one thing still refused outright: a crowd of tracks dying
            # together is a person, and no ranking should have to sort that out.
            if crowd:
                self.rejected.append({
                    "frame": frame, "track": tid, "cls": cls,
                    "pocket": pocket.name, "age": age,
                    "why": f"{len(gone)} tracks vanished together"})
                continue
            (pots if near else self.far).append(
                Pot(frame, tid, cls, pocket.name, round(dist, 1),
                    round(closed, 1), age, hand, int(x), int(y)))
        return pots

    def rank(self, pot):
        """Sort key: the most convincing claim first.

        The cue ball leads - a missed scratch is the costliest verdict on the
        table. Then the claims made on a clear table, because a track that went
        out while a hand was over the cloth has an explanation that costs
        nothing to believe. Then the claims that look like a ball going into a
        pocket: an established track, closing on the mouth. A claim that fails
        all of them is still a claim; it just goes to the back of the queue.
        """
        return (pot.cls != CUE,
                pot.busy,
                pot.age < self.min_age,
                pot.closed < self.approach,
                pot.dist - pot.closed)

    def confidence(self, pot):
        """M-14: how sure this claim is, as a number the HUD can show.

        Built from the three things that actually separate a pot from a broken
        track, weighted by how much each one is worth on this footage: an
        established track counts for most, a clear table next, and closing on
        the mouth last, because a hard-struck ball can be credited on a single
        frame of evidence and still be right.

        The number is not a probability and is not presented as one. It exists
        so that M-14 can mark the shaky calls with a dot and so our own
        debugging is free, which is exactly what the client asked it for.

        A pot the settled table proved without a claim at a mouth has none of
        that evidence to weigh, and gets CENSUS_POT_CONFIDENCE instead.
        """
        if pot.source == INFERRED:
            return CENSUS_POT_CONFIDENCE
        score = 0.45
        if pot.age >= self.min_age:
            score += 0.25
        if not pot.busy:
            score += 0.20
        if pot.closed >= self.approach:
            score += 0.10
        return round(min(1.0, score), 2)


def nearest_pocket(x, y, pockets):
    """(pocket, distance) closest to a point, or (None, inf).

    Kept here as well as in detect_pocket so the rules layer does not have to
    import the calibration UI to answer "which pocket".
    """
    if not pockets:
        return None, float("inf")
    best = min(pockets, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
    return best, ((best.x - x) ** 2 + (best.y - y) ** 2) ** 0.5


def heading_pocket(x, y, was, pockets, cone=HEADING_DEG):
    """(pocket, distance) a ball was travelling toward when last seen.

    "Which pocket" for a ball the settled table proves went down but whose
    track never reached a mouth: a hard-struck ball's id dies mid-flight, and
    its last step still points down the line it was rolling on. Of the pockets
    ahead of it within `cone` degrees, the one that line passes closest to.
    (None, inf) when there is no previous position to take a heading from.
    """
    if not pockets or was is None:
        return None, float("inf")
    vx, vy = x - was[0], y - was[1]
    speed = (vx * vx + vy * vy) ** 0.5
    if not speed:
        return None, float("inf")
    limit = math.cos(math.radians(cone))
    best, best_miss, best_dist = None, float("inf"), float("inf")
    for p in pockets:
        dx, dy = p.x - x, p.y - y
        dist = (dx * dx + dy * dy) ** 0.5
        if not dist:
            return p, 0.0
        if (vx * dx + vy * dy) / (speed * dist) < limit:
            continue
        miss = abs(vx * dy - vy * dx) / speed
        if miss < best_miss:
            best, best_miss, best_dist = p, miss, dist
    return best, best_dist


class ShotSegmenter:
    """Moving / settled, from how far the matched tracks shifted per frame.

    This only finds the EPISODES of motion. Whether an episode was a shot or a
    player leaning over the table is not decided here - it is decided by
    comparing the settled tables on either side of it, in GameSession.

    An episode ends when EVERYTHING is still, because a shot is not over while
    an object ball is still rolling toward a pocket.

    Only tracks seen in BOTH frames are measured. Ids that appear or vanish say
    nothing about motion - that is occlusion, not movement.

    It also keeps a note of WHICH tracks moved and what they were, which is the
    raw material for F-05 and F-03: a shot in which nothing but the cue ball
    ever moved hit nothing, and the first non-cue track to move is the best
    guess at first contact this frame rate allows. "Best guess" is doing real
    work in that sentence - see GameSession._facts for what it is worth.
    """

    def __init__(self, ball_r, move_px=None, settle=SETTLE_FRAMES):
        self.move = move_px if move_px else max(3.0, MOTION_R * ball_r)
        self.settle = settle
        self.prev = {}
        self.moving = False
        self.still = 0
        self.movers = {}          # track id -> class, for this episode
        self.first_mover = None   # (frame, class) of the first object ball
        self.cue_started = None   # ... and the frame the cue ball first moved

    def update(self, tracked, labels=None, frame=0):
        """Returns "start", "end" or None."""
        now = {tid: (x, y) for tid, x, y, _r in tracked}
        labels = labels or {}
        shifted = 0.0
        for tid, (x, y) in now.items():
            if tid not in self.prev:
                continue
            px, py = self.prev[tid]
            step = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if step > shifted:
                shifted = step
            if step <= self.move or not self.moving:
                continue
            label = labels.get(tid)
            cls = label.cls if label is not None else UNKNOWN
            self.movers[tid] = cls
            if cls == CUE:
                if self.cue_started is None:
                    self.cue_started = frame
            elif cls in (STRIPE, SOLID, EIGHT) and self.first_mover is None:
                self.first_mover = (frame, cls)
        self.prev = now

        if not self.moving:
            if shifted > self.move:
                self.moving, self.still = True, 0
                self.movers, self.first_mover, self.cue_started = {}, None, None
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


class TableGeometry:
    """The table, read off the six clicked pockets.

    Three things the rules need and the pockets already contain:

      * the rail lines, so F-04 can ask whether any ball reached a cushion and
        F-08 can mark a ball frozen against one;
      * the long axis, so the head and foot ends can be told apart;
      * the head string, which F-02 needs both to TEST a ball-in-hand placement
        and to DRAW, because the client's note on F-02 asks for the zone on the
        HUD, not just a verdict behind it.

    Which end is the foot is LEARNED rather than assumed: the rack is set at the
    foot spot, and M-09 already detects racks, so the first triangle seen tells
    us which end of the table it sits at and the head string goes at the other.
    Before a rack has been seen the head end is a guess, and `head_known` says
    so - which is what stops the HUD from drawing a confident line in the wrong
    half of the table.
    """

    def __init__(self, pockets, ball_r):
        self.ball_r = max(1, ball_r)
        pockets = list(pockets or [])
        self.pockets = pockets
        corners = [p for p in pockets if p.name in ("TL", "TR", "BL", "BR")]
        pool = corners if len(corners) == 4 else pockets
        self.ok = len(pool) >= 4
        if not self.ok:
            self.x0 = self.y0 = self.x1 = self.y1 = 0
            self.long_axis = "x"
            self.head_at_low = True
            self.head_known = False
            return
        self.x0 = min(p.x for p in pool)
        self.x1 = max(p.x for p in pool)
        self.y0 = min(p.y for p in pool)
        self.y1 = max(p.y for p in pool)
        self.long_axis = "x" if (self.x1 - self.x0) >= (self.y1 - self.y0) else "y"
        self.head_at_low = True     # provisional until a rack is seen
        self.head_known = False
        # The rails as the four lines through the corner pockets, in order round
        # the table - not the bounding box. A camera is never square to the
        # cloth: on this footage the right-hand corners sit 52 px apart in x, so
        # a box puts the right rail that far out at one end, and a ball touching
        # the real cushion there would read as never having reached it.
        by = {p.name: (p.x, p.y) for p in corners}
        self.rails = ([(by["TL"], by["TR"]), (by["TR"], by["BR"]),
                       (by["BR"], by["BL"]), (by["BL"], by["TL"])]
                      if len(corners) == 4 else [])

    @property
    def length(self):
        return (self.x1 - self.x0) if self.long_axis == "x" else (self.y1 - self.y0)

    def learn_foot(self, positions):
        """A rack has been seen: whichever end it sits at is the FOOT.

        After this the head string is the table's real head string rather than a
        coin toss, and F-02 can be enforced instead of merely drawn.
        """
        if not positions or not self.ok:
            return False
        axis = 0 if self.long_axis == "x" else 1
        low, high = (self.x0, self.x1) if axis == 0 else (self.y0, self.y1)
        mid = (low + high) / 2.0
        centre = sum(p[axis] for p in positions) / float(len(positions))
        self.head_at_low = centre > mid      # rack at the high end -> head is low
        self.head_known = True
        return True

    @property
    def head_string(self):
        """The coordinate of the head string along the long axis, or None."""
        if not self.ok:
            return None
        quarter = self.length * HEAD_STRING_FRACTION
        low, high = ((self.x0, self.x1) if self.long_axis == "x"
                     else (self.y0, self.y1))
        return low + quarter if self.head_at_low else high - quarter

    def head_zone(self):
        """(x0, y0, x1, y1) of the kitchen, for the HUD to shade (F-02)."""
        line = self.head_string
        if line is None:
            return None
        if self.long_axis == "x":
            return ((self.x0, self.y0, int(line), self.y1) if self.head_at_low
                    else (int(line), self.y0, self.x1, self.y1))
        return ((self.x0, self.y0, self.x1, int(line)) if self.head_at_low
                else (self.x0, int(line), self.x1, self.y1))

    def behind_head_string(self, x, y):
        """F-02: is this a legal spot for a restricted ball in hand?"""
        zone = self.head_zone()
        if zone is None:
            return None
        x0, y0, x1, y1 = zone
        return x0 <= x <= x1 and y0 <= y <= y1

    def on_rail(self, x, y):
        """Is this ball's centre inside the cushion band? (F-04, F-08)"""
        if not self.ok:
            return False
        band = RAIL_BAND_R * self.ball_r
        if self.rails:
            return any(_line_distance(x, y, a, b) <= band for a, b in self.rails)
        return (x - self.x0 <= band or self.x1 - x <= band
                or y - self.y0 <= band or self.y1 - y <= band)


class GameSession:
    """What main.py holds: everything above, wired to the rules in games.py.

    Feed it every frame's tracks and labels; read the scoreboard off `game`, the
    things needing a tap off `prompts`, and the buttons off `commands()`.
    """

    def __init__(self, pockets, ball_r, fps=25.0,
                 players=("Player 1", "Player 2"), mouth=POCKET_MOUTH,
                 settle=SETTLE_FRAMES, discipline=games.EIGHT_BALL,
                 mode=games.CASUAL, levels=(None, None), defence_marking=False,
                 pocket_marking=None, first_rack_confirmed=True,
                 sample_fps=None, fmt="open", break_assigns=True,
                 innings_rule=games.SCORESHEET_INNINGS, lag_winner=0):
        self.pots = PotDetector(pockets, ball_r, mouth=mouth)
        self.shots = ShotSegmenter(ball_r, settle=settle)
        self.table = TableGeometry(pockets, ball_r)
        self.game = make_game(discipline, players=players, mode=mode,
                              levels=levels, defence_marking=defence_marking,
                              pocket_marking=pocket_marking, fps=fps, fmt=fmt,
                              break_assigns=break_assigns,
                              innings_rule=innings_rule, lag_winner=lag_winner)
        self.discipline = discipline
        self.fps = fps if fps and fps > 0 else 25.0
        # `fps` is the video's own rate and drives every clock; `sample_fps` is
        # how often frames actually ARRIVE (fps / stride), which is what decides
        # how much can be seen between two of them.
        self.sample_fps = sample_fps if sample_fps and sample_fps > 0 else self.fps
        self.ball_r = max(1, ball_r)
        self.moved_r = MOVED_R * self.ball_r
        self.candidates = []       # pots claimed since this episode started
        self.far_claims = []       # ... and vanishings too far out to claim, as a fallback
        self.vanished = []         # ... and every track lost this episode (Vanish)
        # Both windows hold (clear, ...) pairs: `clear` is False on a frame
        # with a hand or a cue over the table, and every reading taken from
        # these windows prefers the frames where it is True. A hand does not
        # merely add blobs, it SUBTRACTS them - the balls underneath it go with
        # it when the intrusion is erased whole - so a busy frame is wrong in
        # both directions and is worth nothing to either the count or positions.
        self.recent = deque(maxlen=settle)     # ball count over the last frames
        self.window = deque(maxlen=settle)     # ... and where they were
        self.census = deque(maxlen=settle)     # ... and what they were
        self.settled_census = None   # the classes on the table at the last settling
        self.before_census = None    # ... as they stood before the shot being judged
        self.last_hold = 0           # frames the shot being judged waited to clear
        self.resting = None        # the table as it stood at the last settling
        # The ball count is only trustworthy at ONE moment: the instant a shot
        # settles. The balls have stopped and the player is still standing back
        # watching them. A count taken any later is inflated by whoever has
        # leaned over the table to line up the next shot - measured on this
        # footage at twenty-two balls on a sixteen-ball table.
        self.settled_balls = None
        self.before = None
        self.balls = 0
        self.frame = 0
        self.has_pockets = bool(pockets)
        self.all_pots = []
        self.ignored = []          # claims that did not survive a gate
        self.episodes = []         # every settling of the table, shot or not
        self.won_at = None         # frame a rack ended
        self.racked_for = 0        # consecutive still frames showing a triangle
        self.intruding = False     # a hand or a cue is over the table NOW
        self.since_clear = INTRUSION_CLEAR   # frames since the last one
        self.pending = None        # a settling held until the table clears
        self.held = 0              # ... and how many frames it has been held
        self.busy_frames = 0       # how much of the video had a hand in it
        self.waits = []            # settlings that had to wait, for the summary

        # ---- the client's suite, beyond the original prototype -------------
        self.prompts = []          # things needing a tap, newest last
        self.settled_at = None     # M-02: the frame everything came to rest
        self.last_credited = []    # M-04 / W-10: revocable until the next shot
        self.replace_at = None     # M-02 / M-07: where a ball has to go back
        self.cue_missing = 0       # M-08
        self.waiting_for_cue = False
        self.frozen = []           # F-08: balls against a cushion, marked only
        self.rail_contact = None   # F-04, accumulated across the episode
        self.first_rack_confirmed = bool(first_rack_confirmed)  # M-09
        self.reviews = []          # F-07, F-09, M-13: flagged for replay
        self.shot_confidence = None    # M-14: the number behind the last call
        self.phantoms = 0
        self.replaced = 0
        self.last_facts = None     # the last judged shot, for re-judging it
        self.label_conf = {}       # track id -> identity confidence (M-06)
        self.last_seen = {}        # class -> last frame a track carried it
        self.cue_pos = None        # where the cue ball is right now (F-02)
        self.rail_tracks = set()   # object balls that reached a rail (M-10)
        self.wedge_frames = 0      # W-11: how long the pair has sat there
        self.hanging = []          # M-02: still-table claims awaiting a count
        self.wedge_dismissed = None
        self.eight_at_rest = False # W-07: was the 8 on the table last settle?
        self.cue_at_rest = False

    # ---- what the HUD reads ------------------------------------------------

    @property
    def busy(self):
        """Is there something on the table that makes this frame untrustworthy?

        True for a frame or two after the intrusion itself is gone - see
        INTRUSION_CLEAR. This is the whole temporal half of the hand problem: a
        hand appears only while someone is shooting and then leaves, while a
        ball stays put for minutes, so the system does not have to filter a hand
        out of a frame perfectly. It only has to know that a hand is there right
        now, and decline to draw conclusions from that frame.

        M-05 is what this is FOR, and M-05 is also explicit about how it should
        LOOK: occlusion is normal play and gets a quiet indicator, never an
        alarming banner. Nothing here escalates, and main.py draws it as a dot.
        """
        return self.intruding or self.since_clear < INTRUSION_CLEAR

    @property
    def prompt(self):
        """The one thing the HUD should be asking about, or None."""
        return self.prompts[-1] if self.prompts else None

    def ask(self, kind, rule, text, options, **data):
        """Raise a prompt, unless the same kind is already on screen.

        Deduplicated by kind because the situations that raise them persist for
        many frames - a wedged pair sits in the jaw until someone deals with it
        - and a queue of forty identical questions is not a question.
        """
        if any(p.kind == kind for p in self.prompts):
            return None
        p = Prompt(kind, rule, text, list(options), dict(data))
        self.prompts.append(p)
        # Quiet: the question has its own panel, and the status line keeps the
        # verdict it is asking about.
        self.game.note("prompt", text, rule=rule, quiet=True)
        return p

    def answer(self, kind, choice=None):
        """Resolve a prompt by tap. Returns True when one was waiting."""
        found = [p for p in self.prompts if p.kind == kind]
        if not found:
            return False
        self.prompts = [p for p in self.prompts if p.kind != kind]
        self._resolve(found[-1], choice)
        return True

    def _resolve(self, prompt, choice):
        game = self.game
        if prompt.kind == "groups" and choice in (STRIPE, SOLID):
            # M-06: the answer assigns groups for whoever was shooting.
            game.confirm_groups(prompt.data.get("shooter", game.turn), choice)
        elif prompt.kind == "wedged":
            if choice == "pocketed":
                self._credit_wedged(prompt.data.get("classes", []))
            else:
                self.wedge_dismissed = prompt.data.get("pocket")
                game.note("note", "WEDGED PAIR - PLAY CONTINUES", rule="W-11",
                          actor="player")
        elif prompt.kind == "illegal_break":
            game.resolve_illegal_break(choice)
            if choice == "re-rack":
                self.resting, self.settled_balls = None, None
        elif prompt.kind == "rack":
            self.first_rack_confirmed = True
            game.note("rack", "RACK CONFIRMED", rule="M-09", actor="player")
        elif prompt.kind == "replace":
            self.replace_at = None
            game.note("note", "BALL REPLACED", rule="M-07", actor="player")
        else:
            game.note("note", f"{prompt.kind.upper()} RESOLVED",
                      rule=prompt.rule, actor="player")

    def _credit_wedged(self, classes):
        """W-11: deemed pocketed, and scored - unless that would end the game.

        The wedged pair belongs to the shot that left it there, so that shot is
        re-judged with the two balls added. If the re-judge ENDS the rack, the
        rule's exception applies: judge the shot again as it originally was,
        drop the balls without the rack-ending condition, and play resumes.
        """
        game = self.game
        if not classes:
            game.note("note", "WEDGED PAIR DROPPED - BALLS NOT IDENTIFIED",
                      rule="W-11", actor="player")
            return
        original, was_over = self.last_facts, game.over
        if self._rejudge(add=classes, why="WEDGED PAIR DROPPED"):
            if game.over and not was_over:
                self.last_facts = original
                self._rejudge(why="WEDGED PAIR WOULD END THE RACK")
                game.drop_without_ending(classes)
            return
        game.drop_without_ending(classes)

    def _rejudge(self, add=(), remove=(), why="SHOT RE-JUDGED"):
        """Walk the last shot back and judge it again with different pots.

        M-02, M-04 and W-11 all change what went down on a shot that has
        ALREADY been scored - a hanging ball dropped late, a ball rattled back
        out, a wedged pair dropped in by hand - and all of them have to land on
        the shot that caused them, with the shooter who played it, not on
        whoever happens to be at the table now. The snapshot M-03 took before
        that shot is exactly the state to re-judge from.

        Returns False when there is no shot to walk back (the next stroke has
        begun, or the rack is locked), and the caller does something that does
        not touch the verdict instead.
        """
        game = self.game
        if self.last_facts is None or not game.record.can_undo:
            return False
        if not game.undo(by="system", why=why):
            return False
        f = games.merge(self.last_facts, add=add, remove=remove)
        game.shot_started(self.settled_at or self.frame)
        game.shot_ended(f, self.frame)
        self.last_facts = f
        return True

    # ---- the HUD's whole command surface -----------------------------------

    def commands(self):
        """[(id, label, enabled)] - what the HUD may offer at this moment.

        Availability is a RULE, not a style choice, and keeping the list here
        rather than in the drawing code is what stops the two from drifting:

          H-02  MARK DEFENCE exists only between a completed shot and the next
                one, and never persistently.
          H-05  END TIME OUT and CANCEL exist only while one is running.
          M-03  the back button disappears entirely once the score is final, and
                REOPEN takes its place.
          F-10  CALL FOUL greys out as soon as the next stroke begins.
        """
        game = self.game
        out = []
        if game.defence_open:
            out.append(("defence", "MARK DEFENCE", True))
        if game.timeouts.active is None:
            out.append(("timeout", "TIME OUT", not game.over))
        else:
            out.append(("end_timeout", "END TIME OUT", True))
            out.append(("cancel_timeout", "CANCEL", True))
        if game.record.can_undo:
            out.append(("undo", "UNDO", True))
        if not game.record.final:
            out.append(("confirm", "CONFIRM FINAL", True))
        else:
            out.append(("reopen", "REOPEN", True))
        out.append(("foul", "CALL FOUL", game.record.foul_window_open(game.shot)))
        # Manual 16d: only while an unmarked 8 is still standing as a win.
        if getattr(game, "unmarked_claim", None) is not None:
            out.append(("call_loss", "CALL LOSS: UNMARKED 8", True))
        # Manual 3.14: frozen balls are declared, never detected into a rule.
        out.append(("frozen", "DECLARE FROZEN", not game.over))
        # Manual 3.4 note: push-outs only in Masters, only after the break.
        if (self.discipline == games.NINE_BALL and game.fmt == "masters"
                and game.shot <= 1 and not game.over):
            out.append(("push_out", "PUSH-OUT", True))
        out.append(("stalemate", "STALEMATE", not game.over))
        out.append(("review", "REVIEW", True))
        return out

    def command(self, name, **kw):
        """Run one HUD tap. Returns True when something happened."""
        game = self.game
        game.frame = self.frame
        if name == "defence":
            return game.mark_defence()
        if name == "timeout":
            return game.start_timeout()
        if name == "end_timeout":
            return game.end_timeout()
        if name == "cancel_timeout":
            return game.cancel_timeout()
        if name == "undo":
            return game.undo()
        if name == "confirm":
            return game.confirm_final()
        if name == "reopen":
            return game.reopen(kw.get("who", "player"))
        if name == "foul":
            return game.mark_foul(kw.get("why", "TOUCHED BALL"))
        if name == "stalemate":
            return game.stalemate()
        if name == "call_pocket":
            return game.call_pocket(kw.get("pocket"))
        if name == "call_loss":
            return bool(getattr(game, "call_unmarked_loss", lambda: False)())
        if name == "frozen":
            return game.declare_frozen(kw.get("ball", "BALL"))
        if name == "swap_groups":
            return bool(getattr(game, "swap_groups", lambda: False)())
        if name == "push_out":
            return bool(getattr(game, "push_out", lambda: False)())
        if name == "review":
            # M-13: BPS does not overrule. The button pulls the footage and says
            # so in the log; the players resolve it themselves with the back
            # button, which is the whole positioning - evidence, not referee.
            self.reviews.append({"frame": self.frame, "clock": self.clock,
                                 "why": "player asked for a review"})
            game.note("review", "REVIEW - LAST 10 SECONDS", rule="M-13",
                      actor="player")
            return True
        if name == "answer":
            return self.answer(kw.get("kind"), kw.get("choice"))
        return False

    # ---- per frame ---------------------------------------------------------

    def update(self, tracked, labels, frame, intruding=False):
        """Returns the pots credited on this frame, for logging/overlay.

        `intruding` comes from the detector: a region was erased from this
        frame's mask that was solid enough to have hidden a ball or grown a
        phantom one - an arm, a hand, a cue being held over the cloth.
        """
        self.frame, self.balls = frame, len(tracked)
        self.game.frame = frame
        self.intruding = bool(intruding)
        self.since_clear = 0 if intruding else self.since_clear + 1
        if self.intruding:
            self.busy_frames += 1
        clear = not self.busy
        self.recent.append((clear, len(tracked)))
        self.window.append((clear, [(x, y) for _t, x, y, _r in tracked]))
        self.census.append((clear, Counter(lb.cls for lb in (labels or {}).values())))
        positions = self.window[-1][1]
        self._observe(tracked, labels, frame)

        # The per-frame observations that are only meaningful on a still, clear
        # table. All three are marks and wait states - none of them calls a foul.
        if clear and not self.shots.moving:
            self._frozen(tracked, labels)
            self._cue_ball_missing(labels)
            self._wedged(tracked, labels)

        # A fresh triangle sitting still means the last rack is over and the
        # next is about to be broken, whatever the rules engine thought was
        # happening. Checked per frame, before anything else: a rack that goes
        # unnoticed scores a whole new game into a rack that is already empty.
        # Hands are NOT excluded here, unlike everywhere a count is taken:
        # racking is done by hand and watched over by both players, so a rack is
        # never seen on a clear table. RACK_FRAMES carries that weight instead.
        if self.shots.moving or not self._racked(positions):
            self.racked_for = 0
        else:
            self.racked_for += 1
            if self.racked_for >= RACK_FRAMES and not self.table.head_known:
                # The opening rack is where F-02's head string is learned from.
                self.table.learn_foot(positions)
            if (self.racked_for >= RACK_FRAMES and not self.game.struck
                    and not self.first_rack_confirmed):
                self.ask("rack", "M-09", "NEW RACK DETECTED - CONFIRM",
                         ["yes"], balls=len(positions))
            if self.racked_for >= RACK_FRAMES and self.game.struck:
                if self._new_rack(frame, positions):
                    return []

        event = self.shots.update(tracked, labels, frame)
        credited = []
        # Motion has resumed with a settling still held: the table never did
        # clear, so judge it now on what was seen rather than lose it into the
        # next shot. This is the only way a held settling can be overtaken.
        if event == "start" and self.pending is not None:
            credited = self._settle(self.pending)
        if event == "start":
            self.candidates = []
            self.far_claims = []
            self.vanished = []
            self.rail_contact = False
            self.rail_tracks = set()
            self.wedge_dismissed = None
            self.replace_at = None
            # A still-table claim never confirmed before play resumed is dropped:
            # the table it would be counted against no longer exists.
            self.ignored.extend(self.hanging)
            self.hanging = []
            # NOT last_credited or settled_at: motion is not a shot until the
            # settle says so, and a player walking past the table must neither
            # close M-04's failsafe nor restart M-02's clock. Both are replaced
            # when the next SHOT is judged, in _settle.

        if self.shots.moving:
            self._watch_rails(tracked)

        claims = self.pots.update(tracked, labels, frame, busy=self.busy)
        far, self.pots.far = self.pots.far, []
        vanished, self.pots.vanished = self.pots.vanished, []
        if self.shots.moving or event == "end":
            self.far_claims.extend(far)
            self.vanished.extend(vanished)
        if claims and (self.shots.moving or event == "end"):
            self.candidates.extend(claims)
        elif claims:
            # A ball cannot go down while nothing is moving - unless it was
            # hanging in the jaw, which is exactly what M-02 is about. The
            # hanging test decides whether this is a late pot worth scoring or a
            # ball that has to be put back - once the table is clear enough to
            # count, not on the frame the claim arrives.
            self._hold_hanging(claims)

        # The settling is where every verdict is taken, and it is taken by
        # counting balls - so it is exactly the thing that must not happen with
        # a hand in frame. Hold it until the table clears. Waiting costs a few
        # frames of latency in the overlay; not waiting costs a wrong verdict,
        # because balls under an arm read as balls that MOVED.
        if event == "end":
            if self.busy:
                self.pending, self.held = frame, 0
            else:
                credited = self._settle(frame)
        elif self.pending is not None:
            self.held += 1
            if not self.busy or self.held >= INTRUSION_WAIT:
                if self.held >= INTRUSION_WAIT:
                    self.waits.append({"frame": self.pending, "held": self.held,
                                       "why": "gave up waiting for a clear table"})
                credited = self._settle(self.pending)

        if self.hanging and not self.shots.moving and self.pending is None:
            self._confirm_hanging()

        # M-04 and W-10, on the same detector and the same threshold, which is
        # the client's own observation: a ball that is back on the table was
        # never potted. Only on a table that has been clear for a whole settle
        # window - a player straightening up uncovers balls, and those are not
        # balls coming back out of a pocket.
        if (not self.shots.moving and self.since_clear >= SETTLE_FRAMES
                and self.last_credited):
            self._reconcile_phantoms(positions, frame)

        # Before anything has happened, the racked table standing still is the
        # reference the first episode will be judged against. Taken from a clear
        # frame: the first thing in this footage is a pair of hands building the
        # rack, and a reference measured through them is wrong for every episode
        # later compared against it.
        if self.resting is None and not self.shots.moving and not self.busy:
            self._rest()
        return credited

    # ---- per-frame bookkeeping ---------------------------------------------

    #: How a vote's state maps onto M-14's confidence scale. `voted` means the
    #: identity vote has settled over several frames; `pending` means it has
    #: not, and a game-deciding call should not rest on it.
    LABEL_CONFIDENCE = {"voted": 0.9, "pending": 0.5}

    def _observe(self, tracked, labels, frame):
        """What each ball is, how sure, and where the cue ball sits."""
        self.cue_pos = None
        for tid, x, y, _r in tracked:
            label = (labels or {}).get(tid)
            if label is None:
                continue
            self.label_conf[tid] = self.LABEL_CONFIDENCE.get(label.confidence,
                                                             0.0)
            self.last_seen[label.cls] = frame
            if label.cls == CUE:
                self.cue_pos = (x, y)

    @property
    def placement(self):
        """F-02: is a restricted ball in hand currently placed legally?

        True / False while a head-string ball in hand is owed and the cue ball
        can be seen; None whenever there is nothing to check - no restriction
        owed, no cue ball in view, or no rack seen yet to say which end is the
        head. None draws the zone without a verdict: it is only enforced once
        it is KNOWN.
        """
        if self.game.ball_in_hand != games.BEHIND_HEAD_STRING:
            return None
        if self.cue_pos is None or not self.table.head_known:
            return None
        return bool(self.table.behind_head_string(*self.cue_pos))

    # ---- F-04 and F-08: the cushions ---------------------------------------

    def _watch_rails(self, tracked):
        """F-04. Did any ball reach a cushion during this shot?

        Cheap, as the client's note says: a ball whose centre enters the band
        along a rail has touched it. Sticky for the whole episode, because the
        question is whether it happened at all, not whether it is happening now.

        With no table geometry the answer is None rather than False, and None
        routes to F-09 - no foul - instead of to a foul nobody can defend.
        """
        if not self.table.ok:
            self.rail_contact = None
            return
        # Only balls that are part of the shot. A ball already frozen against a
        # cushion sits in the band all game, and counting it would make every
        # shot a legal one.
        for tid, x, y, _r in tracked:
            if tid in self.shots.movers and self.table.on_rail(x, y):
                self.rail_contact = True
                if self.shots.movers[tid] != CUE:
                    self.rail_tracks.add(tid)

    def _frozen(self, tracked, labels):
        """F-08. Mark balls against a cushion and hand them to the players.

        Deferred by client decision, and deliberately so: the frozen-ball rail
        requirement is the most complex foul in the book, and a camera that
        half-implements it produces confident wrong calls. So this marks and
        never judges - the HUD shows which balls are frozen, and the players do
        the rest. Do not be tempted to promote it.
        """
        if not self.table.ok:
            self.frozen = []
            return
        self.frozen = [tid for tid, x, y, _r in tracked
                       if self.table.on_rail(x, y)
                       and (labels or {}).get(tid) is not None]

    # ---- M-08: the cue ball ------------------------------------------------

    def _cue_ball_missing(self, labels):
        """M-08. No foul, just a wait, including for a substituted ball.

        On coin-op tables a measle ball regularly gets swallowed and swapped for
        whatever is to hand, and in practice people pick the cue ball up all the
        time. Neither is a rules event, so this produces a wait state and
        nothing else.
        """
        if not labels:
            return
        if any(lb.cls == CUE for lb in labels.values()):
            if self.waiting_for_cue:
                self.waiting_for_cue = False
                self.game.note("note", "CUE BALL BACK", rule="M-08")
            self.cue_missing = 0
            return
        self.cue_missing += 1
        if self.cue_missing == CUE_MISSING_FRAMES and not self.waiting_for_cue:
            self.waiting_for_cue = True
            self.game.note("note", "WAITING FOR THE CUE BALL", rule="M-08")

    # ---- W-11: the wedged pair ---------------------------------------------

    def _wedged(self, tracked, labels):
        """W-11. Two balls leaning off the slate in one jaw: prompt, never call.

        A camera cannot resolve a wedged pair reliably - from overhead two balls
        in a pocket mouth look like two balls in a pocket mouth, which is also
        what a ball about to drop and a ball that rattled out look like. So the
        system says what it thinks it sees and the players decide, which is the
        client's instruction and the only honest option.
        """
        if not self.has_pockets or self.game.over or not self.game.struck:
            self.wedge_frames = 0
            return
        reach = (WEDGE_REACH_R * self.ball_r) ** 2
        touch = (TOUCHING_R * self.ball_r) ** 2
        for pocket in self.pots.pockets:
            if pocket.name == self.wedge_dismissed:
                continue
            near = [(tid, x, y) for tid, x, y, _r in tracked
                    if (x - pocket.x) ** 2 + (y - pocket.y) ** 2 <= reach]
            # Two balls in the jaw TOUCHING each other. Two balls merely resting
            # near the same corner is ordinary play and asks nothing.
            pairs = [(a, b) for i, a in enumerate(near) for b in near[i + 1:]
                     if (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 <= touch]
            if not pairs:
                continue
            # ... and staying there, so a ball rolling past a resting one on its
            # way into the pocket is not a question.
            self.wedge_frames += 1
            if self.wedge_frames < RACK_FRAMES:
                return
            (ta, _xa, _ya), (tb, _xb, _yb) = pairs[0]
            classes = [(labels or {})[t].cls for t in (ta, tb)
                       if (labels or {}).get(t) is not None]
            self.ask("wedged", "W-11",
                     f"TWO BALLS WEDGED IN {pocket.name} - DROP THEM IN?",
                     ["pocketed", "play-on"], classes=classes,
                     pocket=pocket.name)
            return
        self.wedge_frames = 0

    # ---- M-02: the hanging ball --------------------------------------------

    def _hold_hanging(self, claims):
        """Collect claims made on a still table, to be judged once it is clear.

        Two filters first, both about what a hanging ball IS. It sat in the jaw
        for a while, so its track is established; and nothing was leaning over
        the table while it disappeared. A player standing over a pocket fails
        both - and on this footage that is most of what these claims are: an
        earlier version without these filters re-judged one shot four times in
        ninety frames while somebody leant over the bottom-middle pocket.
        """
        if self.settled_at is None:
            self.ignored.extend(claims)
            return
        for c in claims:
            (self.hanging if c.age >= POT_MIN_AGE and not c.busy
             else self.ignored).append(c)

    def _confirm_hanging(self):
        """Believe a still-table pot only when the CLEAR table agrees it is gone.

        Waits for a full settle window with nothing over the cloth, then counts.
        No drop means nothing went down; a drop credits at most that many
        claims, best-ranked first - the same bargain _settle_up makes.
        """
        if self.since_clear < SETTLE_FRAMES:
            return
        lost = (self.settled_balls or 0) - _median(self._clear_counts())
        held, self.hanging = sorted(self.hanging, key=self.pots.rank), []
        if lost < 1:
            self.ignored.extend(held)
            return
        self.ignored.extend(held[lost:])
        self._hanging(held[:lost])

    def _last_spot(self, claim):
        """Where a ball that went down was sitting: the resting ball nearest the
        pocket it went into. The detector forgets a track once it is gone, but
        the last settled table still has it."""
        pocket = next((p for p in self.pots.pockets if p.name == claim.pocket),
                      None)
        if pocket is None or not self.resting:
            return None
        spot = min(self.resting, key=lambda q: (q[0] - pocket.x) ** 2
                                               + (q[1] - pocket.y) ** 2)
        if ((spot[0] - pocket.x) ** 2 + (spot[1] - pocket.y) ** 2
                > (POCKET_MOUTH * self.ball_r) ** 2):
            return None
        return int(spot[0]), int(spot[1])

    def _hanging(self, claims):
        """M-02. A ball falls on a still table: was it hanging, or just sitting?

        The client took the five seconds from CSI/BCA, which is the only
        rulebook that gives a countable number, and was specific about where the
        count starts: the moment ALL balls stop moving, regardless of whether
        the shooter is still at the table. A ball that drops inside that window
        counts. A ball that has been sitting still for longer and then falls by
        itself is NOT scored - it has to be replaced as closely as possible to
        where it sat, and M-07's mechanism is what shows the players where.
        """
        # The same gate every other pot passes: a claim is only as good as the
        # settled count. Most claims on a still table are a player standing
        # over a pocket, and those leave the count exactly where it was - which
        # is why the detector ignored them before M-02 existed, and still does.
        now = _median(self._clear_counts())
        if (self.settled_at is None or self.busy or self.settled_balls is None
                or now >= self.settled_balls):
            self.ignored.extend(claims)
            return
        # ... and the ball has to have been SITTING at that pocket when the
        # table settled, and be gone from there now. A count that dips is not
        # enough on its own: a ball resting near the top-middle pocket blinks
        # out of detection all through this footage, and at 4:41 the dip was
        # scored as a hanging ball dropping - re-judging the shot, handing the
        # turn back and adding a point for a ball still on the cloth.
        spots = [self._last_spot(c) for c in claims]
        if any(s is None or self._seen_now(s) for s in spots):
            self.ignored.extend(claims)
            return
        # Timed from when the ball DROPPED, not from when the count confirmed
        # it - the wait for a clear table must not push a hanging ball over.
        elapsed = (min(c.frame for c in claims) - self.settled_at) / self.fps
        if elapsed <= rules.HANGING_SECONDS:
            # Inside the window: a hanging ball that dropped. It belongs to the
            # shot that just settled, and to the player who played it, so that
            # shot is re-judged with the ball added rather than a new shot being
            # invented for whoever is at the table now.
            if self._rejudge(add=[c.cls for c in claims],
                             why="HANGING BALL DROPPED"):
                self.all_pots.extend(claims)
                self.last_credited.extend(claims)
                self.settled_balls = max(0, (self.settled_balls or 0)
                                         - len(claims))
                self.game.note("pot", "HANGING BALL DROPPED - SCORED",
                               rule="M-02", confidence=min(
                                   self.pots.confidence(c) for c in claims))
            else:
                self.ignored.extend(claims)
            return
        # Outside it: not scored, and the HUD has to say where it goes back.
        self.ignored.extend(claims)
        self.replace_at = self._last_spot(claims[0])
        self.game.note("note", f"BALL FELL AFTER {elapsed:.0f}s - NOT SCORED",
                       rule="M-02")
        self.ask("replace", "M-02", "REPLACE THE BALL WHERE IT SAT",
                 ["done"], at=self.replace_at)

    # ---- M-04 and W-10: the ball that came back ----------------------------

    def _reconcile_phantoms(self, positions, frame):
        """M-04 and W-10, which the client points out are one detector.

        A ball is only pocketed if it is still in the pocket once the table has
        settled - not merely if it crossed the mouth. So a credited pot whose
        ball turns up back on the cloth is taken back.

        Two stages, and the difference between them is only how loud it is.
        Inside two seconds the correction is SILENT, because balls leave the
        table and come back a beat later constantly and a visible correction
        every time would be worse than the error it fixes. After that, and until
        the next shot begins, it still happens but it is announced - a ball that
        flew off the table and was replaced late is reconciled either way, which
        is what guarantees the record ends up correct.
        """
        if self.resting is None:
            return
        reach = (2.0 * self.ball_r) ** 2
        # A ball that is here now and was not here at the last settling.
        fresh = [p for p in positions
                 if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= reach
                            for q in self.resting)]
        if not fresh:
            return
        mouth = (POCKET_MOUTH * self.ball_r) ** 2
        out_of = {pk.name for pk in self.pots.pockets
                  if any((p[0] - pk.x) ** 2 + (p[1] - pk.y) ** 2 <= mouth
                         for p in fresh)}
        if not out_of:
            return
        # A pot credited to the pocket the ball came back out of first. A pot
        # the settled census scored without a claim at a mouth has no pocket
        # anyone saw, so it may have come back out of any of them.
        pot = (next((p for p in self.last_credited
                     if p.source != INFERRED and p.pocket in out_of), None)
               or next((p for p in self.last_credited
                        if p.source == INFERRED), None))
        if pot is None:
            return
        silent = (frame - pot.frame) / self.fps <= rules.PHANTOM_SECONDS
        self.last_credited.remove(pot)
        if pot in self.all_pots:
            self.all_pots.remove(pot)
        self.phantoms += 1
        self.settled_balls = (self.settled_balls or 0) + 1
        # The pot changed the verdict, so taking it back means judging the
        # shot again without it - same shooter, same everything else.
        self._rejudge(remove=[pot.cls],
                      why=f"{pot.cls.upper()} NOT POTTED")
        if not silent:
            self.game.note("note", f"{pot.cls.upper()} CAME BACK OUT OF "
                                   f"{pot.pocket or 'A POCKET'} - NOT POTTED",
                           rule="W-10")
        self.resting = list(positions)

    # ---- M-07: the ball that was moved -------------------------------------

    def _replaced(self, moved, frame):
        """M-07. Never a foul. Logged, flagged for replay, and play continues.

        The product upside the client spotted is why this is worth doing
        properly rather than merely tolerating: because the system knows where
        the ball WAS, the HUD can show players where to put it back. That is the
        same mechanism M-02 needs for a replaced hanging ball, so one feature
        pays for two rules.
        """
        self.replaced += 1
        self.replace_at = moved[0] if moved else None
        self.game.note("note", "BALL MOVED BY HAND - PUT IT BACK", rule="M-07",
                       at=list(self.replace_at) if self.replace_at else None)
        self.reviews.append({"frame": frame, "clock": self.clock,
                             "why": "unexplained position change (M-07)"})

    # ---- M-06: the groups --------------------------------------------------

    def _confirm_groups(self, shooter, credited, was_open):
        """M-06. Ask once, with two thumbnails, and never guess silently.

        Colour accuracy is around 80% today and one confirmation tap beats a
        wrong game. The prompt only goes up when the table WAS open and this
        shot assigned groups off a shaky label - once groups are locked the
        same uncertainty is just a label, not a game-deciding call.

        By the time this runs the rules have already made the assignment. That
        is fine as long as it is visibly provisional until the tap, which is
        what `groups_confirmed` tells the HUD.
        """
        if self.discipline != games.EIGHT_BALL or not was_open:
            return False
        if self.game.open_table:
            return False                 # nothing was assigned: nothing to ask
        objects = [p for p in credited if p.cls in (STRIPE, SOLID)]
        if not objects:
            return False
        # Colour confidence, not pot confidence: the question M-06 asks is
        # whether we know WHAT went down, which is the identity vote's business.
        confidence = min(self.label_conf.get(p.track, 0.0) for p in objects)
        if confidence >= rules.CONTACT_CONFIDENCE:
            return False
        self.game.groups_confirmed = False
        self.ask("groups", "M-06", "CONFIRM GROUPS", [STRIPE, SOLID],
                 shooter=shooter, confidence=confidence,
                 at=[self.pots.last.get(p.track, (0, 0, ""))[:2]
                     for p in objects])
        return True

    # ---- M-09: the rack ----------------------------------------------------

    def _new_rack(self, frame, positions):
        """A fresh triangle. Returns True when the rack was actually started.

        A recording of a pool table is a recording of a match, not of one rack.
        The balls potted so far belong to the game that just ended, and carrying
        them forward means the next game is scored into a rack already full.

        The FIRST rack ever seen on a table is confirmed by a tap (M-09) rather
        than taken on trust: auto-start is the demo-critical wow moment and
        getting it wrong on the very first rack is the worst possible first
        impression. After that the table is known and it is automatic.
        """
        if not self.first_rack_confirmed:
            self.ask("rack", "M-09", "NEW RACK DETECTED - CONFIRM",
                     ["yes"], balls=len(positions))
            return False
        self.table.learn_foot(positions)
        self.game.new_rack(frame)
        self.won_at, self.candidates, self.racked_for = None, [], 0
        self.far_claims, self.vanished = [], []
        self.pending, self.held = None, 0
        self.last_credited, self.replace_at, self.settled_at = [], None, None
        self.prompts = []
        self._rest()
        self.settled_balls = len(self.resting)
        return True

    # ---- settling ----------------------------------------------------------

    def _clear_window(self):
        """The settled window's frames, the ones with nothing leaning in first.

        Falls back to the whole window when every frame in it was busy, because
        a reading taken through a hand still beats no reading at all - and the
        caller that cannot afford that is _settle, which waits for a clear table
        instead of calling this and hoping.
        """
        clear = [pos for ok, pos in self.window if ok]
        return clear if clear else [pos for _ok, pos in self.window]

    def _clear_counts(self):
        """... and the same for the ball counts."""
        counts = [n for ok, n in self.recent if ok]
        return counts if counts else [n for _ok, n in self.recent]

    def _clear_census(self, agg=_median):
        """... and how many balls of each class the settled table shows.

        Per class rather than per ball, because that is what survives a track
        dying: the ball is gone from the count AND from its class, whatever
        happened to the id that carried it.

        `agg` is how the frames of the window are reconciled - the middle
        reading by default, and `max` when the question is whether a class is
        REALLY missing rather than merely mislabelled for a frame.
        """
        seen = [c for ok, c in self.census if ok] or [c for _ok, c in self.census]
        if not seen:
            return Counter()
        return Counter({cls: agg([c.get(cls, 0) for c in seen])
                        for cls in set().union(*(set(c) for c in seen))})

    def _frame_at_rest(self):
        """The frame of the settled window whose count is the most typical one.

        The fewest balls hidden by a hand, and the fewest phantoms from one that
        is halfway out of frame.
        """
        frames = self._clear_window()
        counts = sorted(len(f) for f in frames)
        target = counts[len(counts) // 2] if counts else 0
        return min(frames, key=lambda f: abs(len(f) - target), default=[])

    def _rest(self):
        """Remember the table as it stands now, to compare the next one with.

        Positions only. The COUNT reference is not taken here, because this runs
        at the end of every episode - including a player fiddling at the table,
        who is standing in the shot while it is taken and inflates it.
        """
        self.resting = self._frame_at_rest()
        self.settled_census = self._clear_census()
        if self.settled_balls is None:
            self.settled_balls = len(self.resting)
        recent = self.frame - self.shots.settle
        self.eight_at_rest = self.last_seen.get(EIGHT, -1) >= recent
        self.cue_at_rest = self.last_seen.get(CUE, -1) >= recent

    def _racked(self, positions):
        """Is the table a fresh triangle rather than a game in progress? (M-09)

        Counted by contact, not by how many balls are on the table: after a
        break there can still be sixteen, but they are spread across the cloth.
        A rack is the one arrangement where nearly every ball touches another.

        Nine-ball racks a diamond of nine, so both the count and the contact
        threshold scale with the discipline rather than being hard-coded to a
        triangle of fifteen.
        """
        nine = self.discipline == games.NINE_BALL
        full = FULL_RACK_NINE if nine else FULL_RACK
        need = RACK_TOUCHING_NINE if nine else RACK_TOUCHING
        if len(positions) < full:
            return False
        reach = (TOUCHING_R * self.ball_r) ** 2
        touching = sum(
            1 for i, (x, y) in enumerate(positions)
            if any((x - qx) ** 2 + (y - qy) ** 2 <= reach
                   for j, (qx, qy) in enumerate(positions) if i != j))
        return touching >= need

    def _seen_now(self, point):
        """Is there a ball at this spot in ANY frame of the settled window?

        Any, not all: a ball blinks out for a frame at a time all through this
        footage, and treating one dropped frame as "that ball moved" would turn
        every episode into a shot.

        Busy frames are left out of the search rather than searched and failed.
        A ball under a hand is not at its old spot in that frame and is not
        anywhere else either - it is simply not being seen.
        """
        px, py = point
        return any((px - qx) ** 2 + (py - qy) ** 2 <= self.moved_r ** 2
                   for frame in self._clear_window() for qx, qy in frame)

    def _settle(self, frame):
        """An episode of motion has ended. Was it a shot, and what went down?

        The test is what the table looks like now against how it stood before: a
        shot moves the cue ball AND whatever it hit, or leaves a gap where a
        ball went down. A player reaching across moves nothing, and a player
        placing the ball in hand moves exactly one thing - which is M-07, and is
        now handled rather than merely ignored.

        `frame` may be older than the current frame: a settling that ended with
        a hand still over the table is held until the table clears, and then
        judged here, against the cleared window but logged at the frame the
        balls actually stopped.
        """
        held = self.held
        self.pending, self.held = None, 0
        # Set before anything reads the census: how long this settling waited
        # is part of how much its census is worth (CENSUS_MAX_HOLD).
        self.last_hold = held
        # Overtaken while held, without one clear frame to judge from: every
        # reading below would be taken through the hand, and a ball under a hand
        # reads as a ball that MOVED or LEFT. Measured at 3:04 in this footage:
        # a player aiming with a bridge hand touching the cue ball had it read
        # as gone, scored as a scratch, and remembered without it - so the real
        # pot on the next shot was refused. With no pot claimed there is nothing
        # a verdict could credit, so there is no verdict, and the table the next
        # episode is judged against stays the last one that was actually seen.
        if not any(ok for ok, _pos in self.window) and not self.candidates:
            self.episodes.append({"frame": frame, "held": held,
                                  "verdict": "not judged - the table never cleared"})
            return []
        # More balls gone than any shot can pot: the players are clearing the
        # table, not playing it. Judging that scores a rack-ending call on
        # whatever claim happens to be lying about - at the end of rack 2 seven
        # balls went into their hands at once and the 8 was called potted.
        if (self.settled_balls or 0) - _median(self._clear_counts()) > MAX_SHOT_POTS:
            self.episodes.append({"frame": frame, "held": held,
                                  "was": self.settled_balls,
                                  "now": _median(self._clear_counts()),
                                  "verdict": "not judged - the table was cleared"})
            self._refuse(frame, self.candidates, "the table was cleared")
            self.candidates, self.far_claims, self.vanished = [], [], []
            self.resting, self.settled_balls, self.settled_census = None, None, None
            self.game.note("note", "TABLE CLEARED - NOT A SHOT", rule="M-09",
                           quiet=True)
            return []
        moved = [p for p in (self.resting or []) if not self._seen_now(p)]
        racked = (not self.game.struck) and (self.settled_balls or 0) >= FULL_RACK
        need = SHOT_MOVED_BREAK if racked else SHOT_MOVED

        # A table that is one ball lighter than it was has been SHOT at, however
        # little else appears to have moved. Balls do not leave the cloth on
        # their own, and the rearrangement test can miss a real shot outright: a
        # ball flying into a pocket at 116 px a frame was refused because the
        # others happened to come to rest near where they started.
        #
        # What the missing ball needs is a second witness, and a claim at a
        # mouth is not the only one there is: the settled census naming the
        # class that left says the same thing, and says it for the pots no
        # track ever claimed. At 7:07 in this footage a solid went down with no
        # claim, one ball read as moved, and the episode was refused outright -
        # so the pot went unscored, the turn never passed, and the count stayed
        # wrong for the rest of the rack.
        lost = (self.settled_balls or 0) - _median(self._clear_counts())
        if lost >= 1 and (self.candidates
                          or self._census_gone(lost, self.settled_census)):
            need = min(need, len(moved))
        record = {"frame": frame, "moved": len(moved), "was": self.settled_balls,
                  "now": _median(self._clear_counts()), "held": held,
                  "claims": [p._asdict() for p in self.candidates]}
        self.episodes.append(record)

        if len(moved) < need:
            record["verdict"] = f"not a shot - only {len(moved)} ball(s) moved"
            self._refuse(frame, self.candidates, record["verdict"])
            # M-07: exactly one ball in a new place, with no shot to explain it.
            # Never a foul - the client is explicit that enforcing it would make
            # casual play miserable - and the useful response is to show where
            # the ball goes back.
            # Ball in hand is excluded: placing it is exactly one ball moving,
            # and it is the one hand-moved ball that is supposed to happen.
            if (len(moved) == 1 and not self.candidates and self.game.struck
                    and not self.game.ball_in_hand and not self.game.over):
                self._replaced(moved, frame)
            self.candidates = []
            self._rest()
            return []

        self.before = self.settled_balls
        self.before_census = Counter(self.settled_census or {})
        self.settled_at = frame          # M-02 counts from here - shots only
        shooter = self.game.turn
        self.game.shot_started(frame)
        credited = self._settle_up(frame)
        record["credited"] = [p._asdict() for p in credited]

        was_open = getattr(self.game, "open_table", False)
        self.last_facts = self._facts(credited, moved)
        self.game.shot_ended(self.last_facts, frame)
        self.last_credited = list(credited)
        self.candidates = []
        self._confirm_groups(shooter, credited, was_open)
        if getattr(self.game, "illegal_break", False):
            self.ask("illegal_break", "M-10",
                     f"ILLEGAL BREAK - {self.game.name(self.game.turn)}: "
                     f"PLAY ON OR RE-RACK?", ["play-on", "re-rack"])

        record["verdict"] = self.game.status
        if self.game.over and self.won_at is None:
            self.won_at = frame
        self._rest()
        record["corrected"] = self._reconcile_score()
        # A potted object ball stays down; a potted cue ball is fished out and
        # put back, so the table will hold one more than it does at this moment.
        self.settled_balls = (_median(self._clear_counts()) +
                              sum(1 for p in credited if p.cls == CUE) +
                              self.last_facts.off_table.count(CUE))
        # A re-rack (the 8 on the break) puts fifteen balls back, so nothing
        # counted before it means anything afterwards. Every judged shot leaves
        # the rack struck, so a rack that is not struck now was re-racked.
        if not self.game.struck:
            self.resting, self.settled_balls = None, None
        return credited

    def _reconcile_score(self):
        """Check the score against what the table still holds (8-ball).

        The scoreboard's version of what _settle_up does for a single shot. A
        pot can be missed outright - a ball that drops while a hand crosses the
        frame, or on an episode that was never judged a shot at all - and the
        score then stays one short for the rest of the rack, with nothing in
        the pipeline able to notice. The table notices: seven balls to a group,
        and whatever is not on the cloth is down.

        This is also what keeps the rack-ending call honest, which is the
        expensive half. The 8 is judged against the shooter's count as it stood
        BEFORE the shot - W-01's win and W-02's early 8 are the same event told
        apart by that number - so a player who has really cleared their group
        but is credited with five loses a rack they just won. Correcting the
        score as each shot settles is what stops that, one settling before the
        8 goes down.

        Read only off an unambiguous table: from the frames of the window with
        nothing leaning over it, not held long for a hand (CENSUS_MAX_HOLD),
        and with every ball the count sees carrying exactly one label. A
        flickering vote pushes the classes past the count, and
        then the reading is skipped rather than guessed at - the next settling
        will be clean, and being a shot late costs nothing here.
        """
        game = self.game
        if (self.discipline != games.EIGHT_BALL or game.over or not game.struck
                or self.last_hold > CENSUS_MAX_HOLD
                or not any(ok for ok, _c in self.census)):
            return []
        seen = self._clear_census(max)
        if (sum(seen.get(cls, 0) for cls in CENSUS_CLASSES)
                != _median(self._clear_counts())):
            return []
        return game.reconcile_score({STRIPE: seen.get(STRIPE, 0),
                                     SOLID: seen.get(SOLID, 0)})

    def _facts(self, credited, moved):
        """Everything the rules layer needs about this shot, honestly tristated.

        The three uncertain fields are where this system either keeps the
        client's promises or quietly breaks them, so each is worth a line:

        hit_object    False when nothing but the cue ball moved all shot, which
                      is F-05 and the easiest foul on the table for a camera.
                      None when no cue ball was ever identified, because then
                      "only the cue moved" cannot be established either.

        rail_contact  True as soon as any ball's centre entered the cushion
                      band. False only when the table geometry is known and
                      nothing did. None when there is no geometry to test.

        first_contact The first non-cue ball to start moving after the cue ball
                      did. At 4.9 fps this is a guess, and its CONFIDENCE says
                      so: the two are usually seen moving on the same frame,
                      which is worth very little, and F-03 needs 85% before it
                      will call a foul. Below that F-09 takes over and the shot
                      is logged for review instead of ruled on. That is the
                      designed outcome, not a gap to be closed by raising the
                      numbers until fouls start appearing.
        """
        movers = self.shots.movers
        classes = set(movers.values())
        # A ball that went down was hit, whatever the tracker managed to see
        # move. At 7:07 in this footage a solid was potted on a shot where only
        # the cue ball was ever matched between frames, and F-05 called "no
        # ball hit" against the player who had just potted their own ball.
        potted = any(p.cls in (STRIPE, SOLID, EIGHT) for p in credited)
        objects_moved = bool(classes & {STRIPE, SOLID, EIGHT}) or potted
        saw_cue = CUE in classes
        hit_object = None
        if objects_moved:
            hit_object = True
        elif saw_cue:
            hit_object = False

        first_cls, confidence = None, 0.0
        if self.shots.first_mover:
            seen_at, first_cls = self.shots.first_mover
            if self.shots.cue_started is not None:
                # One clear frame between the cue ball moving and the object
                # ball moving is the only thing that makes this better than a
                # coin toss, and at this frame rate it almost never happens.
                confidence = 0.55 if seen_at - self.shots.cue_started >= 1 else 0.3
            else:
                confidence = 0.2
            if len(classes & {STRIPE, SOLID, EIGHT}) > 1:
                confidence *= 0.7       # several balls moved at once

        self.shot_confidence = (min(self.pots.confidence(p) for p in credited)
                                if credited else None)
        eight = next((p for p in credited if p.cls == EIGHT), None)
        off_table, off_conf = self._off_table(credited)
        # Rail contact SEEN is proof at any frame rate; rail contact NOT seen is
        # only proof at a frame rate that could have seen it (RAIL_MIN_FPS).
        slow = self.sample_fps < RAIL_MIN_FPS
        rail = None if (self.rail_contact is False and slow) else self.rail_contact
        rail_balls = len(self.rail_tracks) if self.table.ok else None
        if slow and rail_balls is not None and rail_balls < games.BREAK_RAIL_BALLS:
            rail_balls = None
        return games.facts(
            pots=[p.cls for p in credited],
            off_table=off_table,
            rail_contact=rail,
            hit_object=hit_object,
            first_contact=first_cls,
            contact_conf=round(confidence, 2),
            moved=len(moved),
            rail_balls=rail_balls,
            # W-03 can lose a rack on this, so only a pocket the 8 was SEEN
            # going into counts - never one inferred from its heading.
            eight_pocket=(eight.pocket if eight and eight.source != INFERRED
                          else None),
            off_conf=off_conf,
        )

    #: How sure an off-the-table inference is. Deliberately under M-14's bar,
    #: so every such call carries a visible dot.
    OFF_TABLE_CONFIDENCE = 0.6

    def _off_table(self, credited):
        """W-07 and F-01's "off the table": a ball that left without a pocket.

        Inferred, never observed - a camera over the cloth does not see the
        floor. Three things all have to hold before the 8 or the cue ball is
        said to have left the table: it was on the table when the last shot
        settled, no track has carried its label since this shot ended, and the
        settled count dropped by more than the pots that were credited. If any
        one of them fails the ball is merely unlabelled, and nothing is said.
        """
        after = _median(self._clear_counts())
        before = self.before if self.before is not None else after
        unexplained = (before - after) - len(credited)
        if unexplained < 1:
            return [], 1.0
        since = self.frame - self.shots.settle
        # One sighting is not enough to say a ball WAS there, and one missed
        # frame is not enough to say it is gone: the 8 at the end of rack 1 had
        # been potted two shots earlier, and a single flickering label was all
        # it took to end the rack against the wrong player. The census is the
        # steadier witness - consistently on the table before, consistently
        # absent now.
        before_cls = Counter(self.before_census or {})
        now_cls = self._clear_census(max)
        gone = []
        for cls, was_there in ((EIGHT, self.eight_at_rest),
                               (CUE, self.cue_at_rest)):
            if not was_there or any(p.cls == cls for p in credited):
                continue
            if before_cls.get(cls, 0) < 1 or now_cls.get(cls, 0) > 0:
                continue
            if self.last_seen.get(cls, -1) < since:
                gone.append(cls)
        return gone[:unexplained], self.OFF_TABLE_CONFIDENCE

    def _settle_up(self, frame):
        """Decide what this shot put down, by comparing the settled tables.

        A disappearing track is weak evidence. At 4.9 fps a struck ball crosses
        far more than the tracker's search radius, so its id dies mid-flight and
        is reborn elsewhere, and two balls colliding can swap ids outright - so
        a track that died beside a pocket may belong to a ball still rolling,
        and the ball that went down may have died as a track nowhere near one.

        What is NOT weak evidence is the table once everything has stopped, and
        it answers both halves of the question without asking a single track to
        survive the shot:

          HOW MANY   the settled count before against the settled count now.
          WHICH      the settled census, class by class: a stripe on the table
                     before and not after went down, whoever carried its id.

        So the census is the authority on what was pocketed, and the claims
        made at the mouths are only asked the one thing they are good for -
        WHICH POCKET (_attribute). A ball that rattled out of the jaw is back
        on the settled table, in both the count and the census, and is simply
        never credited (W-10).

        The claim ranking the census replaced is still here, for the settlings
        whose census cannot be read (_census_gone): then the claims are only
        ever as good as the drop in the count, and the best-explained ones get
        the benefit of it.
        """
        after = _median(self._clear_counts())
        before = self.before if self.before is not None else after
        dropped = max(0, before - after)
        if not dropped:
            self._refuse(frame, self.candidates,
                         f"table still holds {after} balls (was {before})")
            return []

        # A claim for a ball that is still sitting where it was last seen is not
        # a pot, however well it ranks. A ball resting by the top-middle pocket
        # blinks out under passing hands all through this footage, and at 5:32
        # that blink took the credit for a stripe that went down elsewhere.
        still = [c for c in self.candidates if self._still_there(c)]
        self._refuse(frame, still, "ball still sitting where it was")
        near = sorted([c for c in self.candidates if c not in still],
                      key=self.pots.rank)
        # A ball reborn as a new track mid-flight can vanish short of the mouth
        # (POT_FAR_MOUTH). Only a YOUNG track can be that - an established one
        # that dies out there lost its id, not its ball. At 7:30 and 8:12 in
        # this footage exactly that, filling a count that was a ball out, called
        # two scratches on a cue ball that never left the cloth. Nor is it ever
        # the cue ball while the cue ball is in plain view: at 7:30 a one-frame
        # cue track 119 px from a pocket called a scratch with the cue on the
        # cloth.
        cue_here = self.last_seen.get(CUE, -1) >= self.frame - self.shots.settle
        far = sorted([c for c in self.far_claims if c.age < POT_MIN_AGE
                      and not (c.cls == CUE and cue_here)
                      and not self._still_there(c)],
                     key=self.pots.rank)

        gone = self._census_gone(dropped)
        credited, spent = ([], set()) if gone is None else \
            self._attribute(frame, gone, near + far)
        # The census only sees labelled balls. Balls the count lost beyond what
        # it explains went down unlabelled, so only a claim that is unlabelled
        # too can account for them - a claim carrying a class the census says
        # is all still on the table is the id-switch this comparison exists to
        # see through. Unless the census before the shot was itself missing
        # balls: then a labelled ball may have gone down that it never counted.
        blind = gone is None or before > sum(
            (self.before_census or {}).get(c, 0) for c in CENSUS_CLASSES)
        contradicted = []
        for c in near + far:
            if id(c) in spent:
                continue
            if not blind and c.cls != UNKNOWN:
                contradicted.append(c)
            elif len(credited) < dropped:
                credited.append(c)
                spent.add(id(c))
        self._refuse(frame, [c for c in near if c in contradicted],
                     "settled census still shows every ball of that class")
        self._refuse(frame, [c for c in near
                             if id(c) not in spent and c not in contradicted],
                     f"table still holds {after} balls (was {before})")
        credited = self._cue_under_another_label(credited)
        self.all_pots.extend(credited)
        return credited

    def _census_gone(self, dropped, before=None):
        """The classes the settled table lost, or None when it cannot say.

        Before the shot, three stripes, a solid, the 8 and the cue; after it,
        three stripes, the 8 and the cue. The solid went down - at 4:29 in this
        footage exactly that ball's track died 232 px from the nearest pocket
        and nothing claimed it, and at 5:32 an id that died beside a pocket
        scored a stripe that was still rolling.

        The most generous reading of what is still on the table: a class only
        counts as missing when it is missing in EVERY clear frame of the window.
        A stripe read as a solid for one frame looks exactly like a stripe that
        went down, and over this match that flicker credited eleven stripes,
        two of them balls the rack could not even hold.

        And it is only read at all off clean evidence: a settling waited on for
        no longer than CENSUS_MAX_HOLD, read from the frames of the window with
        nothing leaning over the table, and agreeing with the count. More
        classes short than balls gone is a label that changed, not a ball that
        left, and the census is not believed.

        ONE clear frame is enough, and demanding more was expensive. The count
        this census is checked against is taken from the same clear frames, so
        holding the census to a stricter standard than the count only means the
        weaker evidence decides alone. At 10:10 in this footage a cue ball ran
        the length of the table, outjumped the tracker, and died beside the top
        left pocket under a player's arm; the clear frame in that window showed
        the cue ball sitting on the cloth and the 8 gone from it, and refusing
        to read it scored a scratch that never happened, left the 8 that really
        went down unexplained, and handed away the rack.

        `before` is the table this one is compared with, defaulting to the
        census taken before the shot being judged. _settle passes the last
        settling's census directly, because it asks this question before it has
        decided there was a shot at all.
        """
        before = self.before_census if before is None else before
        if (before is None or self.last_hold > CENSUS_MAX_HOLD
                or not any(ok for ok, _c in self.census)):
            return None
        now = self._clear_census(max)
        gone = [cls for cls in CENSUS_CLASSES
                for _ in range(max(0, before.get(cls, 0) - now.get(cls, 0)))]
        return gone if len(gone) <= dropped else None

    def _attribute(self, frame, gone, claims):
        """Put each ball the census lost into a pocket. Returns (pots, spent).

        This is the only thing the in-motion evidence is still asked, and it is
        the part that can afford to be wrong: the census has already decided the
        ball is down, and a pocket mistaken for its neighbour changes a replay
        label, not the score. Best evidence first:

          1. a claim at a mouth carrying the same class;
          2. a claim at a mouth from a track too young for its label to mean
             anything - the class is the census's, the pocket is the claim's.
             Never for the 8, whose pocket W-03 can lose a rack on;
          3. the ball's last heading, from the track of that class that was
             lost during the shot (heading_pocket);
          4. no pocket at all.

        `spent` holds the ids of the claims used, so none is spent twice.
        """
        spent, pots, open_ = set(), [None] * len(gone), []
        for i, cls in enumerate(gone):
            match = next((c for c in claims
                          if id(c) not in spent and c.cls == cls), None)
            if match is None:
                open_.append(i)
                continue
            spent.add(id(match))
            pots[i] = match._replace(source=CENSUS)
        for i in list(open_):
            if gone[i] == EIGHT:
                continue
            match = next((c for c in claims
                          if id(c) not in spent and c.age < POT_MIN_AGE), None)
            if match is None:
                continue
            spent.add(id(match))
            pots[i] = match._replace(cls=gone[i], source=CENSUS)
            open_.remove(i)
        lost = set()
        for i in open_:
            pots[i] = self._heading(frame, gone[i], lost)
        return pots, spent

    def _heading(self, frame, cls, lost):
        """Step 3 and 4 of _attribute: a pot for `cls` from where it was going.

        Of the tracks of that class lost this shot, a clear-table one first and
        then the latest - the ball that went down is the last to disappear. A
        track lost in open table with no heading into any pocket, or one whose
        ball is still sitting where it was lost, lost its id and not its ball.
        """
        best = None
        for v in self.vanished:
            if v.cls != cls or id(v) in lost:
                continue
            pocket, dist = heading_pocket(v.x, v.y, v.was, self.pots.pockets)
            if pocket is None:
                pocket, dist = nearest_pocket(v.x, v.y, self.pots.pockets)
                if dist > self.pots.far_reach:
                    continue
            closed = (nearest_pocket(v.was[0], v.was[1], [pocket])[1] - dist
                      if v.was else 0.0)
            pot = Pot(v.frame, v.track, cls, pocket.name, round(dist, 1),
                      round(closed, 1), v.age, v.busy, v.x, v.y, INFERRED)
            if self._still_there(pot):
                continue
            key = (v.busy, -v.frame)
            if best is None or key < best[0]:
                best = (key, pot, v)
        if best is None:
            return Pot(frame, -1, cls, None, 0.0, 0.0, 0, False, None, None,
                       INFERRED)
        lost.add(id(best[2]))
        return best[1]

    def _still_there(self, claim):
        """Was this claim a ball at rest that the settled table still shows?"""
        return (claim.x is not None and claim.closed < self.pots.approach
                and self._seen_now((claim.x, claim.y)))

    def _cue_under_another_label(self, credited):
        """F-01: the cue ball went down, but its track said it was something else.

        A cue ball running the length of the table is broken into new tracks at
        this frame rate, and a new track's first few votes are taken off a
        blurred white ball - at 5:32 one was voted STRIPE three frames old and
        potted as a stripe, hiding a scratch and scoring the shooter a ball.

        The evidence that settles it is the same one _off_table uses: the cue
        ball was on the table when this shot began and no track has carried its
        label since the balls came to rest. If so, and a credited claim came from
        a track too young for its label to mean anything, that claim was the cue.
        """
        if not self.cue_at_rest or any(p.cls == CUE for p in credited):
            return credited
        if self.last_seen.get(CUE, -1) >= self.frame - self.shots.settle:
            return credited
        young = [p for p in credited if p.age < POT_MIN_AGE]
        if not young:
            return credited
        cue = max(young, key=lambda p: (p.frame, p.closed))
        return [p._replace(cls=CUE) if p is cue else p for p in credited]

    def _refuse(self, frame, claims, why):
        for p in claims:
            self.ignored.append(p)
            self.pots.rejected.append({
                "frame": frame, "track": p.track, "cls": p.cls,
                "pocket": p.pocket, "age": p.age, "why": why})

    # ---- read-only views ---------------------------------------------------

    @property
    def celebration(self):
        """(winner's name, 0..1 through the celebration), or None.

        A rack ends on one ball, and the moment deserves to be visible for
        longer than the single frame the 8 disappears on - a line of status text
        changing is easy to miss entirely.
        """
        if self.won_at is None or self.game.winner is None:
            return None
        elapsed = (self.frame - self.won_at) / self.fps
        if elapsed < 0 or elapsed > WIN_SECONDS:
            return None
        return self.game.players[self.game.winner], elapsed / WIN_SECONDS

    @property
    def undo_banner(self):
        """W-02's ten-second window, as a 0..1 countdown for the HUD.

        A prominent window on top of M-03's ordinary back button, because this
        is a call that ENDS A RACK and the client asked for it specifically.
        """
        until = self.game.undo_until
        if not until or self.frame > until:
            return None
        span = rules.EIGHT_EARLY_UNDO_SECONDS * self.fps
        return max(0.0, min(1.0, (until - self.frame) / span))

    @property
    def unexplained(self):
        """Episodes where the table lost a ball and nothing was credited.

        The self-check worth watching: a ball left the table and the system
        cannot say which, so a pot went unscored.
        """
        return [e for e in self.episodes
                if e.get("was") and e["now"] < e["was"] and not e.get("credited")]

    @property
    def clock(self):
        """Video position as MM:SS - the scoreboard's game clock."""
        secs = int(self.frame / self.fps)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    @property
    def status(self):
        if not self.has_pockets:
            return "NO POCKETS - POTS NOT TRACKED"
        if self.game.timeouts.active is not None:
            who = self.game.name(self.game.timeouts.active["player"])
            return f"TIME OUT - {who}"
        if self.waiting_for_cue:
            return "WAITING FOR THE CUE BALL"
        # A waiting prompt is NOT the status: it has its own panel on the HUD,
        # and in a rendered video nobody can tap it, so letting it take over
        # the status line would hide every verdict after it.
        if self.game.over:
            return self.game.status
        if self.shots.moving:
            return "SHOT IN PLAY"
        # Only while a verdict is actually being held: "a hand is on the table"
        # is true for a third of this footage and would otherwise be the line
        # the board showed most of the time, saying nothing. Held, it is the
        # honest answer to "why has the score not changed yet".
        if self.pending is not None:
            return "WAITING FOR THE TABLE TO CLEAR"
        return self.game.status

    def summary(self):
        out = dict(self.game.summary())
        out.update({"pots": len(self.all_pots), "phantoms": self.phantoms,
                    "replaced": self.replaced, "reviews": len(self.reviews),
                    "unexplained": len(self.unexplained),
                    "busy_frames": self.busy_frames})
        return out
