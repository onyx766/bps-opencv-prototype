"""Hold a steady CUE / EIGHT / STRIPE / SOLID verdict for each tracked ball.

A single frame is not enough to trust. The measurement that separates stripes
from solids is how much white shows, and on this footage the two classes meet
around 0.20: a solid measured 0.18 and a stripe 0.21. A ball read one frame at
a time flickers between the two.

Time fixes it, because the ambiguity is not symmetric. A stripe only looks like
a solid while its band happens to face away; roll it and the band comes round.
A solid has no band to show and never climbs. So a vote over recent frames
settles on the right answer, and keeps the label still while it does.

Two balls are also unique: there is one cue ball and one 8-ball. That matters
because the one thing that genuinely fools the white test is a stripe lying
white-pole-up, which reads as pure white - one measured 0.89 with no colour at
all. Nothing about that ball on its own says it is not the cue. What says so is
that another ball is whiter, and only one of them can be the cue.
"""

from collections import Counter, deque

import ball_class

#: How the four classes are drawn: label text, then ring BGR.
LOOKS = {
    "cue":    ("CUE",    (245, 245, 245)),
    "eight":  ("8",      (95, 95, 95)),
    "stripe": ("STRIPE", (40, 215, 255)),
    "solid":  ("SOLID",  (235, 120, 20)),
}
UNKNOWN_BGR = (150, 150, 150)

VOTE_WINDOW = 7      # frames of history kept per track
MIN_VOTES = 3        # ... and how many it takes before a label is shown

#: Consecutive measurements that disagree with the settled vote before the vote
#: is thrown away rather than averaged into.
#:
#: A vote is only worth taking while the frames in it belong to the SAME ball,
#: and at 4.9 fps they sometimes do not. When the cue ball reaches its object
#: ball the two occupy nearly the same place on the frame where they touch, and
#: the tracker hands the id across: one track was the cue ball, and from the
#: next frame it is a solid rolling toward a pocket. Measured here - track 173
#: read white 0.93-0.99 for a hundred frames, then 0.06, 0.19, 0.05.
#:
#: Averaging across that is worse than useless. The stale cue votes outnumbered
#: the true ones, the uniqueness rule then demoted the losing cue claim to
#: STRIPE, and a solid was potted as the wrong group - a foul against a player
#: who had just potted their own ball. Three measurements agreeing against the
#: vote are not noise; they are a different ball, so the vote starts over.
SWAP_FRAMES = 3


class Label:
    """What the overlay draws for one ball."""

    __slots__ = ("cls", "confidence")

    def __init__(self, cls, confidence):
        self.cls = cls
        self.confidence = confidence

    @property
    def text(self):
        return LOOKS[self.cls][0]

    @property
    def bgr(self):
        return LOOKS[self.cls][1]


class BallClassRegistry:
    """Per-track vote over the last few frames, plus the cue/8 uniqueness rule."""

    def __init__(self):
        self.votes = {}        # track id -> deque of recent class names
        self.fresh = {}        # track id -> the last few RAW measurements
        self.stats = {}        # track id -> the most recent measurement
        self.events = []

    def update(self, tracked, hsv, frame=0):
        """Measure every tracked ball this frame; return {track_id: Label}."""
        self._forget(tid for tid, *_ in tracked)

        white_ref = ball_class.white_reference(hsv, [(x, y, r)
                                                     for _t, x, y, r in tracked])
        for tid, x, y, r in tracked:
            cls, stats = ball_class.classify_ball(hsv, x, y, r, white_ref)
            if cls is None:
                continue
            votes = self.votes.setdefault(tid, deque(maxlen=VOTE_WINDOW))
            fresh = self.fresh.setdefault(tid, deque(maxlen=SWAP_FRAMES))
            fresh.append(cls)
            if self._swapped(votes, fresh):
                self.events.append({"frame": frame, "track": tid,
                                    "type": "swapped",
                                    "from": Counter(votes).most_common(1)[0][0],
                                    "to": fresh[0]})
                votes.clear()
                votes.extend(fresh)
            else:
                votes.append(cls)
            self.stats[tid] = stats

        labels = {}
        for tid, *_ in tracked:
            votes = self.votes.get(tid)
            if not votes:
                continue
            cls, n = Counter(votes).most_common(1)[0]
            labels[tid] = Label(cls, "voted" if n >= MIN_VOTES else "pending")

        self._enforce_unique(labels, frame)
        return labels

    def label(self, tid):
        votes = self.votes.get(tid)
        if not votes:
            return None
        cls, n = Counter(votes).most_common(1)[0]
        return Label(cls, "voted" if n >= MIN_VOTES else "pending")

    @staticmethod
    def _swapped(votes, fresh):
        """Have the last few measurements all disagreed with the settled vote?

        Only asked of a track that HAS a settled vote: a few frames of history
        can disagree with each other honestly, and a young track has nothing
        worth throwing away.
        """
        if len(fresh) < SWAP_FRAMES or len(votes) < MIN_VOTES:
            return False
        if len(set(fresh)) > 1:
            return False
        return fresh[0] != Counter(votes).most_common(1)[0][0]

    def _forget(self, live):
        """Drop tracks the tracker has given up on, so votes cannot outlive them."""
        live = set(live)
        for tid in [t for t in self.votes if t not in live]:
            del self.votes[tid]
            self.fresh.pop(tid, None)
            self.stats.pop(tid, None)

    def _enforce_unique(self, labels, frame):
        """Leave one cue and one 8-ball standing; demote every other claim.

        The winner is the most extreme on the measurement that defines the
        class - whitest for the cue, darkest for the 8 - because a ball that
        merely resembles the cue always resembles it less than the cue does.
        """
        for cls, key in (("cue", "white"), ("eight", "dark")):
            claims = [tid for tid, lb in labels.items() if lb.cls == cls]
            if len(claims) < 2:
                continue
            keep = max(claims, key=lambda t: self.stats.get(t, {}).get(key, 0.0))
            for tid in claims:
                if tid == keep:
                    continue
                labels[tid] = Label(self._demote(cls, self.stats.get(tid, {})),
                                    "voted")
                self.events.append({"frame": frame, "track": tid,
                                    "type": "demoted", "from": cls,
                                    "to": labels[tid].cls})

    @staticmethod
    def _demote(cls, stats):
        """Second-best class for a ball that lost a uniqueness contest.

        A rejected cue is a white-pole-up stripe - that is the only other thing
        on the table which reads as all white. A rejected 8 is whatever it
        would have been had it not been the darkest ball, so re-run the normal
        cut with the darkness test taken away.
        """
        if cls == "cue":
            return "stripe"
        white = stats.get("white", 0.0)
        return "stripe" if white > ball_class.STRIPE_WHITE else "solid"

    def summary(self):
        counts = Counter()
        for tid in self.votes:
            lb = self.label(tid)
            if lb is not None:
                counts[lb.cls] += 1
        return {"tracked": len(self.votes), "by_class": dict(counts)}
