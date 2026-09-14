"""M-12: the match record. Every other rule in the suite is a view of this.

The client's line is "the log is the source of truth; the scoreboard is a view
of it", and that is not a slogan - it is what makes M-03, M-13 and the whole
analytics story possible without paying for them separately.

  * M-03 (undo) is a matter of walking back to a remembered state, not of
    unpicking a running counter. A score kept as a number cannot be undone
    honestly; a score derived from a log can.
  * M-13 (disputes) needs every call to carry the frame it was made on, so the
    review can pull the footage around it. That is the clip pointer.
  * M-14 (confidence) needs every auto-call to carry the number it was made on,
    so the HUD can mark the shaky ones and so our own debugging is free.

WHAT M-03 ACTUALLY ASKS FOR, since it is the subtlest rule in the suite and the
reasoning behind it is the client's own league practice:

  A rack's events stay editable until the NEXT BREAK. Not for a number of
  seconds, not until a confirmation - until somebody breaks. The break is a
  real, observed event the camera can see (M-09), so the boundary is a thing
  that HAPPENED rather than a rule players have to remember. Players reconcile
  a mixed 9-ball score right up until someone racks and breaks, and the client
  was explicit that this "isn't that fast".

  So: no timers anywhere in this file. If you find yourself adding one, the
  rule has been misread.

  A rack that ends with a correction still outstanding is flagged SCORE
  UNCONFIRMED where the breaker can see it - advisory, never blocking. The
  match total stays open until an explicit CONFIRM FINAL SCORE, after which the
  back button disappears entirely and corrections go through a deliberate
  REOPEN, logged with who did it. A locked-out back button is worse than an
  abusable one, because disputes take minutes and a player who cannot fix the
  score will fix it by arguing instead.

F-10 lives here too, because it is the same idea pointed the other way: a foul
may be called until the next stroke begins, and not after. The window closes on
an observed event, so nobody can edit a score retroactively and nobody has to
adjudicate when the window shut.
"""

import copy
import json
import time

#: Event kinds that a player tap can take back (M-03). Structural events - a
#: rack starting, a break being observed - are not undoable on their own: they
#: are the boundaries the undo is measured against.
UNDOABLE = {"pot", "foul", "turn", "assign", "win", "loss", "defence",
            "timeout", "points", "dead", "manual"}

#: Kinds that mean a correction was made and not yet settled, which is what
#: raises SCORE UNCONFIRMED at the end of a rack.
CORRECTION = {"undo", "reopen"}


class Event:
    """One thing that happened, with everything a dispute would need.

    `rule` cites the client's test id, so the log reads against the spec and a
    disputed call can be traced to the row that authorised it.
    """

    __slots__ = ("seq", "frame", "clock", "wall", "rack", "shot", "kind",
                 "text", "rule", "confidence", "actor", "player", "data")

    def __init__(self, seq, frame, clock, rack, shot, kind, text,
                 rule=None, confidence=None, actor="system", player=None,
                 data=None):
        self.seq = seq
        self.frame = frame
        self.clock = clock
        self.wall = time.time()
        self.rack = rack
        self.shot = shot
        self.kind = kind
        self.text = text
        self.rule = rule
        self.confidence = confidence
        self.actor = actor          # "system" or "player"
        self.player = player        # index of the player it concerns, if any
        self.data = data or {}

    @property
    def undoable(self):
        return self.kind in UNDOABLE

    @property
    def uncertain(self):
        """M-14: was this called on evidence we would rather flag than hide?"""
        return self.confidence is not None and self.confidence < 0.85

    @property
    def clip(self):
        """M-13: where to find the footage. The frame IS the pointer."""
        return {"frame": self.frame, "clock": self.clock}

    def as_dict(self):
        return {"seq": self.seq, "frame": self.frame, "clock": self.clock,
                "rack": self.rack, "shot": self.shot, "type": self.kind,
                "text": self.text, "rule": self.rule,
                "confidence": self.confidence, "actor": self.actor,
                "player": self.player, "clip": self.clip, **self.data}


class MatchRecord:
    """The append-only log, plus the undo machinery M-03 builds on top of it.

    Undo is done with SNAPSHOTS rather than by inverting events. Inverting is
    tempting and wrong: "undo a foul" has to put back the turn, the ball in
    hand, the foul counter and, in nine-ball, whichever balls were killed by it,
    and every new rule adds another thing to remember to reverse. A snapshot
    taken before each committed shot cannot drift out of step with the rules,
    because it IS the rules' own state.
    """

    def __init__(self):
        self.events = []
        self._seq = 0
        self._snapshots = []      # [(event count, blob, label)], this rack only
        self.rack_locked = False  # M-03: set by the next break
        self.unconfirmed = False  # M-03: a rack ended mid-correction
        self.final = False        # M-03: CONFIRM FINAL SCORE has been tapped
        self.reopened = 0
        self.stroke_open = False  # F-10: is a shot currently under way?
        self.foul_deadline = None # ... and the shot a late foul may still touch

    # ---- writing ----------------------------------------------------------

    def add(self, frame, clock, rack, shot, kind, text, **kw):
        self._seq += 1
        ev = Event(self._seq, frame, clock, rack, shot, kind, text, **kw)
        self.events.append(ev)
        if kind in CORRECTION:
            self.unconfirmed = True
        return ev

    def snapshot(self, state, label=""):
        """Remember the state before a shot is judged, so it can be walked back.

        Deep-copied on the way in. The game hands over its own live containers
        and would otherwise mutate the snapshot along with itself, which fails
        silently and looks exactly like undo not working.
        """
        self._snapshots.append((len(self.events), copy.deepcopy(state), label))

    # ---- M-03 -------------------------------------------------------------

    @property
    def can_undo(self):
        """Is the back button on screen at all?

        Gone entirely once the score is final - the client asked for nothing
        editable and nothing on screen, not a button that refuses.
        """
        return bool(self._snapshots) and not self.final

    @property
    def undo_label(self):
        """What the back button would take back, so the tap is not a leap."""
        if not self.can_undo:
            return None
        return self._snapshots[-1][2] or "LAST SHOT"

    def undo(self, frame, clock, rack, shot):
        """Walk back one shot. Returns the state to restore, or None.

        Available to EITHER player with no PIN, by client decision: the people
        at the table are the ones who know what happened, and making them prove
        it turns a ten-second fix into an argument.
        """
        if not self.can_undo:
            return None
        count, blob, label = self._snapshots.pop()
        undone = [e for e in self.events[count:] if e.undoable]
        del self.events[count:]
        self.add(frame, clock, rack, shot, "undo",
                 f"UNDO - {label or 'LAST SHOT'}", rule="M-03", actor="player",
                 data={"undone": [e.text for e in undone]})
        return blob

    def lock_rack(self, frame, clock, rack, shot):
        """The next break has been observed: that rack is now history (M-03).

        Any correction still outstanding at this moment is the one the HUD will
        show as SCORE UNCONFIRMED, and it is shown to the BREAKER, who is the
        one person guaranteed to be looking at the table.
        """
        self._snapshots.clear()
        self.rack_locked = True
        if self.unconfirmed:
            self.add(frame, clock, rack, shot, "note",
                     "SCORE UNCONFIRMED - PREVIOUS RACK", rule="M-03")

    def open_rack(self):
        """A new rack: the previous one's snapshots are gone, this one starts clean."""
        self._snapshots.clear()
        self.rack_locked = False
        self.unconfirmed = False

    def confirm_final(self, frame, clock, rack, shot):
        """CONFIRM FINAL SCORE. The back button disappears after this."""
        if self.final:
            return False
        self.final = True
        self.unconfirmed = False
        self._snapshots.clear()
        self.add(frame, clock, rack, shot, "final", "FINAL SCORE CONFIRMED",
                 rule="M-03", actor="player")
        return True

    def reopen(self, frame, clock, rack, shot, who="player"):
        """Deliberate re-entry after confirmation, logged with who did it.

        Available until a new match starts on this table, and then never again -
        after that the result lives only in the log, which is the point of
        keeping one.
        """
        if not self.final:
            return False
        self.final = False
        self.reopened += 1
        self.add(frame, clock, rack, shot, "reopen",
                 f"SCORE REOPENED BY {str(who).upper()}", rule="M-03",
                 actor="player", data={"who": who})
        return True

    # ---- F-10 -------------------------------------------------------------

    def stroke_started(self, shot):
        """A stroke has begun: the previous shot's foul window is shut."""
        self.stroke_open = True
        self.foul_deadline = shot

    def stroke_ended(self, shot):
        """... and it is open again, bound to the shot that just finished."""
        self.stroke_open = False
        self.foul_deadline = shot

    def foul_window_open(self, shot):
        """May a foul still be called against this shot? (F-10)

        Closed once the next stroke begins. Nothing is retroactive, so nobody
        argues about a score that changed after the fact.
        """
        return (not self.stroke_open) and self.foul_deadline == shot

    # ---- reading ----------------------------------------------------------

    def since(self, seq):
        return [e for e in self.events if e.seq > seq]

    def last(self, kinds=None, n=1):
        pool = [e for e in self.events
                if kinds is None or e.kind in kinds]
        return pool[-n:]

    def recent_text(self, n=6):
        return [e.text for e in self.events[-n:]]

    def flagged(self):
        """Everything marked for a human to look at: reviews and shaky calls."""
        return [e for e in self.events
                if e.kind == "review" or e.uncertain]

    def as_list(self):
        return [e.as_dict() for e in self.events]

    def save(self, path, summary=None):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1,
                       "events": self.as_list(),
                       "flagged": [e.as_dict() for e in self.flagged()],
                       "final": self.final,
                       "reopened": self.reopened,
                       "summary": summary or {}}, fh, indent=2)
