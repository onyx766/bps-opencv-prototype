"""The rules, as the client locked them. Pure state - no OpenCV, no frames.

Two disciplines and three modes. What the camera saw arrives as a ShotFacts and
leaves as entries in the match record; nothing in this file knows what a pixel
is, which is what makes the whole suite testable at a desk.

WHAT THE CAMERA CAN AND CANNOT SEE, because the split runs through everything:

  Seen well        a ball vanished into a pocket; the table came to rest; the
                   cue ball is gone; a ball left the table; how many balls
                   moved; whether anything reached the cushion band.
  Seen poorly      which ball hit which first, and in what order. This is the
                   whole of F-03, and the client's answer is a threshold plus a
                   fallback (F-09), not a better guess.
  Not seen at all  intent. Defence (H-02) is the clearest case - it is known
                   only after the shot, by the person who played it - and so is
                   a push shot, a stalemate, and a ball nudged by a sleeve.

Every rule that falls in the second and third buckets is handled by a prompt or
a player tap, and the honest failure mode is always the same: no call, a review
flag, and the footage kept. F-09 states the principle outright, and it is the
default this file falls back to whenever it is unsure - if it cannot be
determined, there is no foul. A wrong call costs more than a missed one,
because a wrong call has to be argued out of the record by two people who are
trying to enjoy a game of pool.

ON DEAD BALLS (N-04), the one rule the client had to correct us on: balls made
during a scratch or a foul are credited to NEITHER player. Not to the opponent.
The invariant that keeps it honest is checked after every nine-ball rack -
p1 + p2 + dead must equal 10 - because the failure it guards against is silent.
"""

from collections import Counter, namedtuple

import rules
import skill_level
from match_record import MatchRecord

CUE, EIGHT, STRIPE, SOLID = "cue", "eight", "stripe", "solid"
UNKNOWN = "unknown"

GROUP_SIZE = 7
OTHER = {STRIPE: SOLID, SOLID: STRIPE}

#: Match modes. The mode is not a cosmetic setting - it changes which rules are
#: enforced, which are advisory, and whether a Skill Level is allowed to matter.
LEAGUE, CASUAL, PRACTICE = "league", "casual", "practice"
MODES = (LEAGUE, CASUAL, PRACTICE)

EIGHT_BALL, NINE_BALL = "8ball", "9ball"

#: Ball in hand, and where. F-02 is the only restricted case in the suite.
ANYWHERE = "anywhere"
BEHIND_HEAD_STRING = "behind-head-string"

#: Balls that must reach a cushion, or a ball be pocketed, for the break to be
#: legal (M-10). Enforced in league, advisory in casual.
BREAK_RAIL_BALLS = 4

#: In a NINE-BALL rack the 9 is the only striped ball on the table. That is not
#: a trick, it is the rack: balls 1-8 are solids and the black, and the 9 is the
#: yellow stripe. So the existing stripe/solid classifier - which is the one
#: measurement this footage supports well - identifies the money ball outright,
#: without needing to read a number off a curved surface at 4.9 fps.
#:
#: Numbers 1-8 are NOT separable this way, which is exactly why N-01 degrades to
#: a review rather than a foul when no number is supplied.
NINE_IS_STRIPE = True


#: Everything the vision layer managed to establish about one shot.
#:
#: Three fields are deliberately TRISTATE - True, False, or None for "could not
#: tell". None is not a failure to fill the field in; it is the answer, and it
#: routes to F-09 instead of to a foul.
ShotFacts = namedtuple(
    "ShotFacts",
    "pots off_table rail_contact hit_object first_contact contact_conf "
    "moved numbers",
)


def facts(pots=(), off_table=(), rail_contact=None, hit_object=None,
          first_contact=None, contact_conf=0.0, moved=0, numbers=()):
    return ShotFacts(list(pots), list(off_table), rail_contact, hit_object,
                     first_contact, float(contact_conf), int(moved),
                     list(numbers))


class TimeOuts:
    """H-05. Manual, never auto-detected, and charged only on END TIME OUT.

    The commit-on-close design is the client's specific pain point: a mis-tap on
    a crowded HUD used to cost a real time-out, and the player who lost it had no
    way to get it back. Here the tap starts a visible timer and nothing else -
    the charge is written when the time-out is closed, so cancelling costs
    nothing and the honest path is also the easy one.

    A team cannot be charged before the rack has been struck, so a time-out
    called while the table is still racked runs its clock but charges nobody.
    """

    def __init__(self, levels=(None, None)):
        self.allowance = [skill_level.timeouts(l) for l in levels]
        self.used = [0, 0]
        self.active = None       # {"player", "frame", "clock", "chargeable"}

    def remaining(self, i):
        return max(0, self.allowance[i] - self.used[i])

    def start(self, player, frame, clock, struck):
        """Tapping TIME OUT. Returns False only if one is already running."""
        if self.active is not None:
            return False
        self.active = {"player": player, "frame": frame, "clock": clock,
                       "chargeable": bool(struck)}
        return True

    def end(self, frame):
        """Tapping END TIME OUT: this is where the charge is written.

        Returns (player, frames elapsed, charged). A time-out over the allowance
        is still recorded - the record should show what happened, and whether a
        team over-used their allowance is a conversation for the two captains,
        not something a scoreboard should refuse mid-match.
        """
        if self.active is None:
            return None
        run = self.active
        self.active = None
        charged = run["chargeable"]
        if charged:
            self.used[run["player"]] += 1
        return run["player"], max(0, frame - run["frame"]), charged

    def cancel(self):
        """The mis-tap path. Nothing is written and nothing is charged."""
        if self.active is None:
            return None
        run = self.active
        self.active = None
        return run["player"]

    def elapsed(self, frame, fps):
        if self.active is None:
            return 0.0
        return max(0.0, (frame - self.active["frame"]) / max(1.0, fps))


class Analytics:
    """M-11's three rates, accumulated in every mode but shown in practice.

    THE THREE ACRONYMS ARE NOT DEFINED IN THE CLIENT'S SUITE. M-11 names TSR,
    CRR and SDR and stops there, so the definitions below are ours and should be
    confirmed before anyone reports a number from them to a player. They are
    written out in full on the HUD for exactly that reason - a rate whose
    definition nobody can state is worse than no rate.

      TSR  Table Success Rate      shots that legally pocketed a ball / shots
      CRR  Cue-ball Retention Rate shots that left the cue ball on the table /
                                   shots, i.e. one minus the scratch rate
      SDR  Shot Defence Rate       shots the shooter marked as defence / shots

    Accumulated per player, and in practice mode that is the whole scoreboard:
    no fouls, no turns, no win conditions, just the three rates going up.
    """

    def __init__(self):
        self.shots = [0, 0]
        self.made = [0, 0]
        self.kept_cue = [0, 0]
        self.defence = [0, 0]

    def shot(self, player, made, scratched):
        self.shots[player] += 1
        if made:
            self.made[player] += 1
        if not scratched:
            self.kept_cue[player] += 1

    def defended(self, player):
        self.defence[player] += 1

    def _rate(self, top, i):
        return top[i] / self.shots[i] if self.shots[i] else 0.0

    def tsr(self, i):
        return self._rate(self.made, i)

    def crr(self, i):
        return self._rate(self.kept_cue, i)

    def sdr(self, i):
        return self._rate(self.defence, i)

    def summary(self, i):
        return {"shots": self.shots[i], "TSR": round(self.tsr(i), 3),
                "CRR": round(self.crr(i), 3), "SDR": round(self.sdr(i), 3)}


class BaseGame:
    """What both disciplines share: players, turns, innings, the record.

    M-15 sets the identity model and it is as small as it can be: Player 1 and
    Player 2 by which side of the table they stand on, names optional, and no
    face recognition ever. This is going into public bars, and the cheapest way
    to keep that promise is to have nothing to keep.
    """

    discipline = EIGHT_BALL

    def __init__(self, players=("Player 1", "Player 2"), mode=CASUAL,
                 levels=(None, None), defence_marking=False,
                 pocket_marking=None, chart=None, fps=25.0):
        self.players = [str(p) for p in players]
        self.mode = mode if mode in MODES else CASUAL
        self.levels = [skill_level.clamp(l) for l in levels]
        self.defence_marking = bool(defence_marking)
        self.chart = chart or skill_level.RaceChart.load()
        self.fps = fps if fps and fps > 0 else 25.0

        # W-03: pocket marking defaults ON for league, OFF for casual and
        # practice. Calling pockets on a bar table for a retail demo is the
        # fastest way to make the system feel like an argument.
        self.pocket_marking = (mode == LEAGUE if pocket_marking is None
                               else bool(pocket_marking))

        self.record = MatchRecord()
        self.timeouts = TimeOuts(self.levels)
        self.analytics = Analytics()

        self.turn = 0
        self.breaker = 0
        self.rack = 1
        self.shot = 0
        self.inning = 1
        self.racks_won = [0, 0]
        self.fouls = [0, 0]
        self.defences = [0, 0]
        self.ball_in_hand = None
        self.struck = False          # has this rack been broken yet?
        self.over = False
        self.winner = None
        self.status = "RACK THEM UP"
        self.status_rule = None
        self.status_confidence = None
        self.frame = 0

        # H-01: innings are anchored on the breaker, so the count only moves
        # once the breaker has actually been to the table.
        self._breaker_shot = False
        # H-02: the shot a MARK DEFENCE tap would attach to, and nothing else.
        self.defence_shot = None
        self.marked_defence = set()
        # W-02: a rack-ending auto-call gets a prominent window of its own.
        self.undo_until = None
        self.called_pocket = None

    # ---- naming and display ------------------------------------------------

    @property
    def handicapped(self):
        """H-03. Is a Skill Level allowed to affect this match at all?"""
        return skill_level.applies(self.mode, self.defence_marking)

    def level_text(self, i):
        """H-04-compliant. Empty outside a match where the level applies."""
        return skill_level.short(self.levels[i]) if self.handicapped else ""

    def race(self):
        """(target, target) in this discipline, or None when levels do not apply."""
        if not self.handicapped:
            return None
        return self.chart.race(self.discipline, self.levels[0], self.levels[1])

    def name(self, i):
        return self.players[i].upper()

    @property
    def clock(self):
        secs = int(self.frame / self.fps)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    # ---- the record --------------------------------------------------------

    def note(self, kind, text, rule=None, confidence=None, actor="system",
             player=None, **data):
        self.status = text
        self.status_rule = rule
        self.status_confidence = confidence
        return self.record.add(self.frame, self.clock, self.rack, self.shot,
                               kind, text, rule=rule, confidence=confidence,
                               actor=actor, player=player, data=data)

    def review(self, text, rule="F-09", confidence=None, **data):
        """F-09's output: no call, a flag, and the footage kept.

        The clip pointer is the frame, which the event carries already, so a
        review costs one log line and buys a dispute that can actually be
        settled by looking at it.
        """
        return self.note("review", f"REVIEW - {text}", rule=rule,
                         confidence=confidence, **data)

    # ---- M-03 --------------------------------------------------------------

    def snapshot(self):
        """Everything the rules mutate, for undo. Subclasses extend this."""
        return {"turn": self.turn, "inning": self.inning, "shot": self.shot,
                "fouls": list(self.fouls), "defences": list(self.defences),
                "ball_in_hand": self.ball_in_hand, "over": self.over,
                "winner": self.winner, "status": self.status,
                "racks_won": list(self.racks_won), "struck": self.struck,
                "breaker_shot": self._breaker_shot,
                "marked_defence": set(self.marked_defence),
                "timeout_used": list(self.timeouts.used)}

    def restore(self, blob):
        self.turn = blob["turn"]
        self.inning = blob["inning"]
        self.shot = blob["shot"]
        self.fouls = list(blob["fouls"])
        self.defences = list(blob["defences"])
        self.ball_in_hand = blob["ball_in_hand"]
        self.over = blob["over"]
        self.winner = blob["winner"]
        self.status = blob["status"]
        self.racks_won = list(blob["racks_won"])
        self.struck = blob["struck"]
        self._breaker_shot = blob["breaker_shot"]
        self.marked_defence = set(blob["marked_defence"])
        self.timeouts.used = list(blob["timeout_used"])

    def undo(self):
        """The back button. Either player, no PIN (M-03)."""
        blob = self.record.undo(self.frame, self.clock, self.rack, self.shot)
        if blob is None:
            return False
        self.restore(blob)
        self.undo_until = None
        self.status = "UNDONE - " + self.status
        return True

    def confirm_final(self):
        return self.record.confirm_final(self.frame, self.clock, self.rack,
                                         self.shot)

    def reopen(self, who="player"):
        return self.record.reopen(self.frame, self.clock, self.rack, self.shot,
                                  who)

    # ---- turns and innings -------------------------------------------------

    def _set_turn(self, player):
        """H-01: the inning ticks when the BREAKER comes back to the table.

        Not when both players have shot, and not on a fixed alternation. The
        client's ruling is that this matches how a scoresheet reads at the table
        - "you break, we are in inning 1" - and it happens to be the easier
        thing for a camera to follow, because it anchors on the one event in the
        rack that is unmistakable.
        """
        if player == self.turn:
            return
        if player == self.breaker and self._breaker_shot:
            self.inning += 1
        self.turn = player

    def _pass_turn(self, shooter):
        self._set_turn(1 - shooter)

    # ---- player-called rules ----------------------------------------------

    def mark_defence(self, shot=None):
        """H-02. Binds to a specific shot, or refuses.

        The button only exists in the window between a completed shot and the
        next one, and the tap is written against THAT shot's number. That is
        what removes the "which shot did that tap mean" ambiguity - and it is
        why there is no persistent defence button anywhere in this system.

        Either player may tap it. The shooter self-marks in practice, but the
        opponent conceding "that was a good safe" is a real thing that happens
        at a table and there is no reason to make it impossible.
        """
        shot = self.defence_shot if shot is None else shot
        if shot is None or shot != self.defence_shot:
            return False
        if shot in self.marked_defence:
            return False
        player = self._shot_owner.get(shot, self.turn)
        self.marked_defence.add(shot)
        self.defences[player] += 1
        self.analytics.defended(player)
        self.note("defence", f"DEFENCE - {self.name(player)}", rule="H-02",
                  actor="player", player=player, shot_marked=shot)
        return True

    @property
    def defence_open(self):
        """Is the MARK DEFENCE button on screen right now? (H-02)"""
        return (self.defence_marking and self.defence_shot is not None
                and self.defence_shot not in self.marked_defence
                and not self.over)

    def mark_foul(self, why="TOUCHED BALL", rule="F-06"):
        """F-06 and friends: a foul only a person can call.

        Subject to F-10 - the window shuts when the next stroke begins, so this
        can never rewrite a shot that is already two shots old.
        """
        if not self.record.foul_window_open(self.shot):
            self.note("note", "TOO LATE - THE NEXT SHOT HAS STARTED",
                      rule="F-10")
            return False
        shooter = self._shot_owner.get(self.shot, self.turn)
        self._foul(shooter, why, rule=rule, actor="player")
        return True

    def stalemate(self):
        """W-09. Player-initiated only, and never auto-declared.

        The rack is voided: innings and defensive shots for it do not count,
        which is why they are rolled back here rather than merely stopped.
        """
        if self.over:
            return False
        for shot in list(self.marked_defence):
            player = self._shot_owner.get(shot, 0)
            self.defences[player] = max(0, self.defences[player] - 1)
        self.marked_defence.clear()
        self.inning = 1
        self._breaker_shot = False
        self.over = True
        self.winner = None
        self.note("stalemate", "STALEMATE - RACK VOID", rule="W-09",
                  actor="player")
        return True

    def call_pocket(self, pocket):
        """W-03. Only meaningful when pocket marking is on."""
        if not self.pocket_marking:
            return False
        self.called_pocket = pocket
        self.note("note", f"8 CALLED - {str(pocket).upper()}", rule="W-03",
                  actor="player", pocket=pocket)
        return True

    # ---- time-outs ---------------------------------------------------------

    def start_timeout(self):
        if self.over or not self.timeouts.start(self.turn, self.frame,
                                                self.clock, self.struck):
            return False
        self.status = f"TIME OUT - {self.name(self.turn)}"
        return True

    def end_timeout(self):
        """Commit. This is the only path that writes a charge (H-05)."""
        done = self.timeouts.end(self.frame)
        if done is None:
            return False
        player, frames, charged = done
        secs = frames / self.fps
        left = self.timeouts.remaining(player)
        text = (f"TIME OUT - {self.name(player)} - {secs:.0f}s"
                if charged else
                f"TIME OUT - {self.name(player)} - {secs:.0f}s - NOT CHARGED")
        self.note("timeout", text, rule="H-05", actor="player", player=player,
                  seconds=round(secs, 1), charged=charged, remaining=left)
        return True

    def cancel_timeout(self):
        """The mis-tap path. Nothing written, nothing charged (H-05)."""
        if self.timeouts.cancel() is None:
            return False
        self.status = "TIME OUT CANCELLED"
        return True

    # ---- the shot ----------------------------------------------------------

    _shot_owner = {}

    def shot_started(self, frame):
        """A stroke has begun. Closes the previous shot's windows (F-10, H-02)."""
        self.frame = frame
        if self.over:
            return
        self.shot += 1
        self._shot_owner = dict(getattr(self, "_shot_owner", {}))
        self._shot_owner[self.shot] = self.turn
        self.ball_in_hand = None
        self.defence_shot = None
        self.undo_until = None
        self.record.stroke_started(self.shot)
        self.record.snapshot(self.snapshot(), f"SHOT {self.shot}")
        if self.turn == self.breaker:
            self._breaker_shot = True

    def shot_ended(self, shot_facts, frame):
        """Judge one shot. Implemented per discipline."""
        raise NotImplementedError

    def _finish_shot(self, shooter, made, scratched):
        """Bookkeeping every shot gets, whatever the verdict was."""
        self.analytics.shot(shooter, made, scratched)
        self.defence_shot = self.shot if self.defence_marking else None
        self.record.stroke_ended(self.shot)

    # ---- outcomes ----------------------------------------------------------

    def _foul(self, shooter, why, rule="F-01", where=ANYWHERE,
              actor="system", confidence=None):
        """F-01's shape for every foul: a counter, a turn, and ball in hand.

        Never a point deduction. The client was explicit and the reasoning is
        sound - deductions are not a pool rule, and a scoreboard that invents one
        confuses players who know the game.
        """
        self.fouls[shooter] += 1
        self.ball_in_hand = where
        self._pass_turn(shooter)
        tail = ("BALL IN HAND" if where == ANYWHERE
                else "BALL IN HAND BEHIND THE HEAD STRING")
        self.note("foul", f"FOUL - {why} - {tail}", rule=rule,
                  confidence=confidence, actor=actor, player=shooter)

    def _pass(self, shooter, why):
        self._pass_turn(shooter)
        self.note("turn", f"{why} - {self.name(self.turn)} TO SHOOT",
                  rule="M-01")

    def _win_rack(self, shooter, why, rule=None, undo_seconds=None):
        self.over, self.winner = True, shooter
        self.racks_won[shooter] += 1
        if undo_seconds:
            self.undo_until = self.frame + int(undo_seconds * self.fps)
        self.note("win", f"{self.name(shooter)} WINS - {why}", rule=rule,
                  player=shooter)

    def _lose_rack(self, shooter, why, rule=None, undo_seconds=None):
        self._win_rack(1 - shooter, why, rule=rule, undo_seconds=undo_seconds)

    # ---- racks -------------------------------------------------------------

    def new_rack(self, frame=0):
        """A fresh triangle. The previous rack is now history (M-03)."""
        self.frame = frame
        self.record.lock_rack(frame, self.clock, self.rack, self.shot)
        self.rack += 1
        self.record.open_rack()
        self.fouls = [0, 0]
        self.ball_in_hand = None
        self.struck = False
        self.over = False
        self.winner = None
        self.inning = 1
        self._breaker_shot = False
        self.marked_defence.clear()
        self.defence_shot = None
        self.undo_until = None
        self.called_pocket = None
        self.timeouts = TimeOuts(self.levels)
        self.breaker = 1 - self.breaker
        self.turn = self.breaker
        self._reset_rack()
        self.note("rack", f"RACK {self.rack} - {self.name(self.turn)} BREAKS",
                  rule="M-09")

    def _reset_rack(self):
        pass

    def summary(self):
        raise NotImplementedError


class EightBallGame(BaseGame):
    """Eight-ball: W-01 to W-11, F-01 to F-10, M-10.

    The rule this class exists to get right is W-06, which the client flagged as
    "the single most common scoring-app bug": shooting at the 8 and missing it
    entirely is a foul and ball in hand, NOT a loss of rack. It sits one line
    away from W-05 - scratching while shooting at the 8, which IS a loss - and
    the two are easy to collapse into one branch by accident. They are kept
    visibly apart below, and each cites its row.
    """

    discipline = EIGHT_BALL

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.group = [None, None]
        self.potted = Counter()
        self.groups_confirmed = True     # M-06 lowers this when colour is shaky

    # ---- reading -----------------------------------------------------------

    @property
    def open_table(self):
        return self.group[0] is None

    def score(self, i):
        return self.potted[self.group[i]] if self.group[i] else 0

    def remaining(self, i):
        return GROUP_SIZE - self.score(i) if self.group[i] else GROUP_SIZE

    def on_the_eight(self, i):
        return self.group[i] is not None and self.score(i) >= GROUP_SIZE

    def group_name(self, i):
        if self.group[i] is None:
            return "OPEN"
        return "STRIPES" if self.group[i] == STRIPE else "SOLIDS"

    def snapshot(self):
        blob = super().snapshot()
        blob.update({"group": list(self.group), "potted": Counter(self.potted),
                     "groups_confirmed": self.groups_confirmed})
        return blob

    def restore(self, blob):
        super().restore(blob)
        self.group = list(blob["group"])
        self.potted = Counter(blob["potted"])
        self.groups_confirmed = blob["groups_confirmed"]

    def _reset_rack(self):
        self.group = [None, None]
        self.potted.clear()
        self.groups_confirmed = True

    # ---- judging -----------------------------------------------------------

    def shot_ended(self, f, frame):
        self.frame = frame
        if self.over:
            return
        shooter = self.turn
        scratch = CUE in f.pots or CUE in f.off_table
        eight_down = EIGHT in f.pots
        eight_off = EIGHT in f.off_table
        objects = [c for c in f.pots if c in (STRIPE, SOLID)]
        was_on_eight = self.on_the_eight(shooter)      # BEFORE this shot's pots

        # M-11: practice has no fouls, no turns and no win conditions. The
        # analytics still accumulate, and that is the whole scoreboard.
        if self.mode == PRACTICE:
            for cls in f.pots:
                self._rack_down(cls)
            self._finish_shot(shooter, bool(objects or eight_down), scratch)
            self.note("pot" if f.pots else "turn",
                      f"{self.name(shooter)} - PRACTICE", rule="M-11")
            return

        for cls in f.pots:
            self._rack_down(cls)

        if not self.struck:
            self._judge_break(shooter, f, objects, eight_down, eight_off,
                              scratch)
            self._finish_shot(shooter, bool(objects), scratch)
            return

        # The 8 ends the rack whichever way it goes, so it is judged first and
        # nothing after it matters.
        if eight_down or eight_off or was_on_eight:
            if self._judge_eight(shooter, f, eight_down, eight_off, scratch,
                                 was_on_eight, objects):
                self._finish_shot(shooter, bool(objects or eight_down), scratch)
                return

        foul = self._judge_fouls(shooter, f, scratch, bool(f.pots))
        if foul:
            self._finish_shot(shooter, bool(objects), scratch)
            return

        if self.open_table:
            # W-08: the table stays open until a ball is legally pocketed on a
            # NON-BREAK shot, which is this branch and only this branch.
            if objects:
                self._assign(shooter, objects[0])
            elif f.pots:
                self.note("pot", f"{self.name(shooter)} POTTED A BALL")
            else:
                self._pass(shooter, "MISS")
            self._finish_shot(shooter, bool(objects), scratch)
            return

        mine = sum(1 for c in objects if c == self.group[shooter])
        theirs = len(objects) - mine
        if theirs and not mine:
            # Bar rules treat potting only an opponent's ball as a foul. The
            # ball stays down and still counts for its owner.
            self._foul(shooter, "WRONG GROUP POTTED", rule="F-03")
        elif mine or UNKNOWN in f.pots:
            verb = "ON THE 8" if self.on_the_eight(shooter) else "CONTINUES"
            self.note("pot", f"{self.name(shooter)} {verb}")
        else:
            self._pass(shooter, "MISS")
        self._finish_shot(shooter, bool(mine), scratch)

    def _judge_break(self, shooter, f, objects, eight_down, eight_off, scratch):
        """The break: F-02, M-10, and W-08's "not on the break" clause."""
        self.struck = True

        if eight_down or eight_off:
            # The 8 on the break re-racks rather than deciding anything.
            self._rerack()
            return

        if scratch:
            # F-02: the one restricted ball-in-hand in the whole suite.
            self._foul(shooter, "SCRATCH ON THE BREAK", rule="F-02",
                       where=BEHIND_HEAD_STRING)
            return

        legal = self._break_legal(f, objects)
        if legal is False:
            if self.mode == LEAGUE:
                self._foul(shooter, "ILLEGAL BREAK", rule="M-10")
                return
            self.note("note", "ILLEGAL BREAK - NOT ENFORCED IN CASUAL PLAY",
                      rule="M-10")

        if objects:
            # W-08 again: a ball down on the break does NOT assign a group.
            # The table stays open and the breaker chooses on their next legal
            # pot, which is the locked behaviour and differs from bar practice.
            self.note("break", f"{self.name(shooter)} BROKE - TABLE OPEN",
                      rule="W-08")
        else:
            self._pass(shooter, "DRY BREAK")

    def _break_legal(self, f, objects):
        """M-10: four balls to a rail, or a ball pocketed. None = could not tell."""
        if objects:
            return True
        if f.rail_contact is None:
            return None
        return f.rail_contact and f.moved >= BREAK_RAIL_BALLS

    def _judge_eight(self, shooter, f, eight_down, eight_off, scratch,
                     was_on_eight, objects):
        """W-01 to W-07. Returns True when the rack has been decided.

        Ordered so the losses that people argue about are unmistakable, and each
        branch carries its row.
        """
        # W-07: off the table is a loss however it got there.
        if eight_off:
            self._lose_rack(shooter, "8 KNOCKED OFF THE TABLE", rule="W-07")
            return True

        # W-05: scratching while shooting at the 8. Distinct from W-06 below,
        # and the distinction is the entire rule.
        if was_on_eight and scratch:
            self._lose_rack(shooter, "SCRATCHED ON THE 8", rule="W-05")
            return True

        if eight_down:
            if not was_on_eight:
                # W-04 and W-02 are the same outcome from different causes, and
                # the log should say which happened.
                cleared = self.on_the_eight(shooter)
                if cleared:
                    self._lose_rack(shooter, "8 WITH THE LAST GROUP BALL",
                                    rule="W-04")
                else:
                    self._lose_rack(shooter, "8 POTTED EARLY", rule="W-02",
                                    undo_seconds=rules.EIGHT_EARLY_UNDO_SECONDS)
                return True
            if self.open_table:
                self._lose_rack(shooter, "8 POTTED ON AN OPEN TABLE",
                                rule="W-02",
                                undo_seconds=rules.EIGHT_EARLY_UNDO_SECONDS)
                return True
            # W-03: the wrong pocket only matters when marking is enabled.
            if (self.pocket_marking and self.called_pocket
                    and f.pots and self.called_pocket != getattr(
                        f, "eight_pocket", self.called_pocket)):
                self._lose_rack(shooter, "8 IN THE WRONG POCKET", rule="W-03")
                return True
            self._win_rack(shooter, "RAN THE TABLE", rule="W-01")
            return True

        if was_on_eight:
            # W-06. Shooting at the 8 and missing it is a FOUL, not a loss -
            # the client calls this the commonest scoring-app bug and it is one
            # branch away from W-05 above. Do not merge them.
            if f.hit_object is False:
                self._foul(shooter, "MISSED THE 8 ENTIRELY", rule="W-06")
                return True
            if f.rail_contact is False and not f.pots:
                self._foul(shooter, "NO RAIL AFTER CONTACT", rule="F-04")
                return True
            if objects:
                # Potting a group ball while on the 8 means a miscount upstream,
                # not a rule: the group was supposed to be clear. Flag it.
                self.review("GROUP BALL POTTED WHILE ON THE 8", rule="M-14")
            return False

        return False

    def _judge_fouls(self, shooter, f, scratch, made_something):
        """F-01, F-03, F-04, F-05, F-09. Returns True when a foul was called."""
        if scratch:
            self._foul(shooter, "SCRATCH", rule="F-01")
            return True

        # F-05: the easiest foul for a camera. Nothing but the cue ball moved,
        # so nothing was hit.
        if f.hit_object is False:
            self._foul(shooter, "NO BALL HIT", rule="F-05")
            return True

        # F-03: wrong group first, but only at the client's confidence bar.
        # Below it, F-09: no call, a review, and the clip kept.
        if (f.first_contact in (STRIPE, SOLID) and not self.open_table
                and f.first_contact != self.group[shooter]):
            if f.contact_conf >= rules.CONTACT_CONFIDENCE:
                self._foul(shooter, "WRONG GROUP HIT FIRST", rule="F-03",
                           confidence=f.contact_conf)
                return True
            self.review("FIRST CONTACT UNCERTAIN", rule="F-09",
                        confidence=f.contact_conf)
            return False

        # F-04: no rail and nothing pocketed after contact.
        if f.rail_contact is False and not made_something and f.hit_object:
            self._foul(shooter, "NO RAIL AFTER CONTACT", rule="F-04")
            return True

        # F-09 in its general form: contact could not be established at all.
        if f.hit_object is None and f.contact_conf < rules.CONTACT_CONFIDENCE:
            self.review("CONTACT NOT ESTABLISHED", rule="F-09",
                        confidence=f.contact_conf)
        return False

    # ---- helpers -----------------------------------------------------------

    def _rack_down(self, cls):
        """Put a ball down, refusing counts the rack cannot hold.

        Class votes flicker, so an eighth stripe is a misread rather than a
        ball. Clamping keeps the scoreboard honest instead of showing 8/7.
        """
        if cls == CUE:
            return
        cap = 1 if cls == EIGHT else GROUP_SIZE
        if cls in (STRIPE, SOLID, EIGHT) and self.potted[cls] >= cap:
            self.note("impossible", f"IGNORED AN EXTRA {cls.upper()}",
                      rule="M-14", confidence=0.0)
            return
        self.potted[cls] += 1

    def _assign(self, shooter, cls):
        """W-08. Groups are inferred here and nowhere else."""
        self.group[shooter] = cls
        self.group[1 - shooter] = OTHER[cls]
        self.note("assign", f"{self.name(shooter)} TAKES "
                            f"{self.group_name(shooter)}", rule="W-08")

    def needs_group_confirmation(self, confidence):
        """M-06: ask once, never guess. True when the HUD should prompt."""
        return self.open_table and confidence < rules.CONTACT_CONFIDENCE

    def confirm_groups(self, shooter, cls):
        """The answer to the M-06 prompt, from a tap."""
        self._assign(shooter, cls)
        self.groups_confirmed = True
        self.note("assign", f"GROUPS CONFIRMED - {self.name(shooter)} ON "
                            f"{self.group_name(shooter)}", rule="M-06",
                  actor="player")

    def _rerack(self):
        self.group = [None, None]
        self.potted.clear()
        self.struck = False
        self.note("rerack", "8 ON THE BREAK - RE-RACK", rule="W-02")

    def summary(self):
        return {"discipline": EIGHT_BALL, "mode": self.mode,
                "players": self.players,
                "levels": [skill_level.describe(l) for l in self.levels]
                          if self.handicapped else ["Player", "Player"],
                "groups": [self.group_name(0), self.group_name(1)],
                "score": [self.score(0), self.score(1)],
                "racks": list(self.racks_won),
                "race": self.race(),
                "fouls": list(self.fouls),
                "defences": list(self.defences),
                "timeouts": list(self.timeouts.used),
                "shots": self.shot, "innings": self.inning,
                "analytics": [self.analytics.summary(0),
                              self.analytics.summary(1)],
                "winner": None if self.winner is None
                          else self.players[self.winner],
                "status": self.status}


class NineBallGame(BaseGame):
    """Nine-ball: N-01 to N-04, on top of the shared fouls.

    Scored in POINTS, not racks - one per ball, two for the 9 - and the rule
    that had to be corrected is N-04: a ball made during a scratch or a foul is
    DEAD and is credited to NEITHER player. An earlier draft gave them to the
    opponent, which is wrong and inflates one side's score in a way nobody
    notices until a league night ends in an argument.

    The invariant that catches it is checked at the end of every rack:

        player 1 points + player 2 points + dead balls == 10

    Ten because balls 1 to 8 are worth a point each and the 9 is worth two. If
    that sum is ever anything else, something has been credited twice or lost,
    and the rack says so in the log rather than quietly being wrong.

    On identifying balls: the 9 is the only stripe in a nine-ball rack, which
    the existing classifier reads well. The numbers on 1 to 8 are not readable
    at this frame rate, so N-01 - lowest ball first - degrades to F-09 unless
    numbers are supplied, and is never guessed at.
    """

    discipline = NINE_BALL

    #: A rack is worth this many points, and the invariant is built on it.
    RACK_POINTS = 10

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.points = [0, 0]
        self.dead = 0
        self.rack_points = [0, 0]
        self.balls_down = 0
        self.lowest = None      # the lowest number still on the table, if known

    def score(self, i):
        return self.points[i]

    def group_name(self, i):
        """Nine-ball has no groups; the HUD reuses this slot for the race."""
        target = self.race()
        return f"TO {target[i]}" if target else "9-BALL"

    def snapshot(self):
        blob = super().snapshot()
        blob.update({"points": list(self.points), "dead": self.dead,
                     "rack_points": list(self.rack_points),
                     "balls_down": self.balls_down, "lowest": self.lowest})
        return blob

    def restore(self, blob):
        super().restore(blob)
        self.points = list(blob["points"])
        self.dead = blob["dead"]
        self.rack_points = list(blob["rack_points"])
        self.balls_down = blob["balls_down"]
        self.lowest = blob["lowest"]

    def _reset_rack(self):
        self._check_invariant()
        self.rack_points = [0, 0]
        self.dead = 0
        self.balls_down = 0
        self.lowest = None

    def _check_invariant(self):
        """N-04's arithmetic, checked out loud rather than trusted."""
        total = self.rack_points[0] + self.rack_points[1] + self.dead
        if self.balls_down and total != self.RACK_POINTS:
            self.note("impossible",
                      f"RACK DOES NOT ADD UP - {self.rack_points[0]} + "
                      f"{self.rack_points[1]} + {self.dead} DEAD = {total}, "
                      f"EXPECTED {self.RACK_POINTS}",
                      rule="N-04", confidence=0.0)
            return False
        return True

    def _value(self, cls):
        """1 point a ball, 2 for the 9 (N-04)."""
        return 2 if self._is_nine(cls) else 1

    @staticmethod
    def _is_nine(cls):
        return cls == STRIPE if NINE_IS_STRIPE else cls == "nine"

    def _credit(self, shooter, classes, fouled):
        """N-04. Balls on a foul shot are dead; the 9 is never dead.

        Returns True when the 9 went down legally, which is the only way the
        rack is won.
        """
        nine_legal = False
        for cls in classes:
            if cls == CUE:
                continue
            self.balls_down += 1
            if self._is_nine(cls):
                if fouled:
                    # N-03: spotted, not dead, not a win. The ball comes back.
                    self.balls_down -= 1
                    self.note("note", "9 SPOTTED - POTTED ON A FOUL",
                              rule="N-03")
                    continue
                self.points[shooter] += 2
                self.rack_points[shooter] += 2
                nine_legal = True
                continue
            if fouled:
                self.dead += 1
                self.note("dead", "DEAD BALL - NOT CREDITED", rule="N-04")
                continue
            self.points[shooter] += 1
            self.rack_points[shooter] += 1
        return nine_legal

    def shot_ended(self, f, frame):
        self.frame = frame
        if self.over:
            return
        shooter = self.turn
        scratch = CUE in f.pots or CUE in f.off_table
        objects = [c for c in f.pots if c != CUE]

        if self.mode == PRACTICE:
            self._credit(shooter, objects, fouled=False)
            self._finish_shot(shooter, bool(objects), scratch)
            self.note("pot" if objects else "turn",
                      f"{self.name(shooter)} - PRACTICE", rule="M-11")
            return

        if not self.struck:
            self._judge_break(shooter, f, objects, scratch)
            self._finish_shot(shooter, bool(objects), scratch)
            return

        foul = self._judge_fouls(shooter, f, scratch)
        nine_legal = self._credit(shooter, objects, fouled=foul)

        if foul:
            self._finish_shot(shooter, bool(objects), scratch)
            return
        if nine_legal:
            self._win_rack(shooter, "9-BALL", rule="N-02")
            self._finish_shot(shooter, True, scratch)
            return
        if objects:
            self.note("points", f"{self.name(shooter)} CONTINUES - "
                                f"{self.points[shooter]} POINTS")
        else:
            self._pass(shooter, "MISS")
        self._finish_shot(shooter, bool(objects), scratch)

    def _judge_break(self, shooter, f, objects, scratch):
        """N-02: the 9 on the snap wins, unless the shooter also scratched."""
        self.struck = True
        nine = [c for c in objects if self._is_nine(c)]
        if scratch:
            self._credit(shooter, objects, fouled=True)
            self._foul(shooter, "SCRATCH ON THE BREAK", rule="F-02",
                       where=BEHIND_HEAD_STRING)
            return
        nine_legal = self._credit(shooter, objects, fouled=False)
        if nine and nine_legal:
            self._win_rack(shooter, "9 ON THE SNAP", rule="N-02")
            return
        if objects:
            self.note("points", f"{self.name(shooter)} BROKE AND CONTINUES")
        else:
            self._pass(shooter, "DRY BREAK")

    def _judge_fouls(self, shooter, f, scratch):
        """F-01, F-04, F-05 and N-01. Returns True when a foul was called."""
        if scratch:
            self._foul(shooter, "SCRATCH", rule="F-01")
            return True
        if f.hit_object is False:
            self._foul(shooter, "NO BALL HIT", rule="F-05")
            return True

        # N-01: lowest-numbered ball first. Enforceable only when a number is
        # actually known - the classifier reads the 9 (the only stripe) but not
        # the numbers on 1 to 8, so the usual case falls through to F-09 and is
        # logged as a review instead of being guessed.
        if self.lowest is not None and f.first_contact is not None:
            if (f.contact_conf >= rules.CONTACT_CONFIDENCE
                    and f.first_contact != self.lowest):
                self._foul(shooter, f"{self.lowest} NOT HIT FIRST",
                           rule="N-01", confidence=f.contact_conf)
                return True
            if f.contact_conf < rules.CONTACT_CONFIDENCE:
                self.review("LOWEST BALL CONTACT UNCERTAIN", rule="F-09",
                            confidence=f.contact_conf)
                return False
        elif f.first_contact is None:
            self.review("BALL NUMBERS NOT READABLE - N-01 NOT ENFORCED",
                        rule="F-09", confidence=f.contact_conf)

        if f.rail_contact is False and not f.pots and f.hit_object:
            self._foul(shooter, "NO RAIL AFTER CONTACT", rule="F-04")
            return True
        return False

    def summary(self):
        self._check_invariant()
        return {"discipline": NINE_BALL, "mode": self.mode,
                "players": self.players,
                "levels": [skill_level.describe(l) for l in self.levels]
                          if self.handicapped else ["Player", "Player"],
                "points": list(self.points),
                "rack_points": list(self.rack_points),
                "dead": self.dead,
                "invariant_ok": (self.rack_points[0] + self.rack_points[1]
                                 + self.dead == self.RACK_POINTS
                                 if self.balls_down else True),
                "racks": list(self.racks_won),
                "race": self.race(),
                "fouls": list(self.fouls),
                "defences": list(self.defences),
                "timeouts": list(self.timeouts.used),
                "shots": self.shot, "innings": self.inning,
                "analytics": [self.analytics.summary(0),
                              self.analytics.summary(1)],
                "winner": None if self.winner is None
                          else self.players[self.winner],
                "status": self.status}


def make_game(discipline=EIGHT_BALL, **kw):
    """The one place a discipline string turns into a rules engine."""
    return (NineBallGame(**kw) if discipline == NINE_BALL
            else EightBallGame(**kw))
