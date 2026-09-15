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

import copy
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

#: Whose turn ending closes an inning. SCORESHEET is the league manual's rule
#: (section 5): the lag loser is the bottom half of every inning, all match, and
#: an inning is marked only when their turn ends. BREAKER is the client CSV's
#: H-01 reading, kept as a setting. The client chose the manual.
SCORESHEET_INNINGS, BREAKER_INNINGS = "scoresheet", "breaker"

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
    "moved numbers rail_balls eight_pocket off_conf",
)


def facts(pots=(), off_table=(), rail_contact=None, hit_object=None,
          first_contact=None, contact_conf=0.0, moved=0, numbers=(),
          rail_balls=None, eight_pocket=None, off_conf=1.0):
    """Build a ShotFacts. `rail_balls` is M-10's count of distinct object balls
    that reached a cushion; `eight_pocket` is where the 8 went, for W-03;
    `off_conf` is how sure an off-the-table call is, for W-07 and M-14."""
    return ShotFacts(list(pots), list(off_table), rail_contact, hit_object,
                     first_contact, float(contact_conf), int(moved),
                     list(numbers), rail_balls, eight_pocket, float(off_conf))


def merge(f, add=(), remove=()):
    """The same shot with pots added (M-02, W-11) or taken away (M-04, W-10)."""
    pots = list(f.pots)
    for cls in remove:
        if cls in pots:
            pots.remove(cls)
    pots.extend(add)
    return f._replace(pots=pots)


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

    def __init__(self, levels=(None, None), fmt=skill_level.OPEN):
        self.allowance = [skill_level.timeouts(l, fmt) for l in levels]
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
                 pocket_marking=None, chart=None, fps=25.0,
                 fmt=skill_level.OPEN, break_assigns=True,
                 innings_rule=SCORESHEET_INNINGS, lag_winner=0):
        self.players = [str(p) for p in players]
        self.mode = mode if mode in MODES else CASUAL
        self.fmt = fmt if fmt in skill_level.FORMATS else skill_level.OPEN
        self.levels = [skill_level.clamp(l, self.discipline) for l in levels]
        # Manual 3.4c: balls of one category down on the break take that group.
        # The CSV's W-08 said never; the client chose the manual, and the old
        # behaviour stays one switch away.
        self.break_assigns = bool(break_assigns)
        self.innings_rule = innings_rule
        # Manual 3.1: the lag winner breaks first and is the TOP of every
        # inning on the scoresheet for the whole match.
        self.lag_winner = lag_winner if lag_winner in (0, 1) else 0
        self.defence_marking = bool(defence_marking)
        self.chart = chart or skill_level.RaceChart.load()
        self.fps = fps if fps and fps > 0 else 25.0

        # W-03: pocket marking defaults ON for league, OFF for casual and
        # practice. Calling pockets on a bar table for a retail demo is the
        # fastest way to make the system feel like an argument.
        self.pocket_marking = (mode == LEAGUE if pocket_marking is None
                               else bool(pocket_marking))

        self.record = MatchRecord()
        self.timeouts = TimeOuts(self.levels, self.fmt)
        self.analytics = Analytics()

        self.turn = self.lag_winner
        self.breaker = self.lag_winner
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
        # W-09: a voided rack's innings do not count toward the match total.
        self.voided = False
        self.innings_total = 0
        # H-03: with Skill Levels in play, the race decides the match.
        self.match_winner = None
        # M-10 in league: waiting on the incoming player's choice.
        self.illegal_break = False
        # Manual 16d: the player whose 8 went into an unmarked pocket, until
        # the opponent either calls loss of game or lets it stand.
        self.unmarked_claim = None
        # Manual 3.13: a stalemated rack is re-broken by the same breaker.
        self.stalemated = False

    # ---- naming and display ------------------------------------------------

    @property
    def handicapped(self):
        """H-03. Is a Skill Level allowed to affect this match at all?"""
        return skill_level.applies(self.mode, self.defence_marking, self.fmt)

    def level_text(self, i):
        """H-04-compliant. Empty outside a match where the level applies."""
        return (skill_level.short(self.levels[i], self.discipline)
                if self.handicapped else "")

    def race(self):
        """(target, target), or None when no race applies.

        Masters is a straight race to 7 in league play, levels or not.
        """
        if self.fmt == skill_level.MASTERS and self.mode == LEAGUE:
            return self.chart.race(self.discipline, None, None, fmt=self.fmt)
        if not self.handicapped:
            return None
        return self.chart.race(self.discipline, self.levels[0], self.levels[1],
                               fmt=self.fmt)

    def name(self, i):
        return self.players[i].upper()

    @property
    def clock(self):
        secs = int(self.frame / self.fps)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    # ---- the record --------------------------------------------------------

    def note(self, kind, text, rule=None, confidence=None, actor="system",
             player=None, quiet=False, **data):
        """Write to the record, and to the status line unless `quiet`.

        Quiet is for bookkeeping that happens alongside a verdict - a dead ball
        on a foul shot, a review flag - which belongs in the log but must not
        overwrite the verdict on the one HUD line that says what just happened.
        """
        if not quiet:
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
                         confidence=confidence, quiet=True, **data)

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
                "timeout_used": list(self.timeouts.used),
                "analytics": copy.deepcopy(self.analytics),
                "voided": self.voided, "match_winner": self.match_winner,
                "illegal_break": self.illegal_break,
                "breaker": self.breaker,
                "unmarked_claim": self.unmarked_claim,
                "stalemated": self.stalemated}

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
        self.analytics = copy.deepcopy(blob["analytics"])
        self.voided = blob["voided"]
        self.match_winner = blob["match_winner"]
        self.illegal_break = blob["illegal_break"]
        self.breaker = blob["breaker"]
        self.unmarked_claim = blob.get("unmarked_claim")
        self.stalemated = blob.get("stalemated", False)

    def undo(self, by="player", why=None):
        """The back button. Either player, no PIN (M-03).

        `by="system"` is the silent walk-back M-02, M-04 and W-11 use to
        re-judge a shot that a late ball has changed; see MatchRecord.undo.
        """
        blob = self.record.undo(self.frame, self.clock, self.rack, self.shot,
                                by=by, why=why)
        if blob is None:
            return False
        self.restore(blob)
        self.undo_until = None
        if by == "player":
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
        """H-01 and manual section 5: when does an inning close?

        SCORESHEET (the default): the lag winner is the top of every inning and
        the lag loser the bottom, for the whole match, and an inning is marked
        only when the bottom player's turn ends by a miss or a foul. A
        break-and-run therefore marks none - "mark complete innings only" - and
        a match that ends mid-inning does not mark the one it ended in.

        BREAKER: the CSV's original reading, where the inning ticks each time
        the breaker of this rack returns to the table.

        `inning` is always the CURRENT inning, so completed innings are
        `inning - 1`, which is what the record and totals store.
        """
        if player == self.turn:
            return
        if self.innings_rule == BREAKER_INNINGS:
            if player == self.breaker and self._breaker_shot:
                self.inning += 1
        elif self.turn == 1 - self.lag_winner:
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
        if why in self.FOUL_REASONS:
            why, rule = self.FOUL_REASONS[why], "TM-15"
        self._foul(shooter, why, rule=rule, actor="player")
        return True

    #: Manual 3.15: the complete and exclusive list of ball-in-hand fouls. CALL
    #: FOUL offers exactly these - local bylaws "may not create ball-in-hand
    #: fouls", and neither may a HUD.
    FOUL_REASONS = {
        "scratch": "CUE BALL POCKETED OR OFF THE TABLE",
        "wrong-ball": "WRONG BALL HIT FIRST",
        "no-rail": "NO RAIL AFTER CONTACT",
        "frozen-rail": "FROZEN BALL RAIL RULE",
        "scoop": "INTENTIONAL SCOOP OVER A BALL",
        "coaching": "ADVICE FROM A NON-COACH",
        "touched-cue": "CUE BALL TOUCHED OUTSIDE BALL IN HAND",
        "double-hit": "DOUBLE HIT OR ALTERED CUE BALL",
        "moved-ball": "CUE BALL HIT A MOVED BALL",
        "no-contact": "NO BALL HIT",
        "bih-touch": "BALL TOUCHED DURING BALL IN HAND",
    }

    def declare_frozen(self, ball="BALL"):
        """Manual 3.14: a ball is frozen only once declared and agreed.

        F-08 defers the rail requirement to the players, and the manual agrees:
        the rule is not in effect until both players say so. This records the
        declaration; any foul that follows is called with CALL FOUL.
        """
        if self.over:
            return False
        self.note("note", f"{str(ball).upper()} DECLARED FROZEN", rule="F-08",
                  actor="player")
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
        self.voided = True
        self.stalemated = True
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

    def _win_rack(self, shooter, why, rule=None, undo_seconds=None,
                  confidence=None):
        self.over, self.winner = True, shooter
        self.racks_won[shooter] += 1
        if undo_seconds:
            self.undo_until = self.frame + int(undo_seconds * self.fps)
        self.note("win", f"{self.name(shooter)} WINS - {why}", rule=rule,
                  player=shooter, confidence=confidence)
        self._check_race()

    def _lose_rack(self, shooter, why, rule=None, undo_seconds=None,
                   confidence=None):
        self._win_rack(1 - shooter, why, rule=rule, undo_seconds=undo_seconds,
                       confidence=confidence)

    def race_progress(self, i):
        """What counts toward the race: racks in 8-ball, points in 9-ball."""
        return self.racks_won[i]

    def _check_race(self):
        """H-03: with Skill Levels in play, the race decides the match."""
        target = self.race()
        if not target or self.match_winner is not None:
            return
        for i in (0, 1):
            if self.race_progress(i) >= target[i]:
                self.match_winner = i
                self.note("match", f"{self.name(i)} WINS THE MATCH",
                          rule="H-03", player=i, quiet=True)
                return

    # ---- racks -------------------------------------------------------------

    def new_rack(self, frame=0):
        """A fresh triangle. The previous rack is now history (M-03)."""
        self.frame = frame
        self.record.lock_rack(frame, self.clock, self.rack, self.shot)
        was_void, was_winner = self.voided, self.winner
        if not was_void:
            self.innings_total += self.inning - 1
        self.voided = False
        self.illegal_break = False
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
        self.unmarked_claim = None
        self.timeouts = TimeOuts(self.levels, self.fmt)
        # Manual 3.1: the winner of each rack breaks the next. A stalemated
        # rack is re-broken by the same breaker (3.13). No result: alternate.
        if was_winner is not None:
            self.breaker = was_winner
        elif not (was_void or self.stalemated):
            self.breaker = 1 - self.breaker
        self.stalemated = False
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
        for cls in f.off_table:
            if cls in (STRIPE, SOLID):
                # Manual 3.8: an object ball on the floor is spotted when the
                # turn ends - the shooter's own ball only once the rest are down.
                self.note("note", f"{cls.upper()} OFF THE TABLE - SPOT IT WHEN "
                                  f"THE TURN ENDS", rule="TM-8", quiet=True)

        # M-11: practice has no fouls, no turns and no win conditions. The
        # analytics still accumulate, and that is the whole scoreboard.
        if self.mode == PRACTICE:
            self.struck = True
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

        foul = self._judge_fouls(shooter, f, scratch, bool(f.pots),
                                 was_on_eight)
        if foul:
            self._finish_shot(shooter, bool(objects), scratch)
            return

        if self.open_table:
            # Manual 3.4d: on an open table a legal pot of ONE category takes
            # that group; a shot that drops one of each leaves the table open.
            if len(set(objects)) == 1:
                self._assign(shooter, objects[0])
            elif objects:
                self.note("pot", f"{self.name(shooter)} POTTED ONE OF EACH - "
                                 f"TABLE STILL OPEN", rule="W-08")
            elif f.pots:
                self.note("pot", f"{self.name(shooter)} POTTED A BALL")
            else:
                self._pass(shooter, "MISS")
            self._finish_shot(shooter, bool(objects), scratch)
            return

        mine = sum(1 for c in objects if c == self.group[shooter])
        theirs = len(objects) - mine
        if theirs and not mine:
            # NOT a foul. The suite's fouls are F-01 to F-10 and potting an
            # opponent's ball is not among them - F-03 is about what is HIT
            # first, and is judged above on contact. M-01 covers this: no ball
            # of the shooter's group went down, so the turn passes. The ball
            # stays down and counts for its owner. (The first prototype called
            # this a foul on bar rules; the locked suite supersedes that.)
            self._pass(shooter, "OPPONENT'S BALL POTTED")
        elif mine or UNKNOWN in f.pots:
            verb = "ON THE 8" if self.on_the_eight(shooter) else "CONTINUES"
            self.note("pot", f"{self.name(shooter)} {verb}")
        else:
            self._pass(shooter, "MISS")
        self._finish_shot(shooter, bool(mine), scratch)

    def _judge_break(self, shooter, f, objects, eight_down, eight_off, scratch):
        """The break: manual 3.3 and 3.4, F-02 and M-10. The order is the rule.

          illegal break   the rack was struck, but fewer than four object balls
                          reached a rail and nothing dropped. Re-racked and
                          re-broken by the same player - or by the OTHER player
                          if the illegal break also scratched. Enforced in
                          league (M-10); advisory in casual, where play goes on.
          8 on the break  a win, unless the breaker fouled the cue ball, which
                          loses (3.4b). An 8 knocked off the table loses too.
          scratch         ball in hand behind the head string, first contact
                          outside it (3.4a, F-02). A foul assigns no group.
          groups          one category down takes that group; both leave the
                          table open (3.4c/d). `break_assigns` off restores the
                          CSV's W-08, where the break never assigns.
        """
        self.struck = True

        legal = self._break_legal(f, objects + ([EIGHT] if eight_down else []))
        if legal is False:
            if self.mode == LEAGUE:
                again = 1 - shooter if scratch else shooter
                self.breaker = self.turn = again
                self.inning = 1
                self._breaker_shot = False
                self._rerack(why=None)
                self.note("rerack", f"ILLEGAL BREAK - RE-RACK - "
                                    f"{self.name(again)} BREAKS", rule="M-10",
                          player=shooter)
                return
            self.note("note", "ILLEGAL BREAK - NOT ENFORCED IN CASUAL PLAY",
                      rule="M-10", quiet=True)

        if eight_down or eight_off:
            if scratch or eight_off:
                self._lose_rack(shooter, "8 ON THE BREAK WITH A FOUL",
                                rule="W-02")
            else:
                self._win_rack(shooter, "8 ON THE BREAK", rule="W-01")
            return

        if scratch:
            self._foul(shooter, "SCRATCH ON THE BREAK", rule="F-02",
                       where=BEHIND_HEAD_STRING)
            return

        if objects and self.break_assigns and len(set(objects)) == 1:
            self._assign(shooter, objects[0])
        elif objects:
            self.note("break", f"{self.name(shooter)} BROKE - TABLE OPEN",
                      rule="W-08")
        else:
            self._pass(shooter, "DRY BREAK")

    def _break_legal(self, f, objects):
        """M-10: four balls to a rail, or a ball pocketed. None = could not tell.

        Counted as distinct object balls that entered the cushion band during
        the break - the same band F-04 uses. With no table geometry there is
        nothing to count against and the answer is None, which is never
        enforced: an unprovable illegal break is not called.
        """
        if objects:
            return True
        if f.rail_balls is None:
            return None
        return f.rail_balls >= BREAK_RAIL_BALLS

    def resolve_illegal_break(self, choice):
        """M-10's league choice, from the incoming player's tap."""
        if not self.illegal_break:
            return False
        self.illegal_break = False
        if choice == "re-rack":
            self.breaker = self.turn
            self._rerack(why=None)
            self.note("rerack", f"RE-RACK - {self.name(self.turn)} BREAKS",
                      rule="M-10", actor="player")
        else:
            self.note("turn", f"PLAY ON - {self.name(self.turn)} TO SHOOT",
                      rule="M-10", actor="player")
        return True

    def _judge_eight(self, shooter, f, eight_down, eight_off, scratch,
                     was_on_eight, objects):
        """W-01 to W-07. Returns True when the rack has been decided.

        Ordered so the losses that people argue about are unmistakable, and each
        branch carries its row.
        """
        # W-07: off the table is a loss however it got there. The camera sees
        # this as the 8 vanishing away from every pocket - an inference, not an
        # observation - so it carries its confidence (M-14) and gets W-02's undo
        # window: a rack-ending call made on inference should be at least as
        # easy to take back as the one the client already asked that for.
        if eight_off:
            inferred = f.off_conf < 1.0
            self._lose_rack(shooter, "8 KNOCKED OFF THE TABLE", rule="W-07",
                            confidence=f.off_conf if inferred else None,
                            undo_seconds=(rules.EIGHT_EARLY_UNDO_SECONDS
                                          if inferred else None))
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
            # Manual 16e: fouling the cue ball while pocketing the 8 loses.
            # Wrong ball first is the foul a camera can reach, and only at
            # F-03's bar; below it the 8 stands and the shot is flagged.
            if f.first_contact in (STRIPE, SOLID):
                if f.contact_conf >= rules.CONTACT_CONFIDENCE:
                    self._lose_rack(shooter, "8 POTTED ON A FOUL", rule="F-03",
                                    confidence=f.contact_conf)
                    return True
                self.review("8 POTTED - FIRST CONTACT UNCERTAIN", rule="F-09",
                            confidence=f.contact_conf)
            # W-03: the wrong pocket only matters when marking is enabled, and
            # only when both the call and the pocket are actually known.
            if (self.pocket_marking and self.called_pocket and f.eight_pocket
                    and self.called_pocket != f.eight_pocket):
                self._lose_rack(shooter, f"8 IN THE WRONG POCKET - CALLED "
                                         f"{self.called_pocket}, WENT IN "
                                         f"{f.eight_pocket}", rule="W-03")
                return True
            self._win_rack(shooter, "RAN THE TABLE", rule="W-01")
            # Manual 16d: an 8 in an unmarked pocket loses only if the opponent
            # calls it. The system records the chance and never takes it.
            if self.pocket_marking and not self.called_pocket:
                self.unmarked_claim = shooter
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
            if any(c == self.group[shooter] for c in objects):
                # Potting one of your own group while on the 8 means a miscount
                # upstream, not a rule: the group was supposed to be clear.
                self.review("GROUP BALL POTTED WHILE ON THE 8", rule="M-14")
            return False

        return False

    def _judge_fouls(self, shooter, f, scratch, made_something,
                     was_on_eight=False):
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
        # The legal first ball is the 8 once the group is clear - judged on the
        # table as it stood BEFORE this shot's pots.
        # Manual 3.4d: on an open table any ball may be hit first except the 8.
        if (self.open_table and f.first_contact == EIGHT
                and f.contact_conf >= rules.CONTACT_CONFIDENCE):
            self._foul(shooter, "8 HIT FIRST ON AN OPEN TABLE", rule="F-03",
                       confidence=f.contact_conf)
            return True
        target = EIGHT if was_on_eight else self.group[shooter]
        if (f.first_contact in (STRIPE, SOLID, EIGHT) and not self.open_table
                and f.first_contact != target):
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

        # F-09 in its general form: contact could not be established at all -
        # usually because no cue ball was identified this shot. No foul, and a
        # review is only worth logging when there was something to review.
        if f.hit_object is None and f.moved and not made_something:
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

    def mark_foul(self, why="TOUCHED BALL", rule="F-06"):
        """A player-called foul - and on the shot that pocketed the 8, a loss.

        Manual 16e: fouling the cue ball while pocketing the 8 loses the game.
        So a double hit called on the winning shot does not merely pass the
        turn on a finished rack; it turns the win into a loss. Still inside
        F-10's window, which is the only time any foul can be called.
        """
        shooter = self._shot_owner.get(self.shot)
        if (self.over and shooter is not None and self.winner == shooter
                and self.record.foul_window_open(self.shot)):
            reason = self.FOUL_REASONS.get(why, why)
            self.racks_won[shooter] -= 1
            self.match_winner = None
            self.unmarked_claim = None
            self.fouls[shooter] += 1
            self._lose_rack(shooter, f"FOUL ON THE 8 - {reason}", rule="TM-16")
            self.record.events[-1].actor = "player"
            return True
        return super().mark_foul(why, rule)

    def swap_groups(self):
        """Manual 3.5 note: a wrong-group foul nobody called in time.

        When the sitting player lets the opponent's turn end without calling it
        and then plays the wrong category themselves, BOTH players take the new
        categories for the rest of the game. Player-called only.
        """
        if self.open_table or self.over:
            return False
        self.group = [self.group[1], self.group[0]]
        self.note("assign", f"GROUPS SWAPPED - {self.name(0)} ON "
                            f"{self.group_name(0)}", rule="TM-5", actor="player")
        return True

    def call_unmarked_loss(self):
        """Manual 16d: the opponent calls loss of game on an unmarked 8."""
        shooter = self.unmarked_claim
        if shooter is None or not self.over or self.winner != shooter:
            return False
        self.unmarked_claim = None
        self.racks_won[shooter] -= 1
        self.match_winner = None
        self._lose_rack(shooter, "8 IN AN UNMARKED POCKET - LOSS CALLED",
                        rule="W-03")
        self.record.events[-1].actor = "player"
        return True

    def _rerack(self, why="RE-RACK"):
        self.group = [None, None]
        self.potted.clear()
        self.struck = False
        if why:
            self.note("rerack", why)

    def drop_without_ending(self, classes):
        """W-11's exception: the balls go down, the rack-ending call does not.

        The rule is exact - if dropping a wedged pair in would end the game,
        play resumes with them dropped but the rack-ending condition not
        triggered - so the classes are counted down and nothing is judged.
        """
        for cls in classes:
            self._rack_down(cls)
        self.note("pot", "WEDGED PAIR DROPPED - PLAY CONTINUES", rule="W-11",
                  actor="player")

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
        self.small_down = 0     # balls 1-8 off the table, credited or dead
        self.lowest = None      # the lowest number still on the table, if known
        self.n01_noted = False  # N-01 says once per rack that it cannot enforce

    def score(self, i):
        """Manual section 5: points past the race target are not marked."""
        target = self.race()
        return min(self.points[i], target[i]) if target else self.points[i]

    def race_progress(self, i):
        """Nine-ball races are in points, and can finish mid-rack."""
        return self.points[i]

    def group_name(self, i):
        """Nine-ball has no groups; the HUD reuses this slot for the race."""
        target = self.race()
        return f"TO {target[i]}" if target else "9-BALL"

    def snapshot(self):
        blob = super().snapshot()
        blob.update({"points": list(self.points), "dead": self.dead,
                     "rack_points": list(self.rack_points),
                     "balls_down": self.balls_down, "lowest": self.lowest,
                     "small_down": self.small_down,
                     "n01_noted": self.n01_noted})
        return blob

    def restore(self, blob):
        super().restore(blob)
        self.points = list(blob["points"])
        self.dead = blob["dead"]
        self.rack_points = list(blob["rack_points"])
        self.balls_down = blob["balls_down"]
        self.lowest = blob["lowest"]
        self.small_down = blob["small_down"]
        self.n01_noted = blob["n01_noted"]

    def _reset_rack(self):
        self.rack_points = [0, 0]
        self.dead = 0
        self.balls_down = 0
        self.small_down = 0
        self.lowest = None
        self.n01_noted = False

    def _check_invariant(self):
        """N-04's arithmetic, checked out loud rather than trusted."""
        total = self.rack_points[0] + self.rack_points[1] + self.dead
        if self.over and self.winner is not None and total != self.RACK_POINTS:
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

    def drop_without_ending(self, classes):
        """W-11's exception, in nine-ball: the balls score, the 9 cannot win.

        A wedged 9 dropped in by hand is spotted instead, which is the nearest
        nine-ball equivalent of "dropped but the rack-ending condition not
        triggered".
        """
        shooter = self._shot_owner.get(self.shot, self.turn)
        self._credit(shooter, [c for c in classes if not self._is_nine(c)],
                     fouled=False)
        if any(self._is_nine(c) for c in classes):
            self.note("note", "9 SPOTTED - WEDGED", rule="W-11", quiet=True)
        self.note("pot", "WEDGED PAIR DROPPED - PLAY CONTINUES", rule="W-11",
                  actor="player")

    def stalemate(self):
        """Manual 3.13, in nine-ball: the game ends but the points STAND.

        The opposite of 8-ball on every count: innings and defensive shots
        remain, and every ball still on the table is marked dead. The 9 is
        never dead, so a stalemated rack adds up to 8 rather than 10 and the
        ten-point invariant is only asserted on racks the 9 actually ended.
        """
        if self.over:
            return False
        left = 8 - self.small_down
        if left > 0:
            self.dead += left
            self.small_down = 8
        self.stalemated = True
        self.over, self.winner = True, None
        self.note("stalemate", f"STALEMATE - POINTS STAND - {left} DEAD",
                  rule="TM-13", actor="player")
        return True

    def push_out(self):
        """Manual 3.4 note: push-outs exist only in Masters, never levelled play.

        Allowed on the shot straight after the break; anything pocketed on it
        is spotted and the opponent chooses who shoots next, which is a
        conversation at the table rather than something to model.
        """
        if self.fmt != skill_level.MASTERS or self.over or self.shot > 1:
            self.note("note", "PUSH-OUT NOT ALLOWED HERE", rule="TM-4",
                      actor="player")
            return False
        self.note("note", f"PUSH-OUT - {self.name(self.turn)}", rule="TM-4",
                  actor="player")
        return True

    def _credit(self, shooter, classes, fouled):
        """N-04. Balls on a foul shot are dead; the 9 is never dead.

        Returns True when the 9 went down legally, which is the only way the
        rack is won.
        """
        nine_legal = False
        for cls in classes:
            if cls == CUE:
                continue
            if self._is_nine(cls):
                if fouled:
                    # N-03: spotted, not dead, not a win. The ball comes back.
                    self.note("note", "9 SPOTTED - POTTED ON A FOUL",
                              rule="N-03", quiet=True)
                    continue
                self.balls_down += 1
                self.points[shooter] += 2
                self.rack_points[shooter] += 2
                nine_legal = True
                continue
            if self.small_down >= 8:
                self.note("impossible", "IGNORED AN EXTRA BALL", rule="M-14",
                          confidence=0.0, quiet=True)
                continue
            self.balls_down += 1
            self.small_down += 1
            if fouled:
                self.dead += 1
                self.note("dead", "DEAD BALL - NOT CREDITED", rule="N-04",
                          quiet=True)
                continue
            self.points[shooter] += 1
            self.rack_points[shooter] += 1
        if nine_legal and self.small_down < 8:
            # The 9 went down early. Whatever is still on the table is recorded
            # as dead, credited to nobody - which is what makes the rack add up
            # to ten however it ended.
            left = 8 - self.small_down
            self.dead += left
            self.small_down = 8
            self.note("dead", f"{left} BALL{'S' if left > 1 else ''} LEFT ON "
                              f"THE TABLE - DEAD", rule="N-04", quiet=True)
        self._check_race()
        return nine_legal

    def shot_ended(self, f, frame):
        self.frame = frame
        if self.over:
            return
        shooter = self.turn
        scratch = CUE in f.pots or CUE in f.off_table
        objects = [c for c in f.pots if c != CUE]
        for cls in f.off_table:
            if cls != CUE:
                # Manual 3.8: off the table, spotted at once, never scored -
                # and that includes the 9.
                self.note("note", f"SPOT THE {'9' if self._is_nine(cls) else 'BALL'}"
                                  f" ON THE FOOT SPOT", rule="TM-8", quiet=True)

        if self.mode == PRACTICE:
            self.struck = True
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
            self._win_rack(shooter, "9-BALL")
            self._check_invariant()
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
        # Manual 3.3: the same illegal-break rule as 8-ball - re-rack, re-broken
        # by the same player, or by the other if the illegal break scratched.
        if (self.mode == LEAGUE and not objects and f.rail_balls is not None
                and f.rail_balls < BREAK_RAIL_BALLS):
            again = 1 - shooter if scratch else shooter
            self.breaker = self.turn = again
            self.inning = 1
            self._breaker_shot = False
            self.struck = False
            self.note("rerack", f"ILLEGAL BREAK - RE-RACK - "
                                f"{self.name(again)} BREAKS", rule="TM-3",
                      player=shooter)
            return
        if scratch:
            # N-02's exception: a 9 on the snap with a scratch is not a win, it
            # is N-03 - spotted - and the rest are dead. F-02's head-string
            # restriction is an 8-ball rule, so this ball in hand is anywhere.
            self._foul(shooter, "SCRATCH ON THE BREAK", rule="F-01")
            self._credit(shooter, objects, fouled=True)
            return
        nine_legal = self._credit(shooter, objects, fouled=False)
        if nine and nine_legal:
            self._win_rack(shooter, "9 ON THE SNAP", rule="N-02")
            self._check_invariant()
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
        elif not self.n01_noted:
            # Once per rack, not once per shot: the reason does not change
            # between shots, and a review on every stroke buries the real ones.
            self.n01_noted = True
            self.review("BALL NUMBERS NOT READABLE - N-01 NOT ENFORCED",
                        rule="F-09", confidence=f.contact_conf)

        if f.rail_contact is False and not f.pots and f.hit_object:
            self._foul(shooter, "NO RAIL AFTER CONTACT", rule="F-04")
            return True
        return False

    def summary(self):
        return {"discipline": NINE_BALL, "mode": self.mode,
                "players": self.players,
                "levels": [skill_level.describe(l) for l in self.levels]
                          if self.handicapped else ["Player", "Player"],
                "points": list(self.points),
                "rack_points": list(self.rack_points),
                "dead": self.dead,
                "invariant_ok": (self.rack_points[0] + self.rack_points[1]
                                 + self.dead == self.RACK_POINTS
                                 if self.over and self.winner is not None
                                 else True),
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
