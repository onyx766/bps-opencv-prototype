"""BPS_Scoring_TestSuite_v1.3 - the client's locked spec, as code.

Every row of the client CSV is a decision that has already been made and signed
off. Most of them are decisions about what the system must NOT do, which is the
part that gets lost first: a spec that says "not auto-called" is invisible in a
codebase unless something writes it down, and the next person to touch the file
re-implements the thing the client spent a meeting deciding to leave out.

So the spec lives here as a table, and every rule carries the HANDLING the
client chose for it. That gives three things for the price of one:

  * the code can be read against the spec - each rule names where it is done;
  * the HUD can show coverage honestly, including the rules it declines to
    judge, because M-14 is about not faking confidence we do not have;
  * `check()` reads the CSV back and reports any rule that has appeared,
    changed status, or gone missing since v1.3, so a new CSV from the client
    fails loudly rather than being silently half-implemented.

HANDLING is the whole point of the table. Five values:

  AUTO      The system calls it, alone, and writes it to the record.
  PROMPT    The system spots the situation and ASKS; a player taps to resolve.
  MANUAL    Only a player tap can create it. Never inferred, ever - these are
            the rules where a camera guessing would actively make play worse.
  LOGGED    Recorded, flagged for replay, and explicitly NOT ruled on.
  DEFERRED  Out of scope for v1 by client decision. Not attempted.

A rule's handling is not an implementation detail. F-06 is MANUAL because a
camera cannot tell a hand from a legal bridge, and auto-calling it "will
infuriate users" (client's words). Promoting it to AUTO later is a product
decision, not a refactor.
"""

import csv
import os

#: Where the client's CSV sits, relative to this file.
SPEC_FILE = "BPS_Scoring_TestSuite_v1.3.csv"
SPEC_VERSION = "1.3"

AUTO = "auto"
PROMPT = "prompt"
MANUAL = "manual"
LOGGED = "logged"
DEFERRED = "deferred"

#: Confidence a first-contact call needs before it may be ruled a foul (F-03).
#: Below it, F-09 applies: no foul, log a review. The number is the client's.
CONTACT_CONFIDENCE = 0.85

#: Seconds a ball must have been stationary before a fall stops counting as
#: part of the shot (M-02). From CSI/BCA's hanging-ball rule, which is the only
#: rulebook that gives a countable number. Counted from the moment ALL balls
#: stop, not from the moment the shooter walks away.
HANGING_SECONDS = 5.0

#: Seconds a vanished ball has to reappear in before its pot is silently taken
#: back (M-04). Short, because this one runs on every shot and a visible
#: correction every time a ball rattles out would be worse than the error.
PHANTOM_SECONDS = 2.0

#: Seconds of undo offered on an auto-called loss of rack (W-02). The call is
#: high-confidence and camera-friendly, but it ends a game, so it gets a window.
EIGHT_EARLY_UNDO_SECONDS = 10.0

#: Seconds of footage the review buffer holds (M-13).
REVIEW_SECONDS = 10.0

#: Seconds a time-out runs for before the timer reads over (H-05). The rulebook
#: says roughly a minute; the timer is visible and advisory, not enforced - it
#: is the shooting team's clock, not a shot clock.
TIMEOUT_SECONDS = 60.0


class Rule:
    """One row of the client's suite, plus where it is implemented."""

    __slots__ = ("id", "handling", "title", "where", "note")

    def __init__(self, rid, handling, title, where, note=""):
        self.id = rid
        self.handling = handling
        self.title = title
        self.where = where
        self.note = note

    @property
    def automated(self):
        return self.handling in (AUTO, PROMPT)

    def __repr__(self):
        return f"<{self.id} {self.handling} {self.title}>"


def _r(*args, **kw):
    return Rule(*args, **kw)


#: The suite. `where` names the module and function that carries the rule, so a
#: reader can go from the client's test id to the code in one hop.
RULES = [
    # ---- fouls ------------------------------------------------------------
    _r("F-01", AUTO, "Scratch", "games.EightBallGame._judge_fouls",
       "Foul counter, never a point deduction."),
    _r("F-02", AUTO, "Scratch on the break", "games.EightBallGame._judge_break",
       "Ball in hand behind the head string; main.draw_head_string shows it."),
    _r("F-03", AUTO, "Wrong group hit first", "games.EightBallGame._judge_fouls",
       f"Only at >= {CONTACT_CONFIDENCE:.0%} confidence; below that, F-09."),
    _r("F-04", AUTO, "No rail after contact", "games.EightBallGame._judge_fouls",
       "logic.RailWatch sees a ball enter the cushion band."),
    _r("F-05", AUTO, "No object ball hit at all", "games.EightBallGame._judge_fouls",
       "Nothing but the cue ball moved."),
    _r("F-06", MANUAL, "Ball touched by hand or cue", "games.BaseGame.mark_foul",
       "HUD tap only. A camera cannot separate a hand from a legal bridge."),
    _r("F-07", LOGGED, "Double hit / push / scoop", "logic.GameSession._log_contact",
       "Flagged for replay above a contact-duration threshold. Never ruled on."),
    _r("F-08", DEFERRED, "Frozen-ball rail requirement", "logic.GameSession._frozen",
       "Marked frozen on the HUD and handed to the players."),
    _r("F-09", AUTO, "Contact ambiguous", "games.EightBallGame._judge_fouls",
       "No foul. Logged 'review' with a clip pointer."),
    _r("F-10", AUTO, "Late foul call", "match_record.MatchRecord.foul_window_open",
       "The window closes when the next stroke begins."),

    # ---- HUD and handicapping --------------------------------------------
    _r("H-01", AUTO, "Inning tracking", "games.BaseGame._set_turn",
       "Breaker-anchored: increments when the breaker comes back to the table."),
    _r("H-02", MANUAL, "Defensive shot", "games.BaseGame.mark_defence",
       "Transient, shot-bound button. Never persistently on screen."),
    _r("H-03", AUTO, "Skill Level handicapping", "skill_level.RaceChart",
       "League mode only, and only with H-02 marking enabled."),
    _r("H-04", AUTO, "Naming", "skill_level.describe",
       "'Skill Level 1-7'. The letters A-P-A never appear in product copy."),
    _r("H-05", MANUAL, "Time-outs and coaching", "games.TimeOuts",
       "Charged on END TIME OUT, so a mis-tap costs nothing."),

    # ---- match mechanics --------------------------------------------------
    _r("M-01", AUTO, "Turn advance", "games.BaseGame.shot_ended",
       "No confirmation prompt, ever."),
    _r("M-02", AUTO, "Score commit timing", "logic.GameSession._hanging",
       f"Commits at rest; a ball falling after {HANGING_SECONDS:.0f}s is replaced."),
    _r("M-03", AUTO, "Undo", "match_record.MatchRecord.undo",
       "Editable until the next break. No PIN, no timers."),
    _r("M-04", AUTO, "Phantom pocket", "logic.GameSession._reconcile_phantoms",
       f"Silent inside {PHANTOM_SECONDS:.0f}s; failsafe until the next shot."),
    _r("M-05", AUTO, "Occlusion", "logic.GameSession.busy",
       "Quiet indicator. Never an alarming banner - occlusion is normal play."),
    _r("M-06", PROMPT, "Colour below threshold", "logic.GameSession._confirm_groups",
       "Asks once, with two thumbnails. Never silently guesses."),
    _r("M-07", AUTO, "Ball moved by hand", "logic.GameSession._replaced",
       "Never a foul. The HUD shows where to put it back."),
    _r("M-08", AUTO, "Cue ball removed", "logic.GameSession._cue_ball_missing",
       "Wait state until it returns, including a substituted ball."),
    _r("M-09", PROMPT, "Rack detection", "logic.GameSession._racked",
       "Auto-detected; confirmed by tap on the first ever rack per table."),
    _r("M-10", AUTO, "Break legality", "games.EightBallGame._judge_break",
       "Enforced in league mode, advisory in casual."),
    _r("M-11", AUTO, "Practice mode", "games.Analytics",
       "No fouls, no turns, no win conditions. TSR / CRR / SDR only."),
    _r("M-12", AUTO, "Match record", "match_record.MatchRecord",
       "The log is the source of truth; the scoreboard is a view of it."),
    _r("M-13", MANUAL, "Disputed call", "main.draw_review",
       "Pulls the last 10 seconds. BPS does not overrule."),
    _r("M-14", AUTO, "Confidence display", "main.scoreboard",
       "Sub-threshold calls carry a visible dot."),
    _r("M-15", AUTO, "Player identity", "games.BaseGame",
       "Player 1 / Player 2 by table side. No face recognition, ever."),

    # ---- nine-ball --------------------------------------------------------
    _r("N-01", AUTO, "Lowest ball first", "games.NineBallGame._judge_fouls"),
    _r("N-02", AUTO, "9 on the snap", "games.NineBallGame._judge_break"),
    _r("N-03", AUTO, "9 on a foul is spotted", "games.NineBallGame._judge_nine"),
    _r("N-04", AUTO, "Points and dead balls", "games.NineBallGame._credit",
       "Invariant: p1 + p2 + dead = 10 per rack."),

    # ---- winning and losing the rack --------------------------------------
    _r("W-01", AUTO, "Legal 8 in the marked pocket", "games.EightBallGame._judge_eight"),
    _r("W-02", AUTO, "8 potted early", "games.EightBallGame._judge_eight",
       f"{EIGHT_EARLY_UNDO_SECONDS:.0f}-second undo window."),
    _r("W-03", AUTO, "8 in the wrong pocket", "games.EightBallGame._judge_eight",
       "Only when pocket marking is enabled."),
    _r("W-04", AUTO, "8 with the last group ball", "games.EightBallGame._judge_eight"),
    _r("W-05", AUTO, "Scratch shooting at the 8", "games.EightBallGame._judge_eight"),
    _r("W-06", AUTO, "Missing the 8 entirely", "games.EightBallGame._judge_eight",
       "Foul and ball in hand. NOT a loss - the commonest scoring-app bug."),
    _r("W-07", AUTO, "8 knocked off the table", "games.EightBallGame._judge_eight"),
    _r("W-08", AUTO, "Open table", "games.EightBallGame._assign",
       "Open until a legal pot on a NON-BREAK shot."),
    _r("W-09", MANUAL, "Stalemate", "games.BaseGame.stalemate",
       "Player-initiated only. Never auto-declared."),
    _r("W-10", AUTO, "Ball bounces back out", "logic.GameSession._reconcile_phantoms",
       "Same detector and threshold as M-04."),
    _r("W-11", PROMPT, "Two balls wedged in the jaw", "logic.GameSession._wedged",
       "Deemed pocketed, but the HUD prompts rather than auto-calling."),
]

BY_ID = {rule.id: rule for rule in RULES}


def rule(rid):
    """The spec row behind a call, so log entries can cite it."""
    return BY_ID.get(rid)


def cite(rid):
    """'W-06 Missing the 8 entirely' - what a log entry and the HUD show."""
    r = BY_ID.get(rid)
    return f"{rid} {r.title}" if r else rid


def by_handling(handling):
    return [r for r in RULES if r.handling == handling]


def coverage():
    """Counts per handling, for the one-line startup banner."""
    out = {}
    for r in RULES:
        out[r.handling] = out.get(r.handling, 0) + 1
    return out


def read_spec(path=None):
    """The client CSV as {test_id: row}. Returns {} when the file is absent.

    The CSV is the client's artefact and is never written to - it is read back
    only to check this table against it.
    """
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                SPEC_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {row["test_id"]: row for row in csv.DictReader(fh)
                if row.get("test_id")}


def check(path=None):
    """Compare the CSV against this table.

    Returns (missing, extra, unlocked):
      missing   in the CSV, not implemented here - a new rule arrived
      extra     implemented here, not in the CSV - a rule was withdrawn
      unlocked  rows whose status is no longer 'locked' - still being argued

    Called at startup so a v1.4 CSV dropped into the folder says so, instead of
    being three rules ahead of the code in silence.
    """
    spec = read_spec(path)
    if not spec:
        return [], [], []
    missing = sorted(set(spec) - set(BY_ID))
    extra = sorted(set(BY_ID) - set(spec))
    unlocked = sorted(rid for rid, row in spec.items()
                      if (row.get("status") or "").strip().lower() != "locked")
    return missing, extra, unlocked


def banner(path=None):
    """One line for the startup log: coverage, and any drift from the CSV."""
    cov = coverage()
    line = (f"Spec:   v{SPEC_VERSION}  {len(RULES)} rules  "
            f"({cov.get(AUTO, 0)} auto, {cov.get(PROMPT, 0)} prompt, "
            f"{cov.get(MANUAL, 0)} player-called, {cov.get(LOGGED, 0)} logged, "
            f"{cov.get(DEFERRED, 0)} deferred)")
    missing, extra, unlocked = check(path)
    if missing:
        line += f"\n        NEW IN THE CSV, NOT IMPLEMENTED: {', '.join(missing)}"
    if extra:
        line += f"\n        IMPLEMENTED BUT GONE FROM THE CSV: {', '.join(extra)}"
    if unlocked:
        line += f"\n        NOT LOCKED, MAY STILL CHANGE: {', '.join(unlocked)}"
    return line


if __name__ == "__main__":
    print(banner())
    for r in RULES:
        print(f"  {r.id}  {r.handling:<8}  {r.title:<34}  {r.where}")
