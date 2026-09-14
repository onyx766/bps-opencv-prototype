"""Skill Level 1-7: races, time-out allowance, and the words used for them.

H-04 is a naming rule with teeth, so it is worth stating plainly at the top of
the module that implements it. Three constraints, all locked:

  * The system is called SKILL LEVEL. A player is "a Skill Level 4".
  * The word HANDICAP never appears in player-facing copy. It appears in this
    docstring because this is not player-facing copy; it does not appear in any
    string this module can return.
  * The three letters of the league body whose chart this resembles never
    appear in product, UI, or marketing. Not in a label, not in a tooltip, not
    in a log line a player could ever see. The client is having counsel confirm
    the position and the cheap way to stay out of that conversation is to never
    have written the name down.

`assert_clean()` at the bottom enforces the third one over every string this
module can produce, and main.py runs it at startup. It is a cheap test and it
protects a legal position, which is a good trade.

H-03 is the other half: a Skill Level is applied ONLY inside a sanctioned
league match with defence marking enabled. Practice, casual play, and a league
player racking up against a stranger are all just players, and `applies()` is
the single place that decides. Outside that, nothing here is consulted and no
stored level is ever read - which is also the privacy posture M-15 wants.

THE RACE CHART IS PROVISIONAL. The client has the authoritative chart already
coded on their side (their note names the file). The table below is a
placeholder with the right SHAPE so the rest of the system can be built and
demonstrated against it, and `load()` will take a JSON override without a code
change. Every number here should be replaced from the client's chart before
anything is played for a result that matters, and `PROVISIONAL` stays True
until it is, so the HUD can say so rather than quietly implying authority.
"""

import json
import os

#: Skill Levels the product recognises. H-04 fixes this at 1-7.
MIN_LEVEL, MAX_LEVEL = 1, 7

#: An unrated player - someone with no established level yet. Treated as the
#: lowest bracket for allowances, and given the shortest race.
UNRATED = None

#: Set False once the client's own chart has been dropped in. While True the
#: HUD marks races as provisional, because a race is the one number in a league
#: match that decides who won.
PROVISIONAL = True

RACE_FILE = "race_charts.json"

#: Games needed to win, keyed by (my level, their level). Provisional - see the
#: module docstring. The shape is what matters: a lower Skill Level always needs
#: fewer games than a higher one, equal levels are an even race, and the gap
#: widens with the difference rather than exploding.
_EIGHT_BALL = {
    1: {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2},
    2: {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2},
    3: {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2},
    4: {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3},
    5: {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4},
    6: {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5},
    7: {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5},
}

#: Nine-ball races in POINTS, not games - a nine-ball rack is worth 10 points
#: (N-04) and a race is a points target per player. Provisional, as above.
_NINE_BALL = {1: 14, 2: 19, 3: 25, 4: 31, 5: 38, 6: 46, 7: 55}

#: Time-outs per game (H-05). The rulebook allowance, and the one number in
#: this file that is NOT provisional: it is stated outright in the client's
#: H-05 row - two at the bottom of the scale, one above it.
_TIMEOUTS_LOW = 2        # Skill Level 1-3, and unrated
_TIMEOUTS_HIGH = 1       # Skill Level 4 and up
_TIMEOUT_SPLIT = 4


def clamp(level):
    """A level inside 1-7, or None for unrated. Nothing else gets through."""
    if level is None:
        return UNRATED
    try:
        level = int(level)
    except (TypeError, ValueError):
        return UNRATED
    return max(MIN_LEVEL, min(MAX_LEVEL, level))


def describe(level):
    """How a level is written in front of players. H-04's actual output."""
    level = clamp(level)
    return "Unrated" if level is UNRATED else f"Skill Level {level}"


def short(level):
    """The compact HUD form, for a scoreboard strip that has no room."""
    level = clamp(level)
    return "SL -" if level is UNRATED else f"SL {level}"


def timeouts(level):
    """Time-outs allowed per game at this level (H-05)."""
    level = clamp(level)
    if level is UNRATED or level < _TIMEOUT_SPLIT:
        return _TIMEOUTS_LOW
    return _TIMEOUTS_HIGH


class RaceChart:
    """Games (or points) each player needs, given both Skill Levels.

    Holds both disciplines because a match is one or the other and the caller
    should not have to know which table to reach into.
    """

    def __init__(self, eight=None, nine=None, provisional=PROVISIONAL):
        self.eight = eight or _EIGHT_BALL
        self.nine = nine or _NINE_BALL
        self.provisional = provisional

    @classmethod
    def load(cls, path=None):
        """Take the client's chart if it is sitting next to us, else the stub.

        JSON shape, both keys optional:
            {"eight_ball": {"4": {"5": 3, ...}, ...},
             "nine_ball":  {"4": 31, ...}}

        A file that loads is treated as authoritative, which is what clears the
        provisional marker off the HUD.
        """
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    RACE_FILE)
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return cls()
        eight = {int(a): {int(b): int(n) for b, n in row.items()}
                 for a, row in (data.get("eight_ball") or {}).items()} or None
        nine = {int(a): int(n)
                for a, n in (data.get("nine_ball") or {}).items()} or None
        return cls(eight, nine, provisional=not (eight or nine))

    def _level(self, level):
        """Unrated races as the lowest bracket - the shortest race on the chart."""
        level = clamp(level)
        return MIN_LEVEL if level is UNRATED else level

    def eight_ball(self, mine, theirs):
        """Racks I need to win an 8-ball match against them."""
        row = self.eight.get(self._level(mine), {})
        return row.get(self._level(theirs), 2)

    def nine_ball(self, mine, _theirs=None):
        """Points I need to win a 9-ball match. Nine-ball races are one-sided."""
        return self.nine.get(self._level(mine), 14)

    def race(self, discipline, mine, theirs):
        """(my target, their target) for whichever game is being played."""
        if discipline == "9ball":
            return self.nine_ball(mine), self.nine_ball(theirs)
        return self.eight_ball(mine, theirs), self.eight_ball(theirs, mine)


def applies(mode, defence_marking):
    """Is a stored Skill Level allowed to affect anything right now? (H-03)

    Both conditions, every time. A league match without defence marking has an
    inning count and a safety count that nobody vouched for, and a race built on
    those is worse than no race at all - so the level is simply not applied and
    everyone is just a player. Practice and casual never reach the second test.
    """
    return mode == "league" and bool(defence_marking)


#: Substrings that must never reach a player. Checked over everything this
#: module can return, and over the HUD's own strings from main.py.
_FORBIDDEN = ("apa", "handicap", "handicapped", "handicapping")


def clean(text):
    """Does this string keep H-04's promises?

    Word-boundary-free deliberately: 'APA' inside a longer token is still the
    letters on screen, and no legitimate product string in this system contains
    them. The check is case-insensitive because a lowercase slip is the likely
    one.
    """
    low = str(text).lower()
    return not any(bad in low for bad in _FORBIDDEN)


def offenders(strings):
    return [s for s in strings if not clean(s)]


def assert_clean():
    """Every string this module can produce, checked against H-04.

    Run at startup. If this ever fails, something player-facing has picked up a
    word the client decided would not appear in the product, and it should fail
    at the console rather than on a screen in a bar.
    """
    produced = [describe(None), short(None)]
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        produced += [describe(level), short(level)]
    bad = offenders(produced)
    if bad:
        raise AssertionError(f"H-04 violated by: {bad}")
    return True
