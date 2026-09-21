"""Skill Levels: races, time-out allowances, and the words used for them.

H-04 is a naming rule with teeth, so it is worth stating plainly at the top of
the module that implements it. Three constraints, all locked:

  * The system is called SKILL LEVEL. A player is "a Skill Level 4".
  * The word HANDICAP never appears in player-facing copy. It appears in this
    docstring because this is not player-facing copy; it does not appear in any
    string this module can return.
  * The three letters of the league body whose rulebook this follows never
    appear in product, UI, or marketing. Not in a label, not in a tooltip, not
    in a log line a player could ever see.

`assert_clean()` at the bottom enforces the third one over every string this
module can produce, and main.py runs it at startup.

THE RANGES AND CHARTS come from the league's team manual, section 4 ("Games
Must Win" and "Points Required To Win"). By client decision they replace the
CSV's "1-7 everywhere": the charts are built on the real ranges and do not
survive being squeezed into another one.

    8-ball   Skill Level 2 to 7. Races in GAMES, from a two-way chart.
    9-ball   Skill Level 1 to 9. Races in POINTS, one target per player.
    Doubles  the two players' levels are ADDED and read off a separate chart.

A player with no established level starts at Skill Level 3 (manual, General
Rule 6). A level carried from the other discipline maps across unchanged,
except that 9-ball 1 becomes 8-ball 2 and 9-ball 8 or 9 becomes 8-ball 7.

H-03 is the other half: a Skill Level is applied ONLY inside a sanctioned league
match with defence marking enabled. Practice, casual play, and a league player
racking up against a stranger are all just players, and `applies()` is the
single place that decides.
"""

import json
import os

EIGHT_BALL, NINE_BALL = "8ball", "9ball"

#: Skill Levels per discipline, from the manual's charts.
RANGE = {EIGHT_BALL: (2, 7), NINE_BALL: (1, 9)}
MIN_LEVEL = min(lo for lo, _hi in RANGE.values())
MAX_LEVEL = max(hi for _lo, hi in RANGE.values())

#: A player with no established level yet. Races as a new player, which the
#: manual starts at 3; allowances treat them as the lower bracket (Rule 18).
UNRATED = None
NEW_PLAYER_LEVEL = 3

#: Levels at or above this are "senior" for the team cap (General Rule 26).
SENIOR_LEVEL = 6

#: The charts are the manual's own, so nothing here is provisional any more. A
#: race_charts.json dropped next to this file still overrides them, for a local
#: league whose bylaws differ.
PROVISIONAL = False

RACE_FILE = "race_charts.json"

#: 8-ball Games Must Win, singles: _EIGHT[mine][theirs] = games I need.
#: Transcribed from the chart row by row; the opponent's number is the same
#: table read the other way round, which the tests check.
_EIGHT = {
    2: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2},
    3: {2: 3, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2},
    4: {2: 4, 3: 3, 4: 3, 5: 3, 6: 3, 7: 2},
    5: {2: 5, 3: 4, 4: 4, 5: 4, 6: 4, 7: 3},
    6: {2: 6, 3: 5, 4: 5, 5: 5, 6: 5, 7: 4},
    7: {2: 7, 3: 6, 4: 5, 5: 5, 6: 5, 7: 5},
}

#: 8-ball Doubles Games Must Win, by COMBINED level. 6 stands for "6 or less".
#: Stored as (mine, theirs) because this chart is not perfectly symmetric.
_EIGHT_DOUBLES = {
    6:  {6: (2, 2), 7: (2, 3), 8: (2, 3), 9: (2, 4), 10: (2, 4), 11: (2, 5), 12: (2, 5)},
    7:  {6: (3, 2), 7: (3, 3), 8: (3, 3), 9: (3, 4), 10: (3, 4), 11: (2, 4), 12: (2, 5)},
    8:  {6: (3, 2), 7: (3, 3), 8: (3, 3), 9: (3, 4), 10: (3, 4), 11: (3, 5), 12: (3, 5)},
    9:  {6: (4, 2), 7: (4, 3), 8: (4, 3), 9: (4, 4), 10: (4, 4), 11: (3, 4), 12: (3, 5)},
    10: {6: (4, 2), 7: (4, 3), 8: (4, 3), 9: (4, 4), 10: (4, 4), 11: (3, 4), 12: (3, 5)},
    11: {6: (5, 2), 7: (4, 2), 8: (5, 3), 9: (4, 3), 10: (4, 3), 11: (4, 4), 12: (4, 5)},
    12: {6: (5, 2), 7: (5, 2), 8: (5, 3), 9: (5, 3), 10: (5, 3), 11: (5, 4), 12: (5, 5)},
}

#: 9-ball Points Required To Win, singles.
_NINE = {1: 14, 2: 19, 3: 25, 4: 31, 5: 38, 6: 46, 7: 55, 8: 65, 9: 75}

#: 9-ball Doubles Points Required To Win, by combined level. 4 = "4 or less".
_NINE_DOUBLES = {4: 19, 5: 22, 6: 25, 7: 28, 8: 31, 9: 35, 10: 38, 11: 42, 12: 46}

#: Match formats, which change time-outs and whether levels apply at all.
OPEN, LADIES, THREE_PERSON, DOUBLES, MASTERS, WORLD = (
    "open", "ladies", "3-person", "doubles", "masters", "world")
FORMATS = (OPEN, LADIES, THREE_PERSON, DOUBLES, MASTERS, WORLD)

#: Masters is non-handicapped: a race to 7 games, whoever is playing.
MASTERS_RACE = 7

#: Time-outs per game (General Rule 18): two for levels 1-3 and unrated, one
#: for 4 and up.
_TIMEOUTS_LOW = 2
_TIMEOUTS_HIGH = 1
_TIMEOUT_SPLIT = 4


def clamp(level, discipline=None):
    """A level inside the discipline's range, or None for unrated.

    With no discipline the widest range (1-9) is used, which is what a stored
    level is before anyone knows which game it will be played in.
    """
    if level is None:
        return UNRATED
    try:
        level = int(level)
    except (TypeError, ValueError):
        return UNRATED
    lo, hi = RANGE.get(discipline, (MIN_LEVEL, MAX_LEVEL))
    return max(lo, min(hi, level))


def racing_level(level, discipline):
    """The level a player actually races at: unrated races as a new player."""
    level = clamp(level, discipline)
    return clamp(NEW_PLAYER_LEVEL, discipline) if level is UNRATED else level


def carry_over(level, from_discipline, to_discipline):
    """A level established in one game, starting out in the other (Rule 6)."""
    if level is None or from_discipline == to_discipline:
        return clamp(level, to_discipline)
    level = int(level)
    if from_discipline == NINE_BALL and to_discipline == EIGHT_BALL:
        if level <= 1:
            return 2
        if level >= 8:
            return 7
    return clamp(level, to_discipline)


def describe(level, discipline=None):
    """How a level is written in front of players. H-04's actual output."""
    level = clamp(level, discipline)
    return "Unrated" if level is UNRATED else f"Skill Level {level}"


def short(level, discipline=None):
    """The compact HUD form, for a scoreboard strip that has no room."""
    level = clamp(level, discipline)
    return "SL -" if level is UNRATED else f"SL {level}"


def senior(level):
    """Does this level count toward the two-senior cap? (Rule 26)"""
    return level is not None and int(level) >= SENIOR_LEVEL


def timeouts(level, fmt=OPEN):
    """Time-outs allowed per game at this level, in this format (Rule 18).

    Doubles matches get one per game whatever the levels; Masters gets none;
    the World Championships give one per player per rack.
    """
    if fmt == MASTERS:
        return 0
    if fmt in (DOUBLES, WORLD):
        return 1
    level = clamp(level)
    if level is UNRATED or level < _TIMEOUT_SPLIT:
        return _TIMEOUTS_LOW
    return _TIMEOUTS_HIGH


class RaceChart:
    """Games (8-ball) or points (9-ball) each side needs, from both levels."""

    def __init__(self, eight=None, nine=None, provisional=PROVISIONAL):
        self.eight = eight or _EIGHT
        self.nine = nine or _NINE
        self.eight_doubles = _EIGHT_DOUBLES
        self.nine_doubles = _NINE_DOUBLES
        self.provisional = provisional

    @classmethod
    def load(cls, path=None):
        """The manual's charts, or a local override if one is present.

        JSON shape, both keys optional:
            {"eight_ball": {"4": {"5": 3, ...}, ...},
             "nine_ball":  {"4": 31, ...}}
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
        return cls(eight, nine, provisional=False)

    def eight_ball(self, mine, theirs):
        """Racks I need to win an 8-ball match against them."""
        a = racing_level(mine, EIGHT_BALL)
        b = racing_level(theirs, EIGHT_BALL)
        return self.eight.get(a, {}).get(b, 2)

    def nine_ball(self, mine, _theirs=None):
        """Points I need to win a 9-ball match. Nine-ball races are one-sided."""
        return self.nine.get(racing_level(mine, NINE_BALL), 14)

    def doubles(self, discipline, mine, theirs):
        """(my target, their target) from COMBINED levels (pairs of levels)."""
        a = sum(racing_level(l, discipline) for l in mine)
        b = sum(racing_level(l, discipline) for l in theirs)
        if discipline == NINE_BALL:
            key = lambda n: max(4, min(12, n))
            return self.nine_doubles[key(a)], self.nine_doubles[key(b)]
        key = lambda n: max(6, min(12, n))
        return self.eight_doubles[key(a)][key(b)]

    def race(self, discipline, mine, theirs, fmt=OPEN):
        """(my target, their target) for whichever game and format is played."""
        if fmt == MASTERS:
            return MASTERS_RACE, MASTERS_RACE
        if discipline == NINE_BALL:
            return self.nine_ball(mine), self.nine_ball(theirs)
        return self.eight_ball(mine, theirs), self.eight_ball(theirs, mine)


def applies(mode, defence_marking, fmt=OPEN):
    """Is a stored Skill Level allowed to affect anything right now? (H-03)

    Both conditions, every time: league mode, and defence marking on. A league
    match without defence marking has an inning count and a safety count that
    nobody vouched for, and a race built on those is worse than no race at all.
    Masters is never levelled - it is a straight race to 7.
    """
    return mode == "league" and bool(defence_marking) and fmt != MASTERS


#: Substrings that must never reach a player.
_FORBIDDEN = ("apa", "handicap", "handicapped", "handicapping", "equalizer")


def clean(text):
    """Does this string keep H-04's promises? Case-insensitive, substring."""
    low = str(text).lower()
    return not any(bad in low for bad in _FORBIDDEN)


def offenders(strings):
    return [s for s in strings if not clean(s)]


def assert_clean():
    """Every string this module can produce, checked against H-04."""
    produced = [describe(None), short(None)]
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        produced += [describe(level), short(level)]
    bad = offenders(produced)
    if bad:
        raise AssertionError(f"H-04 violated by: {bad}")
    return True
