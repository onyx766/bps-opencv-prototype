"""Team-match arithmetic from the league manual's General Rules.

A table camera scores one game at a time. A league night is five of them (or
three), played by a line-up each captain declares match by match, under limits
on how strong that line-up may be and with fixed points for anything that does
not get played. None of that is visible to a camera, and all of it is pure
arithmetic - so it lives here, with no frames and no rules engine, and the HUD
or a scoresheet export can ask it questions.

WHAT IS HERE (manual General Rules, by number)
    11  matches per team match, per format
    12  who declares first, alternating
    14  bye points
    15  forfeit points, forfeits must be the last matches
    18  time-outs per format (delegated to skill_level.timeouts)
    25  Team Skill Level Limit: the 23-, 13-, 14- and 10-Rules, the reduced
        line-ups a team that cannot comply must play, and the penalty
    26  at most two senior (Skill Level 6+) players in a match
    27  playoffs: a match ends when the lead cannot be overcome; ties in
        standings and in playoff matches; playoff eligibility
    23  local lowest attainable level

WHAT IS NOT, because it is league administration rather than a score:
membership and fees, age and identity checks, roster add/drop windows, gambling,
equipment, amateur status, conduct penalties, protests, appeals and tournament
qualification. Those are decisions people make about people, and a scoring
system that pretended to enforce them would only be in the way.

MATCH POINTS PER INDIVIDUAL MATCH are NOT in the manual. It says 8-ball matches
are "worth up to three points" and 9-ball "up to 20", and that the split is read
off a chart printed on the scoresheet - which is not reproduced in the manual.
`MatchPoints` therefore takes that chart as data and refuses to guess without
it. Supply `match_points.json` from a real scoresheet.
"""

import json
import os

import skill_level
from skill_level import (DOUBLES, EIGHT_BALL, LADIES, MASTERS, NINE_BALL, OPEN,
                         THREE_PERSON)

#: Individual matches in one team match (Rule 11 and League Organization).
MATCHES = {OPEN: 5, LADIES: 3, THREE_PERSON: 3, DOUBLES: 3, MASTERS: 3}

#: Team Skill Level Limit (Rule 25). Doubles is "check your Local Bylaws"; 10 is
#: the figure the rule is named after. Masters has no limit.
LIMIT = {OPEN: 23, LADIES: 13, THREE_PERSON: 14, DOUBLES: 10, MASTERS: None}

#: Reduced line-ups for a team whose roster cannot comply (Rule 25): play this
#: many players within this total and forfeit the rest.
REDUCED = {OPEN: [(4, 19), (3, 15)], LADIES: [(2, 10)], THREE_PERSON: [(2, 11)]}

#: Senior players allowed in one team match (Rule 26).
MAX_SENIORS = 2

#: Bye points (Rule 14): (format, discipline) -> points.
BYE = {(OPEN, EIGHT_BALL): 8, (DOUBLES, EIGHT_BALL): 8,
       (OPEN, NINE_BALL): 60, (DOUBLES, NINE_BALL): 60,
       (LADIES, EIGHT_BALL): 5, (THREE_PERSON, EIGHT_BALL): 5,
       (THREE_PERSON, NINE_BALL): 40, (MASTERS, EIGHT_BALL): 15,
       (MASTERS, NINE_BALL): 15}

#: Forfeit points per individual match (Rule 15): (regular, playoffs).
FORFEIT_MATCH = {(OPEN, EIGHT_BALL): (2, 3), (OPEN, NINE_BALL): (15, 20),
                 (LADIES, EIGHT_BALL): (2, 3), (THREE_PERSON, EIGHT_BALL): (2, 3),
                 (THREE_PERSON, NINE_BALL): (15, 20),
                 (MASTERS, EIGHT_BALL): (5, 7), (MASTERS, NINE_BALL): (5, 7)}

#: Doubles format forfeits: singles match, then the doubles match (double).
FORFEIT_DOUBLES = {EIGHT_BALL: (2, 4), NINE_BALL: (15, 30)}

#: Full team forfeit (Rule 15).
FORFEIT_TEAM = {(OPEN, EIGHT_BALL): 8, (OPEN, NINE_BALL): 60,
                (LADIES, EIGHT_BALL): 5, (THREE_PERSON, EIGHT_BALL): 5,
                (THREE_PERSON, NINE_BALL): 40, (DOUBLES, EIGHT_BALL): 8,
                (DOUBLES, NINE_BALL): 60, (MASTERS, EIGHT_BALL): 15,
                (MASTERS, NINE_BALL): 15}

#: Most points one individual match can earn (League Structure: Scoring).
MATCH_MAX = {(OPEN, EIGHT_BALL): 3, (LADIES, EIGHT_BALL): 3,
             (THREE_PERSON, EIGHT_BALL): 3, (DOUBLES, EIGHT_BALL): 3,
             (OPEN, NINE_BALL): 20, (THREE_PERSON, NINE_BALL): 20,
             (DOUBLES, NINE_BALL): 20, (MASTERS, EIGHT_BALL): 7,
             (MASTERS, NINE_BALL): 7}

#: ... and the doubles match inside a Doubles team match.
DOUBLES_MATCH_MAX = {EIGHT_BALL: 6, NINE_BALL: 40}

#: Time rules (Time Guidelines, Rule 15).
NEXT_PLAYER_SECONDS = 60
FORFEIT_LATE_MINUTES = 15

#: Matches a player must have played with the team to play in playoffs (27).
PLAYOFF_MIN_MATCHES = 4

#: Individual forfeits above which a team cannot be the wild card (27).
WILD_CARD_MAX_FORFEITS = 5


def bye_points(fmt, discipline):
    return BYE.get((fmt, discipline), 0)


def forfeit_points(fmt, discipline, playoffs=False, doubles_match=False):
    """Points for one forfeited individual match (Rule 15)."""
    if fmt == DOUBLES:
        singles, doubles = FORFEIT_DOUBLES[discipline]
        points = doubles if doubles_match else singles
        if discipline == EIGHT_BALL and playoffs:
            points += 2 if doubles_match else 1
        return points
    regular, playoff = FORFEIT_MATCH.get((fmt, discipline), (0, 0))
    return playoff if playoffs else regular


def team_forfeit_points(fmt, discipline):
    return FORFEIT_TEAM.get((fmt, discipline), 0)


def declares_first(coin_winner_declares_first, match_index):
    """Which team declares first in match `match_index` (0-based, Rule 12).

    Returns 0 for the coin-toss winner, 1 for the other team. Whoever declares
    first in match one declares second in match two, and so on.
    """
    first = 0 if coin_winner_declares_first else 1
    return first if match_index % 2 == 0 else 1 - first


def lowest_attainable(established_level):
    """Rule 23: an established player cannot drop more than one level."""
    if established_level is None:
        return None
    return max(skill_level.MIN_LEVEL, int(established_level) - 1)


class LineupCheck:
    """The verdict on a team's line-up so far."""

    __slots__ = ("ok", "total", "limit", "seniors", "problems")

    def __init__(self, ok, total, limit, seniors, problems):
        self.ok = ok
        self.total = total
        self.limit = limit
        self.seniors = seniors
        self.problems = problems

    def __repr__(self):
        return (f"<LineupCheck ok={self.ok} total={self.total}/{self.limit} "
                f"seniors={self.seniors} {self.problems}>")


def check_lineup(declared, roster=(), fmt=OPEN, eligible=None):
    """Would this line-up break Rule 25 or Rule 26?

    `declared` is the Skill Levels already put up, in order. `roster` is every
    level on the team; the undeclared remainder is used to PROJECT the rest of
    the match, because Rule 25 says a team playing less than a full match must
    show it would not have exceeded the limit had every match been played - so
    the cheapest possible finish is added to what has been declared.

    `eligible` (a parallel list of booleans over `roster`) drops ineligible
    players from the projection, as Rule 25 Note 3 requires in playoffs.
    """
    limit = LIMIT.get(fmt)
    declared = [int(l) for l in declared]
    problems = []
    seniors = sum(1 for l in declared if skill_level.senior(l))
    if fmt != MASTERS and seniors > MAX_SENIORS:
        problems.append(f"{seniors} senior players - at most {MAX_SENIORS}")

    total = sum(declared)
    if limit is not None:
        pool = [int(l) for i, l in enumerate(roster)
                if eligible is None or eligible[i]]
        remaining = list(pool)
        for l in declared:
            if l in remaining:
                remaining.remove(l)
        still_to_play = max(0, MATCHES.get(fmt, len(declared)) - len(declared))
        projected = total + sum(sorted(remaining)[:still_to_play])
        if total > limit:
            problems.append(f"line-up totals {total} - limit is {limit}")
        elif projected > limit and len(remaining) >= still_to_play:
            problems.append(f"cheapest full line-up totals {projected} - "
                            f"limit is {limit}")
    return LineupCheck(not problems, total, limit, seniors, problems)


def reduced_lineup(roster, fmt=OPEN):
    """For a team that cannot comply at all: how many to play, and within what.

    Returns (players, total_limit, forfeits) or None when a full line-up fits.
    """
    limit = LIMIT.get(fmt)
    need = MATCHES.get(fmt)
    if limit is None or need is None:
        return None
    cheapest = sorted(int(l) for l in roster)
    if len(cheapest) >= need and sum(cheapest[:need]) <= limit:
        return None
    for players, cap in REDUCED.get(fmt, []):
        if sum(cheapest[:players]) <= cap:
            return players, cap, need - players
    return 0, 0, need


def limit_penalty(points_before, fmt, discipline, broken_at, matches=None):
    """Rule 25's penalty: (offending team's points, non-offending team's points).

    The offender gets nothing for the whole team match. The other team keeps
    what it won before the violation, plus forfeit points for the match in
    which the limit was broken and every match after it.
    """
    matches = matches or MATCHES.get(fmt, 5)
    per = forfeit_points(fmt, discipline)
    return 0, points_before + per * (matches - broken_at)


def clinched(points, remaining_max):
    """Rule 27: has a playoff match been decided before it finished?

    `points` is (team A, team B). `remaining_max` is the most points still
    available across every unplayed individual match. Returns the index of the
    team that cannot be caught, or None.
    """
    a, b = points
    if a > b + remaining_max:
        return 0
    if b > a + remaining_max:
        return 1
    return None


def playoff_tie_winner(fmt, individual_wins):
    """Rule 27, ties in a playoff match: who wins on individual matches.

    `individual_wins` is the sequence of individual-match winners in playing
    order (0 or 1). For Doubles, the doubles match is the LAST entry.
    """
    if fmt == DOUBLES:
        return individual_wins[-1] if individual_wins else None
    if fmt in (MASTERS, LADIES, THREE_PERSON):
        return individual_wins[0] if individual_wins else None
    count = [0, 0]
    for winner in individual_wins:
        count[winner] += 1
        if count[winner] == 2:
            return winner
    return None


def standings_tie(head_to_head, last_meeting_points=None,
                  last_meeting_matches=None, weekly_points=None):
    """Rule 27, two teams tied in the standings.

    head_to_head          (A, B) points across every meeting this session
    last_meeting_points   (A, B) points the last time they played
    last_meeting_matches  (A, B) individual matches won that night
    weekly_points         [(A, B), ...] newest week first, for teams that never
                          met - the tie goes to whoever scored more in the
                          last week, then the week before, and so on.
    Returns 0, 1, or None if every step is still level.
    """
    steps = []
    if head_to_head is not None:
        steps += [head_to_head, last_meeting_points, last_meeting_matches]
    else:
        steps += list(weekly_points or [])
    for pair in steps:
        if not pair:
            continue
        if pair[0] != pair[1]:
            return 0 if pair[0] > pair[1] else 1
    return None


def playoff_eligible(matches_played_with_team):
    return matches_played_with_team >= PLAYOFF_MIN_MATCHES


def wild_card_eligible(forfeited_matches, fees_current=True, ruled_ineligible=False):
    return (forfeited_matches <= WILD_CARD_MAX_FORFEITS and fees_current
            and not ruled_ineligible)


class MatchPoints:
    """Points one individual match earns, from the scoresheet's own chart.

    Deliberately empty without data: the manual names the chart but does not
    print it, and a guessed split would silently move league standings.
    JSON shape:
        {"8ball": {"<games the loser won>": [winner, loser], ...},
         "9ball": {"<points the loser scored>": [winner, loser], ...}}
    where 9-ball keys are the lower bound of each band.
    """

    FILE = "match_points.json"

    def __init__(self, table=None):
        self.table = table or {}

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    cls.FILE)
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    @property
    def available(self):
        return bool(self.table)

    def award(self, discipline, loser_result):
        """(winner points, loser points), or None when no chart was supplied."""
        chart = self.table.get(discipline)
        if not chart:
            return None
        bands = sorted((int(k), v) for k, v in chart.items())
        chosen = None
        for lower, split in bands:
            if loser_result >= lower:
                chosen = split
        return tuple(chosen) if chosen else None
