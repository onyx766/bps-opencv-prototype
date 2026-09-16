"""BPS_Scoring_TestSuite_v1.3, as tests.

One test per row of the client's CSV, named after its test id, so this file
reads against the spreadsheet line for line. `test_every_csv_row_has_a_test`
parses the ids back out of the method names and fails if the client adds a row
nobody has written a test for - which is the point of a locked suite.

No video and no OpenCV frames: the rules are pure state, and the session-level
rows drive GameSession's own methods with hand-built positions.

Run:  python -m unittest test_rules -v
"""

import collections
import importlib
import json
import os
import re
import tempfile
import unittest

import cv2
import numpy as np

import games
import logic
import rules
import server
import skill_level
import team_match
from detect_pocket import Pocket
from games import (ANYWHERE, BEHIND_HEAD_STRING, CASUAL, CUE, EIGHT, LEAGUE,
                   NINE_BALL, PRACTICE, SOLID, STRIPE, EightBallGame,
                   NineBallGame, facts)
from match_record import MatchRecord
from track_identity import Label

FPS = 10
R = 10
#: A 1000 x 500 landscape table, pockets on the corners and the long rails.
POCKETS = [Pocket("TL", 0, 0), Pocket("TM", 500, 0), Pocket("TR", 1000, 0),
           Pocket("BL", 0, 500), Pocket("BM", 500, 500), Pocket("BR", 1000, 500)]


def legal(**kw):
    """A shot that breaks no rule on its own: contact made, a rail reached."""
    base = dict(hit_object=True, rail_contact=True, rail_balls=4, moved=3)
    base.update(kw)
    return base


def shot(g, **kw):
    frame = g.frame + 10
    g.shot_started(frame)
    g.shot_ended(facts(**legal(**kw)), frame)
    return g


def eight(**kw):
    kw.setdefault("fps", FPS)
    return EightBallGame(**kw)


def nine(**kw):
    kw.setdefault("fps", FPS)
    return NineBallGame(**kw)


def dry_break(g):
    """Player 1 breaks legally, pots nothing; Player 2 comes to the table."""
    return shot(g, moved=10, rail_balls=5)


def on_solids(g):
    """Player 1 on solids with one down, at the table, in inning 2."""
    dry_break(g)            # P1 -> P2
    shot(g)                 # P2 misses -> P1
    shot(g, pots=[SOLID])   # P1 takes solids and stays at the table
    return g


def on_the_eight(g):
    on_solids(g)
    g.potted[SOLID] = games.GROUP_SIZE
    return g


def session(**kw):
    kw.setdefault("fps", FPS)
    return logic.GameSession(POCKETS, R, **kw)


def cited(g, kind=None):
    return [e.rule for e in g.record.live() if kind is None or e.kind == kind]


def triangle(cx=750, cy=250, n=5):
    return [(cx + k * 17.3, cy + (j - k / 2.0) * 20)
            for k in range(n) for j in range(k + 1)]


def judged(s, g, **kw):
    """A shot judged through the session, so it can be re-judged later."""
    frame = g.frame + 10
    f = facts(**legal(**kw))
    g.shot_started(frame)
    g.shot_ended(f, frame)
    s.last_facts, s.settled_at = f, frame
    return frame


class Fouls(unittest.TestCase):

    def test_F01_scratch_is_a_counter_not_a_deduction(self):
        g = on_solids(eight())
        before = g.score(0)
        shot(g, pots=[CUE])
        self.assertEqual(g.fouls, [1, 0])
        self.assertEqual(g.turn, 1)
        self.assertEqual(g.ball_in_hand, ANYWHERE)
        self.assertEqual(g.score(0), before)
        self.assertIn("F-01", cited(g, "foul"))

    def test_F02_scratch_on_break_restricts_ball_in_hand(self):
        g = eight()
        shot(g, pots=[CUE], moved=10, rail_balls=5)
        self.assertEqual(g.ball_in_hand, BEHIND_HEAD_STRING)
        self.assertIn("F-02", cited(g, "foul"))

        t = logic.TableGeometry(POCKETS, R)
        self.assertFalse(t.head_known)
        self.assertTrue(t.learn_foot(triangle()))      # rack at the right end
        self.assertTrue(t.head_known)
        self.assertEqual(t.head_string, 250)
        self.assertTrue(t.behind_head_string(100, 250))
        self.assertFalse(t.behind_head_string(600, 250))

    def test_F03_wrong_group_first_only_at_85_percent(self):
        g = on_solids(eight())
        shot(g, first_contact=STRIPE, contact_conf=0.9)
        self.assertEqual(g.fouls, [1, 0])
        self.assertIn("F-03", cited(g, "foul"))

        low = on_solids(eight())
        shot(low, first_contact=STRIPE, contact_conf=0.5)
        self.assertEqual(low.fouls, [0, 0])
        self.assertIn("F-09", cited(low, "review"))

    def test_F04_no_rail_after_contact(self):
        g = on_solids(eight())
        shot(g, rail_contact=False)
        self.assertEqual(g.fouls, [1, 0])
        self.assertIn("F-04", cited(g, "foul"))

    def test_F04_rails_follow_the_camera_angle_and_frame_rate(self):
        skewed = [Pocket("TL", 0, 0), Pocket("TM", 500, 0),
                  Pocket("TR", 1000, 0), Pocket("BL", 0, 500),
                  Pocket("BM", 480, 500), Pocket("BR", 940, 500)]
        t = logic.TableGeometry(skewed, R)
        self.assertTrue(t.on_rail(955, 450))      # against the slanted cushion
        self.assertFalse(t.on_rail(500, 250))

        slow = session(sample_fps=5)
        slow.rail_contact = False
        self.assertIsNone(slow._facts([], []).rail_contact)
        fast = session(sample_fps=30)
        fast.rail_contact = False
        self.assertIs(fast._facts([], []).rail_contact, False)

    def test_F05_no_object_ball_hit(self):
        g = on_solids(eight())
        shot(g, hit_object=False)
        self.assertEqual(g.fouls, [1, 0])
        self.assertIn("F-05", cited(g, "foul"))

    def test_F06_hand_foul_is_player_called_only(self):
        self.assertEqual(rules.rule("F-06").handling, rules.MANUAL)
        g = on_solids(eight())
        shot(g)
        self.assertEqual(g.fouls, [0, 0])
        self.assertTrue(g.mark_foul())
        self.assertEqual(g.fouls, [1, 0])          # against the shooter
        last = g.record.last({"foul"})[0]
        self.assertEqual((last.rule, last.actor), ("F-06", "player"))

    def test_F07_double_hit_is_never_ruled_on(self):
        self.assertEqual(rules.rule("F-07").handling, rules.LOGGED)
        g = on_solids(eight())
        for _ in range(4):
            shot(g)
        self.assertNotIn("F-07", cited(g, "foul"))

    def test_F08_frozen_balls_are_marked_not_judged(self):
        self.assertEqual(rules.rule("F-08").handling, rules.DEFERRED)
        s = session()
        s._frozen([(1, 5, 250, R), (2, 500, 250, R)],
                  {1: Label("solid", "voted"), 2: Label("solid", "voted")})
        self.assertEqual(s.frozen, [1])
        self.assertEqual(s.game.fouls, [0, 0])

    def test_F09_undetermined_contact_is_no_foul(self):
        g = on_solids(eight())
        shot(g, hit_object=None, contact_conf=0.0)
        self.assertEqual(g.fouls, [0, 0])
        self.assertIn("F-09", cited(g, "review"))

    def test_F10_foul_window_closes_at_the_next_stroke(self):
        g = on_solids(eight())
        shot(g)
        self.assertTrue(g.record.foul_window_open(g.shot))
        g.shot_started(g.frame + 10)
        self.assertFalse(g.mark_foul())
        self.assertEqual(g.fouls, [0, 0])
        self.assertIn("F-10", cited(g))


class Hud(unittest.TestCase):

    def test_H01_innings_are_anchored_on_the_breaker(self):
        g = eight()
        self.assertEqual(g.inning, 1)
        dry_break(g)
        self.assertEqual((g.turn, g.inning), (1, 1))
        shot(g)
        self.assertEqual((g.turn, g.inning), (0, 2))
        shot(g, pots=[SOLID])
        self.assertEqual((g.turn, g.inning), (0, 2))
        shot(g)
        self.assertEqual((g.turn, g.inning), (1, 2))
        shot(g)
        self.assertEqual((g.turn, g.inning), (0, 3))

    def test_H02_defence_button_is_transient_and_shot_bound(self):
        self.assertNotIn("defence",
                         [c[0] for c in session(defence_marking=True).commands()])
        g = eight(defence_marking=True)
        dry_break(g)
        self.assertTrue(g.defence_open)
        marked = g.defence_shot
        self.assertTrue(g.mark_defence())
        self.assertFalse(g.mark_defence())
        self.assertEqual(g.defences, [1, 0])
        g.shot_started(g.frame + 10)
        self.assertFalse(g.defence_open)
        self.assertFalse(g.mark_defence(marked))

        off = eight()
        dry_break(off)
        self.assertFalse(off.defence_open)

    def test_H03_skill_level_only_in_league_with_defence_marking(self):
        self.assertFalse(eight(mode=CASUAL, levels=(3, 6),
                               defence_marking=True).handicapped)
        self.assertFalse(eight(mode=LEAGUE, levels=(3, 6)).handicapped)
        self.assertIsNone(eight(mode=PRACTICE, levels=(3, 6),
                                defence_marking=True).race())
        self.assertEqual(eight(mode=CASUAL, levels=(3, 6)).level_text(0), "")
        g = eight(mode=LEAGUE, levels=(3, 6), defence_marking=True)
        self.assertTrue(g.handicapped)
        race = g.race()
        self.assertEqual(len(race), 2)
        self.assertTrue(all(n > 0 for n in race))

    def test_H04_skill_level_naming(self):
        skill_level.assert_clean()
        self.assertEqual(skill_level.describe(4), "Skill Level 4")
        self.assertEqual(skill_level.describe(None), "Unrated")
        self.assertEqual(skill_level.describe(12), "Skill Level 9")
        self.assertEqual(skill_level.describe(12, games.EIGHT_BALL),
                         "Skill Level 7")
        self.assertFalse(skill_level.clean("APA rated"))
        self.assertFalse(skill_level.clean("handicap 4"))
        s = session(mode=LEAGUE, levels=(4, 5), defence_marking=True)
        for _cid, label, _on in s.commands():
            self.assertTrue(skill_level.clean(label), label)
        self.assertTrue(skill_level.clean(json.dumps(s.summary())))

    def test_H05_timeouts_commit_on_close(self):
        g = eight(levels=(3, 5))
        self.assertEqual(g.timeouts.allowance, [2, 1])
        self.assertEqual(skill_level.timeouts(None), 2)

        self.assertTrue(g.start_timeout())
        self.assertTrue(g.end_timeout())
        self.assertEqual(g.timeouts.used, [0, 0])     # rack not struck yet

        dry_break(g)                                   # Player 2 at the table
        self.assertTrue(g.start_timeout())
        self.assertTrue(g.cancel_timeout())
        self.assertEqual(g.timeouts.used, [0, 0])     # mis-tap: no charge
        self.assertEqual(len(g.record.last({"timeout"}, n=9)), 1)

        self.assertTrue(g.start_timeout())
        g.frame += 30 * FPS
        self.assertTrue(g.end_timeout())
        self.assertEqual(g.timeouts.used, [0, 1])
        self.assertEqual(g.timeouts.remaining(1), 0)
        last = g.record.last({"timeout"})[0]
        self.assertTrue(last.data["charged"])
        self.assertEqual(last.data["seconds"], 30.0)


class Mechanics(unittest.TestCase):

    def test_M01_turn_advances_without_a_prompt(self):
        g = on_solids(eight())
        shot(g, pots=[SOLID])
        self.assertEqual(g.turn, 0)
        shot(g)
        self.assertEqual(g.turn, 1)
        self.assertNotIn("prompt", [e.kind for e in g.record.events])

        # Potting only an opponent's ball passes the turn. It is not a foul.
        opp = on_solids(eight())
        shot(opp, pots=[STRIPE])
        self.assertEqual((opp.turn, opp.fouls, opp.potted[STRIPE]),
                         (1, [0, 0], 1))

    def _hanging(self, seconds, still_there=False):
        s = session()
        g = s.game
        s.settled_balls = 16
        frame = judged(s, g, moved=10, rail_balls=5)      # a dry break
        self.assertEqual(g.turn, 1)
        s.resting = [(14, 14), (500, 250)]      # one ball sat in the TL jaw
        s.frame = frame + int(seconds * FPS)
        s.since_clear = logic.SETTLE_FRAMES
        if still_there:
            s.window.extend([(True, [(14, 14), (500, 250)])] * logic.SETTLE_FRAMES)
        s._hold_hanging([logic.Pot(s.frame, 7, SOLID, "TL", 20.0, 5.0, 50,
                                   False)])
        s._confirm_hanging()
        return s, g

    def test_M02_hanging_ball_counts_inside_five_seconds_only(self):
        s, g = self._hanging(3)
        self.assertEqual((g.turn, g.potted[SOLID]), (0, 1))  # breaker's shot
        self.assertIn("M-02", cited(g))

        late, lg = self._hanging(6)
        self.assertEqual((lg.turn, lg.potted[SOLID]), (1, 0))
        self.assertEqual(late.prompt.kind, "replace")

        # A player leaning over a pocket: a young track lost under a hand is
        # refused outright, and an old one is refused once the clear table
        # turns out to still hold every ball.
        crowd = session()
        cg = crowd.game
        crowd.settled_balls = 16
        judged(crowd, cg, moved=10, rail_balls=5)
        crowd.since_clear = logic.SETTLE_FRAMES
        crowd.recent.extend([(True, 16)] * logic.SETTLE_FRAMES)
        crowd._hold_hanging([
            logic.Pot(cg.frame + 5, 7, SOLID, "TL", 20.0, 5.0, 50, False),
            logic.Pot(cg.frame + 5, 8, SOLID, "TL", 20.0, 5.0, 1, True)])
        self.assertEqual(len(crowd.hanging), 1)
        crowd._confirm_hanging()
        self.assertEqual((cg.turn, cg.potted[SOLID]), (1, 0))
        self.assertEqual(crowd.hanging, [])

        # A dip in the count while the ball is still sitting in the jaw is a
        # ball blinking out of detection, not a ball dropping: no re-judge, no
        # turn handed back, no point.
        blink, bg = self._hanging(3, still_there=True)
        self.assertEqual((bg.turn, bg.potted[SOLID]), (1, 0))
        self.assertNotIn("M-02", cited(bg))

    def test_M05_a_settling_never_seen_clear_is_not_judged(self):
        # Occlusion is normal play: a hand held over the table until the next
        # stroke begins gives nothing to judge by, so no shot is invented.
        s = session()
        g = s.game
        s.settled_balls = 16
        judged(s, g, moved=10, rail_balls=5)
        s.resting = [(100, 100), (200, 200), (300, 300)]
        s.window.extend([(False, [(100, 100)])] * logic.SETTLE_FRAMES)
        s.recent.extend([(False, 1)] * logic.SETTLE_FRAMES)
        turn, shots = g.turn, g.shot
        self.assertEqual(s._settle(g.frame + 10), [])
        self.assertEqual((g.turn, g.shot, g.fouls), (turn, shots, [0, 0]))
        self.assertEqual(s.resting, [(100, 100), (200, 200), (300, 300)])

    def test_M03_undo_until_the_next_break(self):
        g = on_solids(eight())
        shot(g)
        self.assertEqual(g.turn, 1)
        self.assertTrue(g.undo())
        self.assertEqual(g.turn, 0)
        self.assertTrue(any(e.data.get("reverted") for e in g.record.events))
        self.assertTrue(g.record.unconfirmed)

        g.new_rack(g.frame + 10)
        self.assertFalse(g.record.can_undo)
        self.assertTrue(any("SCORE UNCONFIRMED" in e.text
                            for e in g.record.events))

        dry_break(g)
        self.assertTrue(g.record.can_undo)
        self.assertTrue(g.confirm_final())
        self.assertFalse(g.record.can_undo)
        self.assertTrue(g.reopen("Player 2"))
        self.assertFalse(g.record.final)
        self.assertEqual(g.record.last({"reopen"})[0].data["who"], "Player 2")

    def _phantom(self, late):
        s = session()
        g = s.game
        on_solids(g)
        frame = judged(s, g, pots=[SOLID])
        pot = logic.Pot(frame, 9, SOLID, "TL", 30.0, 10.0, 50, False)
        s.last_credited, s.all_pots = [pot], [pot]
        s.resting, s.settled_balls = [(300, 300), (600, 200)], 14
        self.assertEqual((g.potted[SOLID], g.turn), (2, 0))
        s._reconcile_phantoms([(300, 300), (600, 200), (25, 25)],
                              frame + (3 * FPS if late else FPS // 2))
        self.assertEqual((g.potted[SOLID], g.turn), (1, 1))
        self.assertEqual(s.phantoms, 1)
        return g

    def test_M04_W10_ball_back_on_the_table_was_never_potted(self):
        self.assertNotIn("W-10", cited(self._phantom(late=False)))
        self.assertIn("W-10", cited(self._phantom(late=True)))

    def test_M05_occlusion_is_quiet(self):
        s = session()
        s.update([], {}, 1, intruding=True)
        self.assertTrue(s.busy)
        s.update([], {}, 2)
        self.assertTrue(s.busy)
        s.update([], {}, 3)
        self.assertFalse(s.busy)
        self.assertNotIn("BLOCKED", s.status)

    def test_M06_shaky_colour_asks_to_confirm_groups(self):
        s = session()
        g = s.game
        dry_break(g)
        shot(g)                                   # P1 at the table, open
        frame = judged(s, g, pots=[STRIPE])
        pot = logic.Pot(frame, 5, STRIPE, "TL", 20.0, 5.0, 50, False)
        s.label_conf[5] = 0.5
        self.assertTrue(s._confirm_groups(0, [pot], was_open=True))
        self.assertEqual(s.prompt.kind, "groups")
        self.assertFalse(g.groups_confirmed)
        self.assertTrue(s.answer("groups", SOLID))
        self.assertEqual(g.group, [SOLID, STRIPE])
        self.assertTrue(g.groups_confirmed)

        s.label_conf[5] = 0.9
        self.assertFalse(s._confirm_groups(0, [pot], was_open=True))

    def test_M07_ball_moved_by_hand_is_never_a_foul(self):
        s = session()
        s.game.struck = True
        s._replaced([(120, 340)], 50)
        self.assertEqual(s.replace_at, (120, 340))
        self.assertEqual(s.game.fouls, [0, 0])
        self.assertIn("M-07", cited(s.game))
        self.assertEqual(len(s.reviews), 1)

    def test_M08_missing_cue_ball_is_a_wait(self):
        s = session()
        for _ in range(logic.CUE_MISSING_FRAMES):
            s._cue_ball_missing({1: Label("solid", "voted")})
        self.assertTrue(s.waiting_for_cue)
        self.assertEqual(s.status, "WAITING FOR THE CUE BALL")
        self.assertEqual(s.game.fouls, [0, 0])
        s._cue_ball_missing({2: Label("cue", "voted")})
        self.assertFalse(s.waiting_for_cue)

    def test_M09_rack_detection_and_first_rack_confirmation(self):
        s = session(first_rack_confirmed=False)
        grid = [(60 + 80 * (i % 5), 60 + 80 * (i // 5)) for i in range(15)]
        self.assertTrue(s._racked(triangle()))
        self.assertFalse(s._racked(grid))
        self.assertTrue(session(discipline=NINE_BALL)._racked(triangle(n=4)[:9]))

        s.game.struck = True
        self.assertFalse(s._new_rack(10, triangle()))
        self.assertEqual((s.prompt.kind, s.game.rack), ("rack", 1))
        self.assertTrue(s.answer("rack", "yes"))
        self.assertTrue(s._new_rack(20, triangle()))
        self.assertEqual(s.game.rack, 2)
        self.assertTrue(s.table.head_known)

        # A rack is racked BY HAND, with both players standing over it: in the
        # footage the second triangle sat for 130 frames and not one was clear.
        # It must still be seen with a hand on the table the whole time.
        s = session()
        s.game.struck = True
        balls = [(i, x, y, R) for i, (x, y) in enumerate(triangle())]
        for frame in range(logic.RACK_FRAMES + 2):
            s.update(balls, {}, frame, intruding=True)
        self.assertEqual(s.game.rack, 2)

    def test_M10_break_legality_enforced_in_league_only(self):
        # League: the manual's remedy (3.3) - re-rack, same breaker again.
        g = eight(mode=LEAGUE)
        shot(g, moved=8, rail_balls=2)
        self.assertFalse(g.struck)
        self.assertEqual((g.fouls, g.turn, g.breaker), ([0, 0], 0, 0))
        self.assertIn("M-10", cited(g, "rerack"))
        # ... and the other player breaks if the illegal break scratched.
        s = eight(mode=LEAGUE)
        shot(s, pots=[CUE], moved=8, rail_balls=2)
        self.assertEqual((s.struck, s.breaker, s.turn), (False, 1, 1))

        c = eight(mode=CASUAL)
        shot(c, moved=8, rail_balls=2)
        self.assertFalse(c.illegal_break)
        self.assertIn("M-10", cited(c))

        unknown = eight(mode=LEAGUE)
        shot(unknown, moved=8, rail_balls=None)
        self.assertFalse(unknown.illegal_break)

    def test_M11_practice_has_no_fouls_turns_or_wins(self):
        g = eight(mode=PRACTICE)
        shot(g, pots=[CUE, SOLID])
        shot(g, pots=[EIGHT])
        shot(g)
        self.assertEqual((g.fouls, g.turn, g.over, g.winner),
                         ([0, 0], 0, False, None))
        a = g.analytics
        self.assertEqual(a.shots[0], 3)
        self.assertAlmostEqual(a.tsr(0), 2 / 3)
        self.assertAlmostEqual(a.crr(0), 2 / 3)

    def test_M12_every_rack_writes_an_event_log(self):
        g = on_solids(eight())
        shot(g, pots=[CUE])
        for e in g.record.events:
            self.assertIn("frame", e.as_dict()["clip"])
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            g.record.save(path, summary=g.summary())
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        finally:
            os.remove(path)
        self.assertEqual(data["summary"]["fouls"], [1, 0])
        foul = [e for e in data["events"] if e["type"] == "foul"][0]
        self.assertEqual(foul["rule"], "F-01")

    def test_M13_review_pulls_footage_and_does_not_overrule(self):
        s = session()
        g = on_solids(s.game)
        score = g.score(0)
        self.assertTrue(s.command("review"))
        last = g.record.last()[0]
        self.assertEqual((last.kind, last.actor, last.rule),
                         ("review", "player", "M-13"))
        self.assertEqual(g.score(0), score)

    def test_M14_sub_threshold_calls_are_marked(self):
        g = on_solids(eight())
        shot(g, first_contact=STRIPE, contact_conf=0.4)
        self.assertTrue(any(e.uncertain for e in g.record.flagged()))
        s = session()
        weak = logic.Pot(1, 1, SOLID, "TL", 20.0, 0.0, 1, True)
        strong = logic.Pot(1, 1, SOLID, "TL", 20.0, 10.0, 50, False)
        self.assertLess(s.pots.confidence(weak), rules.CONTACT_CONFIDENCE)
        self.assertGreaterEqual(s.pots.confidence(strong),
                                rules.CONTACT_CONFIDENCE)

    def test_M15_players_by_table_side(self):
        self.assertEqual(eight().players, ["Player 1", "Player 2"])
        self.assertEqual(session().game.players, ["Player 1", "Player 2"])


class NineBall(unittest.TestCase):

    def test_N01_lowest_ball_first_needs_numbers(self):
        g = nine()
        dry_break(g)
        g.lowest = 2
        shot(g, first_contact=5, contact_conf=0.9)
        self.assertEqual(g.fouls, [0, 1])
        self.assertIn("N-01", cited(g, "foul"))

        blind = nine()
        dry_break(blind)
        shot(blind)
        shot(blind)
        self.assertEqual(blind.fouls, [0, 0])
        self.assertEqual(len(blind.record.last({"review"}, n=9)), 1)

    def test_N02_nine_on_the_snap(self):
        g = nine()
        shot(g, pots=[SOLID, STRIPE], moved=9)
        self.assertEqual((g.over, g.winner), (True, 0))
        self.assertIn("N-02", cited(g))
        self.assertTrue(g.summary()["invariant_ok"])

        s = nine()
        shot(s, pots=[STRIPE, CUE], moved=9)
        self.assertFalse(s.over)
        self.assertEqual((s.fouls, s.points), ([1, 0], [0, 0]))

    def test_N03_nine_on_a_foul_is_spotted(self):
        g = nine()
        dry_break(g)
        shot(g, pots=[STRIPE, CUE])
        self.assertFalse(g.over)
        self.assertEqual((g.turn, g.ball_in_hand, g.points), (0, ANYWHERE, [0, 0]))
        self.assertIn("N-03", cited(g))

    def test_N04_dead_balls_and_the_ten_point_invariant(self):
        g = nine()
        shot(g, pots=[SOLID, SOLID], moved=9)       # P1 breaks, makes 2, stays
        self.assertEqual((g.points, g.turn), ([2, 0], 0))
        shot(g, pots=[SOLID, CUE])                  # scratch: that ball is dead
        self.assertEqual((g.points, g.dead, g.turn), ([2, 0], 1, 1))
        shot(g, pots=[SOLID])                       # P2 makes one
        shot(g, pots=[STRIPE])                      # ... and the 9, early
        self.assertTrue(g.over)
        self.assertEqual(g.points, [2, 3])
        self.assertEqual(g.points[0] + g.points[1] + g.dead, 10)
        self.assertTrue(g.summary()["invariant_ok"])
        self.assertNotIn("impossible", [e.kind for e in g.record.events])


class Winning(unittest.TestCase):

    def test_W01_legal_eight_wins(self):
        g = on_the_eight(eight())
        shot(g, pots=[EIGHT])
        self.assertEqual((g.over, g.winner), (True, 0))
        self.assertIn("W-01", cited(g))

        m = on_the_eight(eight(pocket_marking=True))
        m.call_pocket("TL")
        shot(m, pots=[EIGHT], eight_pocket="TL")
        self.assertEqual(m.winner, 0)

    def test_W02_eight_early_loses_with_an_undo_window(self):
        g = on_solids(eight())
        shot(g, pots=[EIGHT])
        self.assertEqual((g.over, g.winner), (True, 1))
        self.assertIn("W-02", cited(g))
        self.assertIsNotNone(g.undo_until)
        self.assertTrue(g.undo())
        self.assertEqual((g.over, g.turn, g.racks_won), (False, 0, [0, 0]))

    def test_W03_wrong_pocket_only_with_marking(self):
        self.assertTrue(eight(mode=LEAGUE).pocket_marking)
        g = on_the_eight(eight(pocket_marking=True))
        g.call_pocket("TL")
        shot(g, pots=[EIGHT], eight_pocket="BR")
        self.assertEqual(g.winner, 1)
        self.assertIn("W-03", cited(g))

        off = on_the_eight(eight(mode=CASUAL))
        self.assertFalse(off.call_pocket("TL"))
        shot(off, pots=[EIGHT], eight_pocket="BR")
        self.assertEqual(off.winner, 0)

    def test_W04_eight_with_last_group_ball(self):
        g = on_solids(eight())
        g.potted[SOLID] = 6
        shot(g, pots=[SOLID, EIGHT])
        self.assertEqual(g.winner, 1)
        self.assertIn("W-04", cited(g))

    def test_W05_scratch_shooting_at_the_eight_loses(self):
        g = on_the_eight(eight())
        shot(g, pots=[CUE])
        self.assertEqual((g.over, g.winner), (True, 1))
        self.assertIn("W-05", cited(g))

    def test_W06_missing_the_eight_is_a_foul_not_a_loss(self):
        g = on_the_eight(eight())
        shot(g, hit_object=False)
        self.assertFalse(g.over)
        self.assertEqual((g.fouls, g.turn, g.ball_in_hand), ([1, 0], 1, ANYWHERE))
        self.assertIn("W-06", cited(g, "foul"))
        self.assertNotIn("W-05", cited(g))

    def test_W07_eight_off_the_table_loses(self):
        g = on_solids(eight())
        shot(g, off_table=[EIGHT], off_conf=0.6)
        self.assertEqual((g.over, g.winner), (True, 1))
        self.assertIn("W-07", cited(g))
        self.assertIsNotNone(g.undo_until)          # inferred, so undoable

    def test_W08_groups_and_the_open_table(self):
        # Setting off: the CSV's W-08 - the break never assigns.
        g = eight(break_assigns=False)
        shot(g, pots=[SOLID], moved=10)
        self.assertTrue(g.open_table)
        self.assertEqual(g.turn, 0)
        shot(g, pots=[STRIPE])
        self.assertEqual(g.group, [STRIPE, SOLID])
        self.assertIn("W-08", cited(g, "assign"))
        # Default, the manual (3.4c): one category on the break takes it ...
        m = eight()
        shot(m, pots=[SOLID, SOLID], moved=10)
        self.assertEqual(m.group, [SOLID, STRIPE])
        # ... and one of each leaves it open, on the break and after it (3.4d).
        o = eight()
        shot(o, pots=[SOLID, STRIPE], moved=10)
        self.assertTrue(o.open_table)
        shot(o, pots=[SOLID, STRIPE])
        self.assertTrue(o.open_table)

    def test_W09_stalemate_voids_the_rack(self):
        self.assertEqual(rules.rule("W-09").handling, rules.MANUAL)
        g = eight(defence_marking=True)
        dry_break(g)
        g.mark_defence()
        shot(g)
        shot(g)
        self.assertEqual(g.defences, [1, 0])
        self.assertTrue(g.stalemate())
        self.assertEqual((g.over, g.winner, g.defences), (True, None, [0, 0]))
        g.new_rack(g.frame + 10)
        self.assertEqual(g.innings_total, 0)

        n = eight()
        dry_break(n)
        shot(n)
        n.new_rack(n.frame + 10)
        # One COMPLETED inning: the lag loser's miss closed it (manual 5).
        self.assertEqual(n.innings_total, 1)

    def test_W11_wedged_pair_prompts_and_never_ends_the_rack(self):
        s = session()
        s.game.struck = True
        wedged = [(1, 10, 12, R), (2, 22, 8, R)]
        labels = {1: Label("solid", "voted"), 2: Label("eight", "voted")}
        for _ in range(logic.RACK_FRAMES):
            s._wedged(wedged, labels)
        self.assertEqual(s.prompt.kind, "wedged")

        e = session()
        g = e.game
        on_solids(g)
        g.potted[SOLID] = 6
        judged(e, g)                                 # P1 misses -> P2
        e._credit_wedged([SOLID, EIGHT])             # would be W-04
        self.assertFalse(g.over)
        self.assertEqual((g.potted[SOLID], g.potted[EIGHT], g.turn), (7, 1, 1))
        self.assertIn("W-11", cited(g))

        t = session()
        h = t.game
        on_solids(h)
        judged(t, h)
        t._credit_wedged([SOLID, SOLID])
        self.assertEqual((h.potted[SOLID], h.turn), (3, 0))


class Manual(unittest.TestCase):
    """The league team manual's rules, by section, beyond the CSV rows."""

    def test_TM3_TM4_eight_on_the_break(self):
        g = eight()
        shot(g, pots=[EIGHT], moved=10)
        self.assertEqual((g.over, g.winner), (True, 0))
        lost = eight()
        shot(lost, pots=[EIGHT, CUE], moved=10)
        self.assertEqual((lost.over, lost.winner), (True, 1))

    def test_TM4_eight_hit_first_on_an_open_table(self):
        g = eight()
        dry_break(g)
        shot(g, first_contact=EIGHT, contact_conf=0.9)
        self.assertEqual(g.fouls, [0, 1])

    def test_TM1_winner_breaks_next_rack(self):
        g = eight(lag_winner=1)
        self.assertEqual((g.turn, g.breaker), (1, 1))
        g.over, g.winner = True, 0
        g.new_rack(100)
        self.assertEqual(g.breaker, 0)

    def test_SC8_break_and_run_marks_no_inning(self):
        g = eight()
        shot(g, pots=[SOLID], moved=10)          # breaker takes solids
        g.potted[SOLID] = games.GROUP_SIZE
        shot(g, pots=[EIGHT])
        self.assertEqual((g.winner, g.inning - 1), (0, 0))
        g.new_rack(g.frame + 10)
        self.assertEqual(g.innings_total, 0)

    def test_SC9_points_past_the_target_are_not_marked(self):
        g = nine(mode=LEAGUE, levels=(1, 9), defence_marking=True)
        self.assertEqual(g.race(), (14, 75))
        g.points[0] = 20
        self.assertEqual(g.score(0), 14)

    def test_TM13_nine_ball_stalemate_points_stand(self):
        g = nine()
        shot(g, pots=[SOLID, SOLID], moved=9)
        self.assertTrue(g.stalemate())
        self.assertEqual((g.points, g.dead), ([2, 0], 6))
        g.new_rack(g.frame + 10)
        self.assertEqual(g.breaker, 0)            # same breaker re-breaks

    def test_TM15_foul_on_the_winning_eight_loses(self):
        g = on_the_eight(eight())
        shot(g, pots=[EIGHT])
        self.assertEqual(g.winner, 0)
        self.assertTrue(g.mark_foul("double-hit"))
        self.assertEqual((g.winner, g.racks_won), (1, [0, 1]))
        self.assertIn("TM-16", cited(g, "win"))

    def test_TM16_unmarked_eight_is_the_opponents_call(self):
        g = on_the_eight(eight(pocket_marking=True))
        shot(g, pots=[EIGHT])
        self.assertEqual((g.winner, g.unmarked_claim), (0, 0))
        self.assertTrue(g.call_unmarked_loss())
        self.assertEqual((g.winner, g.racks_won), (1, [0, 1]))

    def test_TM5_uncalled_wrong_group_swaps_categories(self):
        g = on_solids(eight())
        self.assertTrue(g.swap_groups())
        self.assertEqual(g.group, [STRIPE, SOLID])

    def test_TM4_push_out_only_in_masters(self):
        self.assertFalse(nine().push_out())
        m = nine(fmt="masters")
        dry_break(m)
        self.assertTrue(m.push_out())

    def test_EQ1_race_charts(self):
        c = skill_level.RaceChart()
        for mine, theirs, a, b in ((2, 7, 2, 7), (4, 7, 2, 5), (5, 7, 3, 5),
                                   (6, 7, 4, 5), (3, 3, 2, 2), (7, 7, 5, 5)):
            self.assertEqual(c.race(games.EIGHT_BALL, mine, theirs), (a, b))
        self.assertEqual(c.nine_ball(9), 75)
        self.assertEqual(c.nine_ball(None), 25)          # unrated races as 3
        self.assertEqual(c.doubles(games.EIGHT_BALL, (3, 3), (4, 4)), (2, 3))
        self.assertEqual(c.doubles(games.NINE_BALL, (2, 2), (5, 6)), (19, 42))
        self.assertEqual(c.race(games.NINE_BALL, 2, 2, fmt="masters"), (7, 7))
        self.assertFalse(c.provisional)

    def test_EQ2_starting_levels(self):
        self.assertEqual(skill_level.carry_over(1, "9ball", "8ball"), 2)
        self.assertEqual(skill_level.carry_over(9, "9ball", "8ball"), 7)
        self.assertEqual(skill_level.carry_over(5, "8ball", "9ball"), 5)
        self.assertEqual(skill_level.clamp(9, "8ball"), 7)

    def test_GR18_timeouts_per_format(self):
        self.assertEqual(skill_level.timeouts(3), 2)
        self.assertEqual(skill_level.timeouts(5), 1)
        self.assertEqual(skill_level.timeouts(3, "doubles"), 1)
        self.assertEqual(skill_level.timeouts(3, "masters"), 0)

    def test_GR25_GR26_team_limits(self):
        roster = [7, 6, 5, 3, 3, 2]
        self.assertTrue(team_match.check_lineup([7, 6, 5], roster).ok)
        self.assertFalse(team_match.check_lineup([7, 6, 6], [7, 6, 6, 2, 2]).ok)
        bad = team_match.check_lineup([7, 7, 5, 3], [7, 7, 5, 3, 3])
        self.assertFalse(bad.ok)
        self.assertEqual(team_match.reduced_lineup([5, 5, 5, 5, 5]),
                         (3, 15, 2))
        self.assertIsNone(team_match.reduced_lineup([2, 3, 4, 5, 6]))
        self.assertEqual(team_match.limit_penalty(4, "open", "8ball", 3),
                         (0, 8))

    def test_GR12_GR14_GR15_GR23_points_and_order(self):
        self.assertEqual(team_match.forfeit_points("open", "8ball"), 2)
        self.assertEqual(team_match.forfeit_points("open", "8ball", True), 3)
        self.assertEqual(team_match.forfeit_points("open", "9ball", True), 20)
        self.assertEqual(team_match.bye_points("open", "9ball"), 60)
        self.assertEqual(team_match.team_forfeit_points("masters", "8ball"), 15)
        self.assertEqual(team_match.declares_first(True, 0), 0)
        self.assertEqual(team_match.declares_first(True, 1), 1)
        self.assertEqual(team_match.lowest_attainable(5), 4)
        self.assertIsNone(team_match.MatchPoints().award("8ball", 1))

    def test_GR27_playoffs(self):
        self.assertEqual(team_match.clinched((10, 4), 5), 0)
        self.assertIsNone(team_match.clinched((10, 6), 5))
        self.assertEqual(team_match.playoff_tie_winner("open", [1, 0, 1]), 1)
        self.assertEqual(team_match.playoff_tie_winner("doubles", [0, 0, 1]), 1)
        self.assertEqual(team_match.playoff_tie_winner("ladies", [1, 0]), 1)
        self.assertEqual(team_match.standings_tie((12, 12), (3, 5)), 1)
        self.assertEqual(team_match.standings_tie(None,
                                                  weekly_points=[(8, 8), (9, 7)]),
                         0)
        self.assertTrue(team_match.wild_card_eligible(5))
        self.assertFalse(team_match.wild_card_eligible(6))

    def test_manual_rules_point_at_real_code(self):
        ids = [r.id for r in rules.MANUAL_RULES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & set(rules.BY_ID))
        for r in rules.MANUAL_RULES:
            module, *attrs = r.where.split(".")
            obj = importlib.import_module(module)
            for attr in attrs:
                self.assertTrue(hasattr(obj, attr), f"{r.id}: {r.where}")
                obj = getattr(obj, attr)


class Spec(unittest.TestCase):

    def test_every_csv_row_has_a_test(self):
        spec = rules.read_spec()
        if not spec:
            self.skipTest("client CSV not present")
        tested = set()
        loader = unittest.TestLoader()
        for case in (Fouls, Hud, Mechanics, NineBall, Winning):
            for name in loader.getTestCaseNames(case):
                tested.update(f"{a}-{b}"
                              for a, b in re.findall(r"_([A-Z])(\d\d)", name))
        self.assertEqual(sorted(set(spec) - tested), [])

    def test_spec_table_matches_csv(self):
        self.assertEqual(rules.check(), ([], [], []))

    def test_every_rule_points_at_real_code(self):
        for r in rules.RULES:
            module, *attrs = r.where.split(".")
            obj = importlib.import_module(module)
            for attr in attrs:
                self.assertTrue(hasattr(obj, attr), f"{r.id}: {r.where}")
                obj = getattr(obj, attr)

    def test_record_keeps_reverted_events(self):
        rec = MatchRecord()
        rec.snapshot({"x": 1}, "SHOT 1")
        rec.add(1, "00:00", 1, 1, "foul", "FOUL")
        self.assertEqual(rec.undo(2, "00:00", 1, 1), {"x": 1})
        self.assertEqual(len(rec.events), 2)
        self.assertEqual(len(rec.live()), 1)


class Detection(unittest.TestCase):
    """Not a CSV row: the intrusion filter every count and pot rests on."""

    BR = 20

    def masks(self):
        table = np.zeros((700, 500), np.uint8)
        table[10:690, 30:470] = 255
        return table, np.zeros_like(table)

    def test_ball_against_the_cushion_strip_is_not_an_intrusion(self):
        # The cushion nose reads as a thin non-felt strip touching the table
        # edge. A ball resting on it used to be erased with it as "too long" /
        # "reaches in", so the table counted one ball short and a pot off that
        # rail was never seen.
        table, nf = self.masks()
        # As on the real table: the strip runs a little inside the mask's edge
        # and meets it only at its ends.
        cv2.rectangle(nf, (60, 12), (68, 688), 255, -1)
        cv2.circle(nf, (87, 300), self.BR, 255, -1)
        cv2.circle(nf, (87, 460), self.BR, 255, -1)
        _labels, found = server.find_intrusions(nf, self.BR, table)
        self.assertEqual(found, [])
        balls = server.find_balls(nf, self.BR, table=table)
        self.assertEqual(sorted((round(y, -1)) for _x, y, _r in balls), [300, 460])

    def test_fingers_over_the_cushion_are_still_an_intrusion(self):
        # Fingers resting over the rail cross the same strip, and each one is
        # ball-sized once it is shed - but they run up to the table's edge,
        # which a ball resting on the cushion does not.
        table, nf = self.masks()
        cv2.rectangle(nf, (60, 12), (68, 688), 255, -1)
        for y in (200, 240, 280):
            cv2.circle(nf, (50, y), self.BR, 255, -1)    # over the edge
        _labels, found = server.find_intrusions(nf, self.BR, table)
        self.assertEqual(len(found), 1)

    def test_hand_resting_on_the_rail_is_an_intrusion_away_from_pockets(self):
        # A hand laid on a rail is small and tidy - two fingers read as two
        # balls - but it runs up to the table's outer edge. A ball in a
        # pocket's jaw reaches that edge too, so the pockets are left out.
        table, nf = self.masks()
        rim = server.outer_rim(table, [Pocket("BL", 30, 690)], 5 * self.BR)
        cv2.rectangle(nf, (30, 300), (90, 360), 255, -1)     # on the rail
        cv2.rectangle(nf, (30, 620), (90, 670), 255, -1)     # at the pocket
        _labels, found = server.find_intrusions(nf, self.BR, table, rim=rim)
        self.assertEqual([it.why for it in found], ["rests over the rail"])
        self.assertLess(found[0].y, 400)

    def test_arm_cue_and_bridge_hand_are_still_intrusions(self):
        table, nf = self.masks()
        # An arm reaching in over the edge, and a cue lying on the cloth.
        cv2.line(nf, (30, 600), (330, 300), 255, 2 * self.BR)
        cv2.circle(nf, (330, 300), 2 * self.BR, 255, -1)
        cv2.line(nf, (400, 40), (400, 640), 255, 24)
        # A bridge hand whose cue tip is as thin as the cushion strip: shedding
        # the tip must not bring the hand in under the length limit.
        cv2.circle(nf, (130, 120), 46, 255, -1)
        cv2.line(nf, (176, 120), (370, 120), 255, 12)
        _labels, found = server.find_intrusions(nf, self.BR, table)
        self.assertEqual(len(found), 3)


class Pots(unittest.TestCase):
    """Not a CSV row: which balls a settled shot actually lost (5:32 in game.mp4)."""

    def settled(self, before=9, after=8):
        s = session()
        s.before = before
        s.recent.extend([(True, after)] * logic.SETTLE_FRAMES)
        return s

    def test_a_ball_still_sitting_where_it_was_is_not_credited(self):
        s = self.settled()
        s.window.extend([(True, [(400, 30), (700, 300)])] * logic.SETTLE_FRAMES)
        still = logic.Pot(100, 1, STRIPE, "TM", 30.0, 0.0, 226, False, 400, 30)
        went = logic.Pot(101, 2, SOLID, "BR", 40.0, 20.0, 50, True, 960, 470)
        s.candidates = [still, went]
        self.assertEqual([p.track for p in s._settle_up(101)], [2])

    def test_a_fast_ball_lost_short_of_the_pocket_only_fills_a_gap(self):
        d = logic.PotDetector(POCKETS, R)
        d.update([(1, 60, 50, R)], {}, 1)          # 78 px out: past the mouth
        for f in range(2, 2 + logic.POT_CONFIRM):
            out = d.update([], {}, f)
        self.assertEqual((out, [p.track for p in d.far]), ([], [1]))

        s = self.settled()
        s.far_claims = [logic.Pot(100, 4, STRIPE, "TL", 81.0, 0.0, 1, True, 60, 55)]
        self.assertEqual([p.track for p in s._settle_up(101)], [4])

        # An established track dying out there lost its id, not its ball: the
        # cue ball at 7:30 was never potted however far the count was out.
        old = self.settled()
        old.far_claims = [logic.Pot(100, 6, CUE, "BR", 117.0, -32.0, 248, False,
                                    900, 400)]
        self.assertEqual(old._settle_up(101), [])

        # ... and a young one is never the cue ball while the cue is in view.
        blip = self.settled()
        blip.frame = 101
        blip.last_seen[CUE] = 100
        blip.far_claims = [logic.Pot(100, 7, CUE, "TM", 119.0, 0.0, 1, False,
                                     560, 110)]
        self.assertEqual(blip._settle_up(101), [])
        s.candidates = [logic.Pot(100, 5, SOLID, "BR", 40.0, 20.0, 50, False,
                                  960, 470)]
        self.assertEqual([p.track for p in s._settle_up(101)], [5])

    def test_a_ball_gone_from_the_census_is_scored_without_a_claim(self):
        # 4:29: a solid went down, its track died 232 px from any pocket, and
        # nothing was credited. The classifier still shows it gone.
        s = self.settled(before=6, after=5)
        s.before_census = collections.Counter({STRIPE: 3, SOLID: 1, EIGHT: 1,
                                               CUE: 1})
        s.census.extend([(True, collections.Counter({STRIPE: 3, EIGHT: 1,
                                                     CUE: 1}))] * logic.SETTLE_FRAMES)
        credited = s._settle_up(101)
        self.assertEqual([(p.cls, p.pocket, p.track) for p in credited],
                         [(SOLID, None, -1)])
        self.assertLess(s.pots.confidence(credited[0]), rules.CONTACT_CONFIDENCE)

        # A class that comes back in any frame of the window was mislabelled
        # for a moment, not potted - even with the count a ball down.
        flicker = self.settled(before=6, after=5)
        flicker.before_census = collections.Counter({STRIPE: 3, SOLID: 1,
                                                     EIGHT: 1, CUE: 1})
        flicker.census.extend([(True, collections.Counter({STRIPE: 2, SOLID: 2,
                                                           EIGHT: 1, CUE: 1}))]
                              * (logic.SETTLE_FRAMES - 1))
        flicker.census.append((True, collections.Counter({STRIPE: 3, SOLID: 1,
                                                          EIGHT: 1, CUE: 1})))
        self.assertEqual(flicker._settle_up(101), [])

        # Two balls gone and the same two classes short: the count and the
        # census are two witnesses agreeing, and both balls are scored.
        two = self.settled(before=6, after=4)
        two.before_census = collections.Counter({STRIPE: 3, SOLID: 1, EIGHT: 1,
                                                 CUE: 1})
        two.census.extend([(True, collections.Counter({STRIPE: 2, EIGHT: 1,
                                                       CUE: 1}))] * logic.SETTLE_FRAMES)
        self.assertEqual(sorted(p.cls for p in two._settle_up(101)),
                         sorted([STRIPE, SOLID]))

        # Two classes short with one ball gone is a label that changed, not a
        # ball that left - and neither is a settling held for a hand.
        relabel = self.settled(before=6, after=5)
        relabel.before_census = collections.Counter({STRIPE: 3, SOLID: 1,
                                                     EIGHT: 1, CUE: 1})
        relabel.census.extend([(True, collections.Counter({STRIPE: 2, EIGHT: 1,
                                                           CUE: 1}))]
                              * logic.SETTLE_FRAMES)
        self.assertEqual(relabel._settle_up(101), [])

        held = self.settled(before=6, after=5)
        held.last_hold = 9
        held.before_census = collections.Counter({STRIPE: 3, SOLID: 1, EIGHT: 1,
                                                  CUE: 1})
        held.census.extend([(True, collections.Counter({STRIPE: 3, EIGHT: 1,
                                                        CUE: 1}))] * logic.SETTLE_FRAMES)
        self.assertEqual(held._settle_up(101), [])

        # ... but a settling held a frame or two is a player straightening up
        # over the shot they just played, and its census is still read. Both
        # of the pots missed at 7:07 and 7:29 in game.mp4 were held exactly two.
        brief = self.settled(before=6, after=5)
        brief.last_hold = 2
        brief.before_census = collections.Counter({STRIPE: 3, SOLID: 1,
                                                   EIGHT: 1, CUE: 1})
        brief.census.extend([(True, collections.Counter({STRIPE: 3, EIGHT: 1,
                                                         CUE: 1}))]
                            * logic.SETTLE_FRAMES)
        self.assertEqual([p.cls for p in brief._settle_up(101)], [SOLID])

        # No drop in the count, no credit, whatever the labels flicker to.
        same = self.settled(before=6, after=6)
        same.before_census = collections.Counter({STRIPE: 3, SOLID: 1, EIGHT: 1,
                                                  CUE: 1})
        same.census.extend([(True, collections.Counter({STRIPE: 4, EIGHT: 1,
                                                        CUE: 1}))] * logic.SETTLE_FRAMES)
        self.assertEqual(same._settle_up(101), [])

    def test_a_ball_that_went_down_was_hit(self):
        # 07:07 in game.mp4: the solid was credited off the census on a shot
        # where only the cue ball was ever matched between frames, and F-05
        # called "NO BALL HIT" on the player who had just potted their own.
        s = session()
        s.shots.movers = {1: CUE}
        pot = logic.Pot(101, 130, SOLID, "TL", 40.0, 20.0, 0, False, 40, 40,
                        logic.CENSUS)
        self.assertTrue(s._facts([pot], []).hit_object)

        # Nothing down and nothing but the cue ball moving is still F-05.
        self.assertIs(s._facts([], []).hit_object, False)

    def test_the_census_is_read_from_the_clear_frames_there_are(self):
        # A window with a hand across most of it still has the frame where the
        # player stood back, and the count is taken from that same frame.
        mixed = session()
        mixed.before = 6
        mixed.recent.extend([(False, 6)] * (logic.SETTLE_FRAMES - 1)
                            + [(True, 5)])
        mixed.before_census = collections.Counter({STRIPE: 3, SOLID: 1,
                                                   EIGHT: 1, CUE: 1})
        mixed.census.extend(
            [(False, collections.Counter({STRIPE: 3, SOLID: 1, EIGHT: 1,
                                          CUE: 1}))] * (logic.SETTLE_FRAMES - 1)
            + [(True, collections.Counter({STRIPE: 3, EIGHT: 1, CUE: 1}))])
        self.assertEqual([p.cls for p in mixed._settle_up(101)], [SOLID])

    def test_a_cue_ball_the_table_still_shows_is_not_a_scratch(self):
        # 10:10 in game.mp4, the rack-deciding shot. The cue ball ran the
        # length of the table, outjumped the tracker and died beside TL under
        # a player's arm, while the 8 it left behind really did go down. The
        # clear frame shows the cue still on the cloth, so the scratch claim is
        # refused and the 8 is credited instead - a rack won, not lost.
        end = session()
        end.before = 4
        end.recent.extend([(False, 4)] * (logic.SETTLE_FRAMES - 1)
                          + [(True, 2)])
        end.before_census = collections.Counter({CUE: 1, EIGHT: 1, STRIPE: 2})
        end.census.extend(
            [(False, collections.Counter({CUE: 1, EIGHT: 1, STRIPE: 2}))]
            * (logic.SETTLE_FRAMES - 1)
            + [(True, collections.Counter({CUE: 1, STRIPE: 1}))])
        end.candidates = [logic.Pot(3000, 136, CUE, "TL", 106.0, 72.3, 803,
                                    True, 120, 130)]
        credited = end._settle_up(3009)
        self.assertEqual(sorted(p.cls for p in credited),
                         sorted([EIGHT, STRIPE]))
        self.assertNotIn(CUE, [p.cls for p in credited])

    def test_W07_needs_the_eight_to_have_been_there_all_along(self):
        # A rack must not end because one frame once called something the 8.
        s = self.settled(before=6, after=5)
        s.eight_at_rest = True
        s.frame, s.last_seen[EIGHT] = 200, 10
        s.before_census = collections.Counter({STRIPE: 3, SOLID: 2, CUE: 1})
        s.census.extend([(True, collections.Counter({STRIPE: 3, SOLID: 1,
                                                     CUE: 1}))] * logic.SETTLE_FRAMES)
        self.assertEqual(s._off_table([])[0], [])

        # Seen steadily before and gone now, it really did leave the table.
        off = self.settled(before=6, after=5)
        off.eight_at_rest = True
        off.frame, off.last_seen[EIGHT] = 200, 10
        off.before_census = collections.Counter({STRIPE: 3, SOLID: 1, EIGHT: 1,
                                                 CUE: 1})
        off.census.extend([(True, collections.Counter({STRIPE: 3, SOLID: 1,
                                                       CUE: 1}))] * logic.SETTLE_FRAMES)
        self.assertEqual(off._off_table([])[0], [EIGHT])

    def test_a_table_being_cleared_is_not_a_shot(self):
        # The end of every rack in this footage: the players pick the balls up.
        s = session()
        g = s.game
        on_solids(g)
        s.settled_balls = 10
        s.resting = [(100, 100), (200, 200)]
        s.recent.extend([(True, 3)] * logic.SETTLE_FRAMES)
        s.window.extend([(True, [(100, 100)])] * logic.SETTLE_FRAMES)
        s.candidates = [logic.Pot(100, 1, EIGHT, "BR", 72.0, 0.0, 30, False,
                                  980, 480)]
        turn, shots = g.turn, g.shot
        self.assertEqual(s._settle(101), [])
        self.assertEqual((g.over, g.turn, g.shot), (False, turn, shots))
        self.assertEqual((s.resting, s.settled_balls), (None, None))

    def test_cue_ball_potted_under_a_young_tracks_label_is_the_cue(self):
        s = self.settled()
        s.frame, s.cue_at_rest = 200, True
        s.last_seen[CUE] = 150                     # unseen since it ran off
        young = logic.Pot(198, 3, STRIPE, "BR", 90.0, 44.0, 3, True, 950, 480)
        s.candidates = [young]
        self.assertEqual([p.cls for p in s._settle_up(200)], [CUE])

        seen = self.settled()
        seen.frame, seen.cue_at_rest = 200, True
        seen.last_seen[CUE] = 198                  # the cue ball is still there
        seen.candidates = [young]
        self.assertEqual([p.cls for p in seen._settle_up(200)], [STRIPE])

        # And the rules then call it what it was: a scratch, the stripe counts.
        g = on_solids(eight())
        shot(g, pots=[SOLID, CUE])
        self.assertEqual((g.fouls, g.turn, g.potted[SOLID]), ([1, 0], 1, 2))

    FULL = {STRIPE: 3, SOLID: 2, EIGHT: 1, CUE: 1}
    SOLID_DOWN = {STRIPE: 3, SOLID: 1, EIGHT: 1, CUE: 1}

    def census(self, before, now, after):
        """A clear settled table: its count, and the classes either side."""
        s = self.settled(before=sum(before.values()), after=after)
        s.before_census = collections.Counter(before)
        s.census.extend([(True, collections.Counter(now))] * logic.SETTLE_FRAMES)
        return s

    def test_the_settled_census_decides_what_went_down(self):
        # The id switch: a stripe's track died beside TM in a collision while
        # the solid went down elsewhere. Every stripe is still on the settled
        # table, so the stripe claim scores nothing and the solid does.
        s = self.census(self.FULL, self.SOLID_DOWN, after=6)
        switched = logic.Pot(100, 3, STRIPE, "TM", 40.0, 25.0, 50, False, 530, 50)
        s.candidates = [switched]
        self.assertEqual([(p.cls, p.pocket, p.source) for p in s._settle_up(101)],
                         [(SOLID, None, logic.INFERRED)])
        self.assertIn(3, [r["track"] for r in s.pots.rejected])

        # ... and a claim at a mouth for the class that did go says where.
        s = self.census(self.FULL, self.SOLID_DOWN, after=6)
        went = logic.Pot(100, 2, SOLID, "BR", 40.0, 20.0, 50, False, 960, 470)
        s.candidates = [switched, went]
        self.assertEqual([(p.cls, p.pocket, p.track, p.source)
                          for p in s._settle_up(101)],
                         [(SOLID, "BR", 2, logic.CENSUS)])

        # A young track's label means nothing, but where it vanished does.
        young = logic.Pot(100, 8, STRIPE, "BL", 40.0, 30.0, 2, False, 40, 460)
        s = self.census(self.FULL, self.SOLID_DOWN, after=6)
        s.candidates = [young]
        self.assertEqual([(p.cls, p.pocket) for p in s._settle_up(101)],
                         [(SOLID, "BL")])

        # ... except for the 8, whose pocket W-03 can lose a rack on.
        s = self.census(self.FULL, {STRIPE: 3, SOLID: 2, CUE: 1}, after=6)
        s.candidates = [young]
        credited = s._settle_up(101)
        self.assertEqual([(p.cls, p.pocket) for p in credited], [(EIGHT, None)])
        self.assertIsNone(s._facts(credited, []).eight_pocket)

    def test_which_pocket_comes_from_the_last_heading(self):
        self.assertEqual(logic.heading_pocket(640, 330, (600, 300), POCKETS)[0].name,
                         "BR")
        self.assertIsNone(logic.heading_pocket(640, 330, None, POCKETS)[0])

        # No claim at any mouth, but the solid's track was lost heading for BR.
        s = self.census(self.FULL, self.SOLID_DOWN, after=6)
        s.vanished = [logic.Vanish(95, 2, SOLID, 640, 330, (600, 300), 40, False)]
        credited = s._settle_up(101)
        self.assertEqual([(p.cls, p.pocket, p.track) for p in credited],
                         [(SOLID, "BR", 2)])
        self.assertLess(s.pots.confidence(credited[0]), rules.CONTACT_CONFIDENCE)

    def test_a_ball_that_rattled_out_is_on_the_settled_table(self):
        # W-10: the solid vanished into the BR jaw and came back out. The
        # settled table still holds it, so nothing went down.
        s = self.census(self.FULL, self.FULL, after=7)
        s.candidates = [logic.Pot(100, 2, SOLID, "BR", 20.0, 20.0, 50, False,
                                  985, 490)]
        self.assertEqual(s._settle_up(101), [])

        # A pot scored off the census with no pocket is still taken back when
        # its ball turns up beside any pocket after the table settled.
        p = session()
        g = p.game
        on_solids(g)
        frame = judged(p, g, pots=[SOLID])
        pot = logic.Pot(frame, -1, SOLID, None, 0.0, 0.0, 0, False, None, None,
                        logic.INFERRED)
        p.last_credited, p.all_pots = [pot], [pot]
        p.resting, p.settled_balls = [(300, 300), (600, 200)], 14
        p._reconcile_phantoms([(300, 300), (600, 200), (975, 480)],
                              frame + FPS // 2)
        self.assertEqual((g.potted[SOLID], g.turn, p.phantoms), (1, 1, 1))

    def _ball_lighter(self, now_census):
        """A settled table one ball lighter, with only one ball seen to move."""
        s = session()
        on_solids(s.game)
        s.settled_balls = 8
        s.resting = [(100, 100), (200, 200), (300, 300)]
        s.settled_census = collections.Counter({SOLID: 4, STRIPE: 2, EIGHT: 1,
                                                CUE: 1})
        s.recent.extend([(True, 7)] * logic.SETTLE_FRAMES)
        s.window.extend([(True, [(100, 100), (200, 200)])] * logic.SETTLE_FRAMES)
        s.census.extend([(True, collections.Counter(now_census))]
                        * logic.SETTLE_FRAMES)
        return s

    def test_a_table_a_ball_lighter_is_a_shot_even_with_no_claim(self):
        # 07:07 in game.mp4: a solid went down, no track claimed it at a mouth,
        # one ball read as having moved, and the episode was refused as "not a
        # shot" - so the pot went unscored, the turn never passed, and the
        # count stayed wrong for the rest of the rack. The census naming the
        # class that left is the second witness the missing ball needs.
        s = self._ball_lighter({SOLID: 3, STRIPE: 2, EIGHT: 1, CUE: 1})
        credited = s._settle(101)
        self.assertEqual([p.cls for p in credited], [SOLID])
        self.assertEqual(s.game.turn, 0)        # their own ball: still at the table

        # With nothing short in the census, a count that dips on its own is
        # still not a shot: no verdict is invented from one ball moving.
        quiet = self._ball_lighter({SOLID: 4, STRIPE: 2, EIGHT: 1, CUE: 1})
        self.assertEqual(quiet._settle(101), [])
        self.assertIn("not a shot", quiet.episodes[-1]["verdict"])

    def test_the_score_is_read_off_what_the_table_still_holds(self):
        g = on_solids(eight())                  # Player 1 on solids, one down
        self.assertEqual(g.score(0), 1)
        # Four solids on the cloth means three are down, not one: two more
        # than the camera caught, which is what it reports adding.
        self.assertEqual(g.reconcile_score({SOLID: 4, STRIPE: 7}), [(SOLID, 2)])
        self.assertEqual((g.score(0), g.score(1)), (3, 0))
        # Never downward: one reading does not take back a ball that was
        # watched going in - M-04 and W-10 do that, on their own evidence.
        self.assertEqual(g.reconcile_score({SOLID: 7, STRIPE: 7}), [])
        self.assertEqual(g.score(0), 3)

    def test_a_missed_pot_is_corrected_at_the_next_settling(self):
        s = session()
        on_solids(s.game)
        s.recent.extend([(True, 13)] * logic.SETTLE_FRAMES)
        s.census.extend([(True, collections.Counter({SOLID: 4, STRIPE: 7,
                                                     EIGHT: 1, CUE: 1}))]
                        * logic.SETTLE_FRAMES)
        self.assertEqual(s._reconcile_score(), [(SOLID, 2)])
        self.assertEqual(s.game.score(0), 3)

        # Not through a hand, and not on a settling that was held for one.
        busy = session()
        on_solids(busy.game)
        busy.recent.extend([(False, 13)] * logic.SETTLE_FRAMES)
        busy.census.extend([(False, collections.Counter({SOLID: 4, STRIPE: 7,
                                                         EIGHT: 1, CUE: 1}))]
                           * logic.SETTLE_FRAMES)
        self.assertEqual(busy._reconcile_score(), [])
        held = session()
        on_solids(held.game)
        held.last_hold = 9
        held.recent.extend([(True, 13)] * logic.SETTLE_FRAMES)
        held.census.extend([(True, collections.Counter({SOLID: 4, STRIPE: 7,
                                                        EIGHT: 1, CUE: 1}))]
                           * logic.SETTLE_FRAMES)
        self.assertEqual(held._reconcile_score(), [])

        # A label that flickers puts the classes out of step with the count,
        # and then the table is not read at all.
        odd = session()
        on_solids(odd.game)
        odd.recent.extend([(True, 12)] * logic.SETTLE_FRAMES)
        odd.census.extend([(True, collections.Counter({SOLID: 4, STRIPE: 7,
                                                       EIGHT: 1, CUE: 1}))]
                          * logic.SETTLE_FRAMES)
        self.assertEqual(odd._reconcile_score(), [])

    def test_a_player_who_cleared_their_group_wins_on_the_eight(self):
        # The 4-5 bug: the camera saw five of Player 1's solids go down, the
        # table shows all seven are gone, and the 8 they then potted was read
        # as "8 POTTED EARLY" - a rack handed to the opponent on a miscount.
        s = session()
        g = s.game
        on_solids(g)
        g.potted[SOLID] = 5
        s.recent.extend([(True, 9)] * logic.SETTLE_FRAMES)
        s.census.extend([(True, collections.Counter({STRIPE: 7, EIGHT: 1,
                                                     CUE: 1}))]
                        * logic.SETTLE_FRAMES)
        self.assertEqual(s._reconcile_score(), [(SOLID, 2)])
        self.assertTrue(g.on_the_eight(0))

        shot(g, pots=[EIGHT])
        self.assertEqual((g.over, g.winner, g.score(0)), (True, 0, 7))
        self.assertIn("W-01", cited(g))

    def test_an_id_switch_mid_shot_does_not_decide_the_pot(self):
        # The whole pipeline, frame by frame. The cue strikes the solid into
        # BR; in the collision the stripe's id dies beside TM and is reborn,
        # and the solid's id dies mid-flight 400 px short of the pocket, is
        # reborn unlabelled, and is lost again 215 px out - past every mouth.
        rest = {1: (200, 250, CUE), 2: (600, 300, SOLID), 3: (560, 60, STRIPE),
                4: (500, 400, EIGHT)}
        after = {1: (560, 295, CUE), 4: (500, 400, EIGHT)}
        script = [rest] * 6 + [
            {**rest, 1: (300, 270, CUE)},
            {**rest, 1: (400, 280, CUE)},
            {**rest, 1: (560, 295, CUE), 2: (640, 330, SOLID),
             3: (530, 50, STRIPE)},
            {**after, 6: (505, 60, STRIPE), 7: (760, 400, None)},
            {**after, 6: (480, 75, STRIPE), 7: (800, 420, None)},
        ] + [{**after, 6: (460, 85, STRIPE)}] * 9
        s = session()
        credited = []
        for f, balls in enumerate(script, 1):
            tracked = [(tid, x, y, R) for tid, (x, y, _c) in balls.items()]
            labels = {tid: Label(c, "voted")
                      for tid, (_x, _y, c) in balls.items() if c}
            credited += s.update(tracked, labels, f)
        self.assertEqual([(p.cls, p.pocket) for p in credited], [(SOLID, "BR")])
        # The stripe claim at TM scored nothing: no pot was credited for it.
        self.assertEqual([p.cls for p in s.all_pots], [SOLID])


if __name__ == "__main__":
    unittest.main()
