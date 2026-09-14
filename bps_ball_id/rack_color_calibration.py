"""
BPS Pro — Rack-Time Color Calibration  (spec decision 2026-08-28)

WHAT THIS IS
    On every rack detect, BEFORE the break, we grab one calibration frame and
    measure the actual hue of the balls under that venue's actual light. Ball
    numbers are then assigned by RELATIVE RANK inside that frame, never by an
    absolute hue swatch.

WHY
    At Blue Dolphin the blue Olhausen cloth plus dim warm light moves absolute
    hue by 20 degrees or more between a lit and a shadowed position on the SAME
    ball, and 3 (red) sits only ~15-25 degrees from 5 (orange). Fixed hue
    windows collapse. Relative rank does not.

GAME MODES  (Mike, 2026-09-09: "make sure it can do less balls")
    Works off whatever the rack actually contains, not a fixed 15:
        8ball  -> 1..15
        9ball  -> 1..9
        10ball -> 1..10
    The ball list per game lives in rules_packs/ball_palette_v1.json as DATA
    (build directive 2026-08-19), so a new game or a palette change is a pack
    push, not a redeploy.

KEY TECHNIQUES
    1. Hue is sampled from an ANNULUS (ring) on the ball face. The specular
       glare spot sits at the centre and poisons a whole-ball average.
    2. For a stripe ball the ring is taken further in, to stay off the white
       band.
    3. One global hue OFFSET is solved per rack (that is the lighting cast),
       then numbers are assigned by optimal relative matching within the
       solid group and the stripe group separately.
    4. 4 vs 8 and 7 vs 15 are decided on brightness/stripe, not hue.

FALLBACK
    If a usable calibration frame cannot be captured (occluded balls, a player
    in frame, too few balls), the previous rack's calibration is reused and the
    rack is flagged lower_confidence in the event log. Play never blocks.

DEPENDENCIES
    numpy only. No cv2, no scipy -- masks are numpy meshgrid, assignment is a
    bounded brute force (<= 7 items per group). Cheap enough for the Pi 5.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

PACK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rules_packs", "ball_palette_v1.json"
)

# ---------------------------------------------------------------- rules pack


@dataclass(frozen=True)
class BallSpec:
    number: int
    name: str
    hue: float
    kind: str  # solid | stripe | cue | eight
    dark: bool


class BallPalette:
    """The rules pack: which balls exist, their seed hues, and per-game sets."""

    def __init__(self, pack: dict):
        self.version = pack.get("version", "unknown")
        self.league_year = pack.get("league_year", "unknown")
        self._by_number = {
            b["number"]: BallSpec(
                number=b["number"], name=b["name"], hue=float(b["hue"]),
                kind=b["kind"], dark=bool(b["dark"]),
            )
            for b in pack["balls"]
        }
        self._games = pack["games"]

    @classmethod
    def load(cls, path: str = PACK_PATH) -> "BallPalette":
        with open(path) as fh:
            return cls(json.load(fh))

    def games(self) -> list[str]:
        return sorted(self._games)

    def spec(self, number: int) -> BallSpec:
        return self._by_number[number]

    def object_balls(self, game: str) -> list[int]:
        if game not in self._games:
            raise ValueError(f"unknown game {game!r}; pack has {self.games()}")
        return list(self._games[game]["object_balls"])

    def min_for_calibration(self, game: str) -> int:
        return int(self._games[game]["min_for_calibration"])

    def label(self, game: str) -> str:
        return self._games[game].get("label", game)


# ---------------------------------------------------------------- measurement


@dataclass
class BallSample:
    """What we actually measured off one ball in the calibration frame."""
    x: float
    y: float
    r: float
    hue: float            # circular mean hue of the ring, OpenCV scale 0-179
    hue_spread: float     # circular std of ring hue -- high means unreliable
    sat: float            # mean saturation of the ring
    val: float            # mean value (brightness) of the ring
    outer_white: float    # fraction of the OUTER ring that is white -> stripe
    white_pct: float      # fraction of whole ball that is white -> cue
    dark_pct: float       # fraction of whole ball that is dark
    v99: float = 0.0      # this ball's 99th-pct brightness (its own white scale)
    v_p25: float = 0.0    # 25th-pct brightness -- glare-proof "how dark is this ball"
    sat_dark: float = 0.0 # mean saturation of the darkest 40% -- 8 reads neutral

    @property
    def stripe_score(self) -> float:
        """How stripe-like this ball is. Outer-edge white ALONE is not enough:
        at Mike's venue (2026-09-09) the 13 and the 14 sat at 0.18/0.19 against
        a 0.20 cutoff and were read as solids. Adding whole-ball whiteness
        separates every labelled ball in that shoot (stripes >= 0.325,
        solids <= 0.283)."""
        return self.outer_white + self.white_pct


@dataclass
class RackCalibration:
    """Result of one rack-time calibration."""
    game: str
    ok: bool
    confidence: str                       # "high" | "low" | "reused"
    hue_offset: float = 0.0               # global lighting cast, degrees/2
    assignments: dict[int, int] = field(default_factory=dict)  # sample idx -> ball number
    samples: list[BallSample] = field(default_factory=list)
    cue_index: int | None = None
    reasons: list[str] = field(default_factory=list)
    pack_version: str = ""
    frame: int = 0
    timestamp: float = 0.0
    reused_from_frame: int | None = None

    def numbers(self) -> dict[int, int]:
        """ball number -> sample index (inverse of assignments)."""
        return {num: idx for idx, num in self.assignments.items()}

    def as_event_log(self) -> dict:
        """Compact dict for the event log (M-12: event log is source of truth)."""
        return {
            "type": "rack_color_calibration",
            "game": self.game,
            "ok": self.ok,
            "confidence": self.confidence,
            "hue_offset": round(self.hue_offset, 2),
            "balls_identified": len(self.assignments),
            "cue_found": self.cue_index is not None,
            "pack_version": self.pack_version,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "reused_from_frame": self.reused_from_frame,
            "reasons": list(self.reasons),
        }


# ------------------------------------------------------------- hue math utils


def _circ_mean(hues_deg2: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Circular mean on the OpenCV hue scale (0-179 == 0-360 degrees)."""
    ang = hues_deg2.astype(np.float64) * (2.0 * math.pi / 180.0)
    if weights is None:
        c, s = np.cos(ang).mean(), np.sin(ang).mean()
    else:
        wsum = max(float(weights.sum()), 1e-9)
        c = float((np.cos(ang) * weights).sum() / wsum)
        s = float((np.sin(ang) * weights).sum() / wsum)
    m = math.atan2(s, c) * (180.0 / (2.0 * math.pi))
    return m % 180.0


def _circ_std(hues_deg2: np.ndarray) -> float:
    ang = hues_deg2.astype(np.float64) * (2.0 * math.pi / 180.0)
    R = math.hypot(float(np.cos(ang).mean()), float(np.sin(ang).mean()))
    R = min(max(R, 1e-9), 1.0)
    return math.sqrt(-2.0 * math.log(R)) * (180.0 / (2.0 * math.pi))


def _hue_dist(a: float, b: float) -> float:
    """Shortest distance between two hues on the 0-179 wheel."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _ring_mask(shape: tuple[int, int], x: float, y: float, r_in: float, r_out: float):
    h, w = shape
    x0, x1 = max(0, int(x - r_out) - 1), min(w, int(x + r_out) + 2)
    y0, y1 = max(0, int(y - r_out) - 1), min(h, int(y + r_out) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    m = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    return (slice(y0, y1), slice(x0, x1)), m


# ------------------------------------------------------------- the calibrator


class RackColorCalibrator:
    """
    Usage:
        cal = RackColorCalibrator(game="9ball")
        result = cal.calibrate(hsv_frame, ball_positions, frame_num=n)
        # later, for any ball on the table:
        number = cal.identify(hsv_frame, x, y, r)     # -> int or None
    """

    # Ring geometry, as a fraction of the detected ball radius.
    SOLID_RING = (0.45, 0.80)   # off the glare spot, inside the edge shadow
    STRIPE_RING = (0.25, 0.55)  # further in, to stay off the white band
    OUTER_RING = (0.80, 1.00)   # used only to detect the stripe band

    # Quality gates
    MAX_HUE_SPREAD = 42.0       # ring hue std above this = unreliable sample
    MIN_SAT_FOR_HUE = 28.0      # below this the ball has no usable hue
    STRIPE_SCORE = 0.30         # outer_white + white_pct above this = stripe ball.
                                # Derived from 19 hand-labelled balls across two
                                # racks of Mike's 2026-09-09 venue shoot.
    EIGHT_TIE = 0.25            # two dark candidates within this relative gap
                                # are separated on neutrality, not brightness.

    def __init__(self, game: str = "8ball", palette: BallPalette | None = None):
        self.palette = palette or BallPalette.load()
        if game not in self.palette.games():
            raise ValueError(f"unknown game {game!r}")
        self.game = game
        self.current: RackCalibration | None = None
        self.history: list[RackCalibration] = []

    # -- public ---------------------------------------------------------

    def set_game(self, game: str) -> None:
        if game not in self.palette.games():
            raise ValueError(f"unknown game {game!r}")
        if game != self.game:
            self.game = game
            self.current = None

    def calibrate(
        self,
        hsv: np.ndarray,
        ball_positions: list[tuple],
        frame: int = 0,
        fps: float = 30.0,
    ) -> RackCalibration:
        """Measure this rack under this light. Falls back on failure."""
        expected = self.palette.object_balls(self.game)
        need = self.palette.min_for_calibration(self.game)
        stamp = frame / fps if fps else time.time()

        # Pass 1: rough sample, to find the rack's white reference (cue ball).
        rough = [
            s for s in (self._sample(hsv, *p) for p in ball_positions) if s is not None
        ]
        if not rough:
            return self._fallback("no usable balls in calibration frame", frame, stamp)
        white_ref = self._white_reference(rough)

        # Pass 2: re-sample with lighting-matched whiteness gates.
        samples = [
            s
            for s in (self._sample(hsv, *p, white_ref=white_ref) for p in ball_positions)
            if s is not None
        ]
        if len(samples) < need:
            return self._fallback(
                f"only {len(samples)} usable balls, need {need} for "
                f"{self.palette.label(self.game)}",
                frame, stamp,
            )

        cue_idx = self._pick_cue(samples)
        objects = [i for i in range(len(samples)) if i != cue_idx]

        assignments, offset, conf_reasons = self._assign(samples, objects, expected)
        if not assignments:
            return self._fallback("could not resolve ball identities", frame, stamp)

        incomplete = any("group size mismatch" in r for r in conf_reasons)
        if incomplete and self.current is not None and self.current.assignments:
            # An occluded rack is worth less than the last clean one. Reuse and flag.
            return self._fallback(
                "rack incomplete in calibration frame (occlusion?)", frame, stamp
            )

        result = RackCalibration(
            game=self.game,
            ok=True,
            confidence="high" if len(assignments) == len(expected) and not conf_reasons else "low",
            hue_offset=offset,
            assignments=assignments,
            samples=samples,
            cue_index=cue_idx,
            reasons=conf_reasons,
            pack_version=self.palette.version,
            frame=frame,
            timestamp=stamp,
        )
        self.current = result
        self.history.append(result)
        return result

    def identify(self, hsv: np.ndarray, x: float, y: float, r: float) -> int | None:
        """Identify one ball mid-game using the current rack calibration."""
        if self.current is None or not self.current.assignments:
            return None
        s = self._sample(hsv, x, y, r)
        if s is None:
            return None
        if s.white_pct > 0.55 and s.sat < 60:
            return 0  # cue

        ref = self._reference_table()
        want_stripe = s.stripe_score > self.STRIPE_SCORE
        best, best_cost = None, float("inf")
        for number, (hue, val, is_stripe, dark) in ref.items():
            if is_stripe != want_stripe:
                continue
            cost = _hue_dist(s.hue, hue)
            # brightness term: this is what separates 4 from 8 and 7 from 1/3/5.
            # Compared against the SAME ball as measured at rack time, so it is
            # already lighting-matched.
            cost += 0.30 * abs(s.val - val)
            if cost < best_cost:
                best, best_cost = number, cost
        return best

    # -- internals ------------------------------------------------------

    def _reference_table(self) -> dict[int, tuple[float, float, bool, bool]]:
        """number -> (measured hue, measured val, is_stripe, dark) from this rack."""
        out = {}
        cur = self.current
        for idx, number in cur.assignments.items():
            spec = self.palette.spec(number)
            s = cur.samples[idx]
            out[number] = (s.hue, s.val, spec.kind == "stripe", spec.dark)
        return out

    def _fallback(self, why: str, frame: int, stamp: float) -> RackCalibration:
        prev = self.current
        if prev is not None and prev.assignments:
            reused = RackCalibration(
                game=self.game, ok=True, confidence="reused",
                hue_offset=prev.hue_offset, assignments=dict(prev.assignments),
                samples=prev.samples, cue_index=prev.cue_index,
                reasons=[f"reused previous calibration: {why}"],
                pack_version=self.palette.version, frame=frame, timestamp=stamp,
                reused_from_frame=prev.frame,
            )
            self.current = reused
            self.history.append(reused)
            return reused
        failed = RackCalibration(
            game=self.game, ok=False, confidence="low",
            reasons=[f"no calibration available: {why}"],
            pack_version=self.palette.version, frame=frame, timestamp=stamp,
        )
        self.history.append(failed)
        return failed

    def _sample(
        self, hsv: np.ndarray, x: float, y: float, r: float, white_ref: float | None = None
    ) -> BallSample | None:
        h, w = hsv.shape[:2]
        if r < 4:
            return None
        shape = (h, w)

        outer = _ring_mask(shape, x, y, r * self.OUTER_RING[0], r * self.OUTER_RING[1])
        whole = _ring_mask(shape, x, y, 0.0, r * 0.92)
        if outer is None or whole is None:
            return None

        (osl, om), (wsl, wm) = outer, whole
        opix = hsv[osl][om]
        wpix = hsv[wsl][wm]
        if len(opix) < 8 or len(wpix) < 20:
            return None

        # Whiteness is judged RELATIVE to this ball's own brightest pixels, so a
        # dim venue does not turn every stripe into a solid.
        ball_v99 = float(np.percentile(wpix[:, 2], 99))
        # Prefer the rack-level white reference (the cue ball) when we have it --
        # that is the white card a pro would put in frame. Fall back to this
        # ball's own brightest pixels on the first pass.
        ref = white_ref if white_ref else ball_v99
        v_gate = max(38.0, 0.58 * ref)
        outer_white = float(np.mean((opix[:, 2] > v_gate) & (opix[:, 1] < 85)))
        white_pct = float(np.mean((wpix[:, 2] > v_gate) & (wpix[:, 1] < 85)))
        dark_pct = float(np.mean(wpix[:, 2] < max(35.0, 0.30 * ball_v99)))

        stripe_like = (outer_white + white_pct) > self.STRIPE_SCORE
        r_in, r_out = self.STRIPE_RING if stripe_like else self.SOLID_RING
        ring = _ring_mask(shape, x, y, r * r_in, r * r_out)
        if ring is None:
            return None
        rsl, rm = ring
        rpix = hsv[rsl][rm]
        if len(rpix) < 10:
            return None

        # Weight hue by saturation: washed-out and glare pixels get no vote.
        sat = rpix[:, 1].astype(np.float64)
        val = rpix[:, 2].astype(np.float64)
        keep = (sat > self.MIN_SAT_FOR_HUE) & (val > 35) & (val < 245)
        hue_src = rpix[keep] if keep.sum() >= 8 else rpix
        weights = hue_src[:, 1].astype(np.float64) + 1.0

        return BallSample(
            x=float(x), y=float(y), r=float(r),
            hue=_circ_mean(hue_src[:, 0], weights),
            hue_spread=_circ_std(hue_src[:, 0]),
            sat=float(sat.mean()), val=float(val.mean()),
            outer_white=outer_white, white_pct=white_pct, dark_pct=dark_pct,
            v99=ball_v99,
            v_p25=float(np.percentile(wpix[:, 2], 25)),
            sat_dark=float(
                wpix[wpix[:, 2] <= np.percentile(wpix[:, 2], 40)][:, 1].mean()
            ),
        )

    @staticmethod
    def _white_reference(samples: list[BallSample]) -> float:
        """
        The rack's white scale: the brightest low-saturation ball surface, i.e.
        the cue ball. Everything bright/dark is judged against this, so a dim
        venue shifts the whole scale instead of breaking the tests.
        """
        cands = [s.val for s in samples if s.sat < 90]
        if cands:
            return float(max(cands))
        return float(max((s.val for s in samples), default=255.0))

    def _pick_cue(self, samples: list[BallSample]) -> int | None:
        best, best_score = None, 0.0
        for i, s in enumerate(samples):
            if s.white_pct > 0.55 and s.sat < 70:
                score = s.white_pct - s.sat / 255.0
                if score > best_score:
                    best, best_score = i, score
        return best

    def _pick_eight(self, samples: list[BallSample], rest_idx: list[int]) -> int:
        """
        Which non-stripe ball is the 8?

        Not "lowest mean brightness": on Mike's 2026-09-09 rack the 8 carried a
        specular highlight and read BRIGHTER on the mean (132) than the 6 (60),
        so the old rule picked the 6 -- the single worst mistake the system can
        make. Two changes: judge darkness on the 25th percentile (glare lives in
        the top quartile), and when the two darkest balls are close, take the
        more NEUTRAL one, because the 6 is a saturated green and the 8 is not.
        """
        ranked = sorted(rest_idx, key=lambda i: samples[i].v_p25)
        best = ranked[0]
        if len(ranked) > 1:
            a, b = samples[ranked[0]], samples[ranked[1]]
            spread = max(a.v_p25, b.v_p25, 1.0)
            if abs(a.v_p25 - b.v_p25) / spread < self.EIGHT_TIE:
                best = ranked[0] if a.sat_dark <= b.sat_dark else ranked[1]
        return best

    def _assign(
        self, samples: list[BallSample], objects: list[int], expected: list[int]
    ) -> tuple[dict[int, int], float, list[str]]:
        """
        Split into stripe / solid groups, solve ONE global hue offset, then
        match by relative rank inside each group.
        """
        reasons: list[str] = []
        exp_solid = [n for n in expected if self.palette.spec(n).kind == "solid"]
        exp_eight = [n for n in expected if self.palette.spec(n).kind == "eight"]
        exp_stripe = [n for n in expected if self.palette.spec(n).kind == "stripe"]

        stripe_idx = [i for i in objects if samples[i].stripe_score > self.STRIPE_SCORE]
        rest_idx = [i for i in objects if i not in stripe_idx]

        # 8-ball first: darkest of the non-stripe balls, and it must be dark.
        assignments: dict[int, int] = {}
        if exp_eight and rest_idx:
            cand = self._pick_eight(samples, rest_idx)
            if samples[cand].dark_pct > 0.30 or samples[cand].v_p25 < 80:
                assignments[cand] = exp_eight[0]
                rest_idx = [i for i in rest_idx if i != cand]
            else:
                reasons.append("8-ball not confidently dark")

        offset = self._solve_offset(samples, rest_idx, exp_solid, stripe_idx, exp_stripe)

        for idx_group, exp_group in ((rest_idx, exp_solid), (stripe_idx, exp_stripe)):
            if not idx_group or not exp_group:
                continue
            pairs, group_reasons = self._match_group(samples, idx_group, exp_group, offset)
            assignments.update(pairs)
            reasons.extend(group_reasons)

        if len(assignments) < max(3, len(expected) // 2):
            return {}, offset, reasons
        return assignments, offset, reasons

    def _solve_offset(self, samples, solid_idx, exp_solid, stripe_idx, exp_stripe) -> float:
        """
        One number describes the venue's colour cast: the rotation that best
        lines the measured hues up with the palette's relative layout.
        """
        meas, seed = [], []
        for idx_group, exp_group in ((solid_idx, exp_solid), (stripe_idx, exp_stripe)):
            usable = [i for i in idx_group if samples[i].sat > 45 and not self.palette_dark_only(exp_group)]
            for i in usable:
                meas.append(samples[i].hue)
            for n in exp_group:
                if not self.palette.spec(n).dark:
                    seed.append(self.palette.spec(n).hue)
        if not meas or not seed:
            return 0.0
        meas_a = np.array(meas, dtype=np.float64)
        seed_a = np.array(seed, dtype=np.float64)
        best_off, best_cost = 0.0, float("inf")
        for off in np.arange(-30.0, 30.5, 0.5):
            shifted = (seed_a + off) % 180.0
            # each measured ball must sit near SOME palette hue
            cost = 0.0
            for m in meas_a:
                cost += min(_hue_dist(m, s) for s in shifted) ** 2
            if cost < best_cost:
                best_off, best_cost = float(off), cost
        return best_off

    def palette_dark_only(self, group: list[int]) -> bool:
        return bool(group) and all(self.palette.spec(n).dark for n in group)

    def _match_group(
        self, samples, idx_group: list[int], exp_group: list[int], offset: float
    ) -> tuple[dict[int, int], list[str]]:
        """
        Optimal one-to-one match inside a group. Cost is circular hue distance
        to the offset-corrected palette layout, plus a brightness term for the
        dark balls (4 purple, 7 maroon). Because the match is one-to-one, 3 and
        5 are separated by which of the two is MORE ORANGE -- relative rank,
        never an absolute swatch.
        """
        reasons: list[str] = []
        idxs = list(idx_group)
        exps = list(exp_group)
        n = min(len(idxs), len(exps))
        if n == 0:
            return {}, reasons

        # Brightness reference for THIS rack: the brightest ball surface present
        # (normally the cue ball). Makes the dark-ball test scale-free.
        v_ref = self._white_reference(samples) or 255.0

        cost = np.zeros((len(idxs), len(exps)), dtype=np.float64)
        for a, i in enumerate(idxs):
            s = samples[i]
            for b, num in enumerate(exps):
                spec = self.palette.spec(num)
                target = (spec.hue + offset) % 180.0
                c = _hue_dist(s.hue, target)
                vn = s.val / v_ref            # 0..1, lighting-independent
                if spec.dark:
                    # dark balls (4 purple, 7 maroon) read low Value, not a hue
                    c += max(0.0, (vn - 0.42)) * 55.0
                else:
                    c += max(0.0, (0.38 - vn)) * 45.0
                if s.hue_spread > self.MAX_HUE_SPREAD:
                    c += 8.0
                cost[a, b] = c

        rows, cols = self._hungarian_or_brute(cost)
        pairs = {}
        for a, b in zip(rows, cols):
            pairs[idxs[a]] = exps[b]
            if cost[a, b] > 22.0:
                reasons.append(
                    f"weak match: ball {exps[b]} cost {cost[a, b]:.1f}"
                )
        if len(idxs) != len(exps):
            reasons.append(
                f"group size mismatch: {len(idxs)} seen vs {len(exps)} expected"
            )
        return pairs, reasons

    @staticmethod
    def _hungarian_or_brute(cost: np.ndarray):
        """Optimal assignment. Groups are <= 7 balls, so brute force is fine."""
        nr, nc = cost.shape
        k = min(nr, nc)
        if nr == nc and k <= 7:
            rows = list(range(nr))
            best, best_total = None, float("inf")
            for col_sel in permutations(range(nc), k):
                total = sum(cost[r, c] for r, c in zip(rows, col_sel))
                if total < best_total:
                    best, best_total = (rows, list(col_sel)), total
            return best
        # Fallback for oversized groups: greedy on the cheapest cell.
        used_r, used_c, rows_out, cols_out = set(), set(), [], []
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            if r in used_r or c in used_c:
                continue
            used_r.add(int(r)); used_c.add(int(c))
            rows_out.append(int(r)); cols_out.append(int(c))
            if len(rows_out) == k:
                break
        return rows_out, cols_out


# --------------------------------------------------- break-window calibration


@dataclass
class TrackedBall:
    """One detected ball in one frame of the break window, with its track id."""
    track_id: int
    x: float
    y: float
    r: float


class BreakWindowCalibrator:
    """
    Calibrate over the BREAK WINDOW instead of the racked frame.

    Why (measured on Mike's 2026-09-09 rack shot, 8/13 ids):
      1. The racked frame usually has no cue ball in it, so the white reference
         gets taken off a stripe's band and every whiteness reading drifts.
      2. A stripe whose band faces away shows no stripe at all from overhead --
         the 9, 13 and 15 all read as solids in that frame. No threshold fixes
         geometry; another frame does.
      3. Touching balls bleed into each other's sample ring.

    All three go away a second later: the cue ball is on the table, the balls
    are apart, and they are ROLLING, so every ball turns its number and its band
    towards the camera at some point. So we watch the first seconds after the
    break, keep the best view of each tracked ball, and calibrate once on the
    merged picture.

    Usage:
        win = BreakWindowCalibrator(RackColorCalibrator(game="8ball"))
        for n, (hsv, tracks) in enumerate(frames):     # tracks: list[TrackedBall]
            win.observe(hsv, tracks, frame=n)
        result = win.finalize(frame=n)                 # -> RackCalibration
    """

    MIN_FRAMES = 3          # merged calibration needs at least this many views
    MIN_TRACK_VIEWS = 2     # a ball seen once is not trusted into the merge
    MIN_SEPARATION = 1.75   # centre distance / own radius; a ball closer than
                            # this to a neighbour bleeds into its sample ring and
                            # is dropped FROM THAT FRAME (not the whole frame --
                            # a touching pair after the break is normal)
    MIN_CLEAN_FRACTION = 0.5  # if more than half the balls are bleeding, the
                            # table is still packed: skip the frame
    MAX_FRAMES = 120        # ~4s at 30fps; the window closes after this

    def __init__(self, calibrator: RackColorCalibrator):
        self.cal = calibrator
        self._views: dict[int, list[BallSample]] = {}
        self._anchors: list[float] = []
        self.frames_seen = 0
        self.frames_used = 0
        self.skipped: list[str] = []

    # -- public ---------------------------------------------------------

    def observe(self, hsv: np.ndarray, tracks: list[TrackedBall], frame: int = 0) -> bool:
        """Take one frame of the window. Returns True if the frame was used."""
        self.frames_seen += 1
        if self.frames_used >= self.MAX_FRAMES:
            return False
        if len(tracks) < 2:
            self.skipped.append(f"frame {frame}: too few balls")
            return False
        clean = self._clean_tracks(tracks)
        if len(clean) < max(2, self.MIN_CLEAN_FRACTION * len(tracks)):
            self.skipped.append(f"frame {frame}: balls still packed")
            return False
        tracks = clean

        rough = [s for s in (self.cal._sample(hsv, t.x, t.y, t.r) for t in tracks) if s]
        if not rough:
            self.skipped.append(f"frame {frame}: nothing samplable")
            return False
        white_ref = self.cal._white_reference(rough)

        frame_samples: dict[int, BallSample] = {}
        for t in tracks:
            s = self.cal._sample(hsv, t.x, t.y, t.r, white_ref=white_ref)
            if s is not None:
                frame_samples[t.track_id] = s

        if self.cal._pick_cue(list(frame_samples.values())) is not None:
            # Cue ball in shot: this frame's white reference is trustworthy, so
            # it anchors the whole window.
            self._anchors.append(white_ref)
        elif self._anchors:
            # No cue ball this frame -- re-sample against the window's anchored
            # white instead of this frame's brightest ball.
            anchor = float(np.median(self._anchors))
            frame_samples = {
                t.track_id: s
                for t in tracks
                for s in [self.cal._sample(hsv, t.x, t.y, t.r, white_ref=anchor)]
                if s is not None
            }
        else:
            # No cue ball and nothing to anchor on yet: this is exactly the
            # racked-frame failure mode. Do not merge it in.
            self.skipped.append(f"frame {frame}: no cue ball, no white anchor yet")
            return False

        for tid, s in frame_samples.items():
            self._views.setdefault(tid, []).append(s)
        self.frames_used += 1
        return True

    def finalize(self, frame: int = 0, fps: float = 30.0) -> RackCalibration:
        """Merge the window into one view per ball and calibrate on that."""
        stamp = frame / fps if fps else time.time()
        if self.frames_used < self.MIN_FRAMES:
            return self.cal._fallback(
                f"break window unusable: only {self.frames_used} clean frames",
                frame, stamp,
            )

        tids = [t for t, v in self._views.items() if len(v) >= self.MIN_TRACK_VIEWS]
        if not tids:
            return self.cal._fallback("no ball tracked across the window", frame, stamp)

        tids.sort()
        self.track_order = tids
        merged = [self._merge(self._views[t]) for t in tids]

        need = self.cal.palette.min_for_calibration(self.cal.game)
        if len(merged) < need:
            return self.cal._fallback(
                f"only {len(merged)} tracked balls, need {need} for "
                f"{self.cal.palette.label(self.cal.game)}",
                frame, stamp,
            )

        expected = self.cal.palette.object_balls(self.cal.game)
        cue_idx = self.cal._pick_cue(merged)
        objects = [i for i in range(len(merged)) if i != cue_idx]
        assignments, offset, reasons = self.cal._assign(merged, objects, expected)
        if not assignments:
            return self.cal._fallback("could not resolve ball identities", frame, stamp)

        reasons = list(reasons)
        reasons.append(f"break window: {self.frames_used}/{self.frames_seen} frames merged")
        result = RackCalibration(
            game=self.cal.game,
            ok=True,
            confidence="high" if len(assignments) == len(expected)
            and not any("mismatch" in r or "weak" in r for r in reasons) else "low",
            hue_offset=offset,
            assignments=assignments,
            samples=merged,
            cue_index=cue_idx,
            reasons=reasons,
            pack_version=self.cal.palette.version,
            frame=frame,
            timestamp=stamp,
        )
        self.cal.current = result
        self.cal.history.append(result)
        return result

    # -- internals ------------------------------------------------------

    def _clean_tracks(self, tracks: list[TrackedBall]) -> list[TrackedBall]:
        """Drop balls whose neighbour is close enough to bleed into their ring."""
        pts = np.array([[t.x, t.y] for t in tracks], dtype=np.float64)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        return [
            t for i, t in enumerate(tracks)
            if nearest[i] >= self.MIN_SEPARATION * max(t.r, 1.0)
        ]

    @staticmethod
    def _merge(views: list[BallSample]) -> BallSample:
        """
        One ball, several views. Hue and brightness are averaged (a single frame
        can catch a highlight), but WHITENESS takes the best view: a band seen
        once is a band, and a ball that never shows white in any frame is a
        genuine solid.
        """
        med = lambda f: float(np.median([f(v) for v in views]))
        hues = np.array([v.hue for v in views], dtype=np.float64)
        weights = np.array([v.sat + 1.0 for v in views], dtype=np.float64)
        last = views[-1]
        best_white = max(views, key=lambda v: v.stripe_score)
        return BallSample(
            x=last.x, y=last.y, r=last.r,
            hue=_circ_mean(hues, weights),
            hue_spread=med(lambda v: v.hue_spread),
            sat=med(lambda v: v.sat),
            val=med(lambda v: v.val),
            outer_white=best_white.outer_white,
            white_pct=best_white.white_pct,
            dark_pct=med(lambda v: v.dark_pct),
            v99=med(lambda v: v.v99),
            v_p25=med(lambda v: v.v_p25),
            sat_dark=med(lambda v: v.sat_dark),
        )
