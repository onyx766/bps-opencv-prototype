"""
BPS Pro — rack calibration hook

Thin adapter that drops rack-time colour calibration into an existing frame
loop (live_demo.py / step8_integrated.py) without touching their internals.

    from rack_calibration_hook import RackCalibrationHook

    hook = RackCalibrationHook(game="9ball")      # "8ball" | "9ball" | "10ball"

    # in the frame loop, AFTER ball detection, BEFORE the break is scored:
    event = rerack.update(positions_xy, frame_num, fps)
    log_entry = hook.on_frame(frame_bgr, positions_xyr, frame_num, fps,
                              rack_event=event)
    if log_entry:
        event_log.append(log_entry)        # M-12: event log is source of truth

    # anywhere a ball number is needed:
    number = hook.identify(frame_bgr, x, y, r)    # 0 = cue, None = unknown

Notes
    - Calibration runs ONCE per rack, on the first frame the rack is stable.
      Cost measured on this workspace: 16 ms for 9/10-ball, 66 ms for 8-ball,
      per rack. identify() is ~0.3 ms per ball.
    - The frame is converted BGR->HSV once and cached per frame number.
    - BPS TRACKS, never enforces: a low-confidence rack is flagged in the log,
      never blocked.
"""

from __future__ import annotations

import numpy as np

from rack_color_calibration import RackColorCalibrator


class RackCalibrationHook:
    def __init__(
        self,
        game: str = "8ball",
        min_gap_frames: int = 15,
        frames_are_hsv: bool = False,
    ):
        # frames_are_hsv=True when the caller's pipeline already holds HSV
        # (skips a redundant colour conversion, and keeps this testable
        # without cv2 installed).
        self.frames_are_hsv = frames_are_hsv
        self.cal = RackColorCalibrator(game=game)
        self.min_gap_frames = min_gap_frames
        self._last_cal_frame = -10_000
        self._hsv_cache: tuple[int, np.ndarray] | None = None

    # -- game mode -------------------------------------------------------

    def set_game(self, game: str) -> None:
        """Switch game type (remote config / venue selection)."""
        self.cal.set_game(game)
        self._last_cal_frame = -10_000

    @property
    def game(self) -> str:
        return self.cal.game

    # -- frame loop ------------------------------------------------------

    def on_frame(
        self,
        frame_bgr: np.ndarray,
        ball_positions: list[tuple],
        frame_num: int,
        fps: float = 30.0,
        rack_event=None,
        force: bool = False,
    ) -> dict | None:
        """
        Returns an event-log dict when a calibration was attempted this frame,
        else None.
        """
        want = force or rack_event is not None or not self.calibrated
        if not want:
            return None
        if frame_num - self._last_cal_frame < self.min_gap_frames and not force:
            return None

        hsv = self._hsv(frame_bgr, frame_num)
        result = self.cal.calibrate(hsv, ball_positions, frame=frame_num, fps=fps)
        self._last_cal_frame = frame_num
        return result.as_event_log()

    def identify(self, frame_bgr: np.ndarray, x: float, y: float, r: float,
                 frame_num: int | None = None) -> int | None:
        hsv = self._hsv(frame_bgr, frame_num if frame_num is not None else -1)
        return self.cal.identify(hsv, x, y, r)

    # -- state -----------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        cur = self.cal.current
        return bool(cur and cur.assignments)

    @property
    def confidence(self) -> str:
        cur = self.cal.current
        return cur.confidence if cur else "none"

    def ball_map(self) -> dict[int, tuple[float, float]]:
        """ball number -> (x, y) as seen in the calibration frame."""
        cur = self.cal.current
        if not cur:
            return {}
        return {
            num: (cur.samples[idx].x, cur.samples[idx].y)
            for idx, num in cur.assignments.items()
        }

    # -- internals -------------------------------------------------------

    def _hsv(self, frame_bgr: np.ndarray, frame_num: int) -> np.ndarray:
        if self._hsv_cache and self._hsv_cache[0] == frame_num and frame_num >= 0:
            return self._hsv_cache[1]
        if (not self.frames_are_hsv) and frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
            import cv2  # imported lazily so the module stays testable without cv2

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        else:
            hsv = frame_bgr
        self._hsv_cache = (frame_num, hsv)
        return hsv
