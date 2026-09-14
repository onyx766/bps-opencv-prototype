"""Tell a ball apart as CUE, EIGHT, STRIPE or SOLID by where its white is.

Seen from overhead, the four classes differ in how much of the ball is white:

    cue     all of it
    stripe  a broad band, because the colour wraps the equator and the poles
            are white - whichever way it lies, a good deal of white shows
    solid   only the small printed number patch, and the glare spot
    eight   almost none of it is bright at all

The catch is that "how much white" alone does NOT separate stripe from solid.
Every ball carries its number in a white circle, and measured over the middle
of the ball a solid yellow reads 0.17 white while a green stripe reads 0.21 -
they overlap, and no threshold can be put between them.

WHERE the white sits does separate them. A solid's white is the number circle,
which sits in the middle of the face; a stripe's white is the poles, which
reach the rim on two opposite sides. So the measurement is taken over an
ANNULUS - the outer part of the ball, with the middle left out. On that ring
the same two balls read 0.10 and 0.48, and across a whole frame solids top out
at 0.10 while stripes start at 0.31: a gap wide enough to put a cut in.

The ring earns its keep twice over. The 8-ball's number circle drops out too -
0.26 white measured over the middle, 0.01 over the ring - and so does the one
thing that genuinely resembles the cue ball, a stripe lying white-pole-up,
which falls from 0.87 to 0.48.

Two smaller details:

  * The ring stops short of the true edge. A sphere lit from above falls off to
    a dark rim, and sampling into it makes every ball look like the 8.

  * The white and dark gates are fractions of a measured reference, not fixed
    values. The reference is the brightest thing on the table, which is the cue
    ball or a stripe's white. Tying the gates to it is what lets the same
    thresholds survive a different venue, a dimmer room or a camera that has
    re-exposed - none of which absolute numbers would survive.
"""

import numpy as np

#: The annulus the white fraction is measured over, as fractions of the ball
#: radius: outside the number circle, inside the dark rim.
RING_IN = 0.55
RING_OUT = 0.95

#: Darkness is measured over the whole inner disc instead, because the 8-ball
#: is dark all over and the ring alone would be thrown by a glare spot on it.
CORE = 0.72

#: White: unsaturated, and bright relative to the brightest ball on the table.
WHITE_SAT = 70
WHITE_V = 0.70

#: Dark: the 8-ball, against the same reference.
DARK_V = 0.45

#: Class cuts on the measured fractions.
EIGHT_DARK = 0.40      # the 8 measured 0.56; no other ball came near
CUE_WHITE = 0.85       # the cue measured 0.96, the whitest stripe 0.54
CUE_SAT = 0.05         # ... and the cue has no colour on it at all
STRIPE_WHITE = 0.20    # solids reach 0.10, stripes start at 0.31

CLASSES = ("cue", "eight", "stripe", "solid")


def sample(hsv, x, y, r):
    """(ring, core) HSV pixels for one ball, or None if it runs off the frame.

    A ball that close to the edge is half cushion anyway, and a clipped sample
    would read whatever the frame border happens to hold.
    """
    k = max(1, int(round(r * RING_OUT)))
    y0, y1, x0, x1 = y - k, y + k + 1, x - k, x + k + 1
    if y0 < 0 or x0 < 0 or y1 > hsv.shape[0] or x1 > hsv.shape[1]:
        return None
    ys, xs = np.ogrid[-k:k + 1, -k:k + 1]
    d2 = xs * xs + ys * ys
    patch = hsv[y0:y1, x0:x1]
    ring = patch[(d2 <= (r * RING_OUT) ** 2) & (d2 >= (r * RING_IN) ** 2)]
    core = patch[d2 <= (r * CORE) ** 2]
    return (ring, core) if len(ring) and len(core) else None


def white_reference(hsv, balls):
    """How bright "white" is in this frame, from the balls themselves.

    The 99th percentile over every ball's pixels, so it lands on the whitest
    thing on the table - the cue ball, or a stripe's band when the cue is off
    the table. Taking it from the balls rather than the whole frame keeps the
    rails, the players and the room lights out of it.
    """
    pix = [s[1][:, 2] for s in (sample(hsv, x, y, r) for x, y, r in balls)
           if s is not None]
    return float(np.percentile(np.concatenate(pix), 99)) if pix else 255.0


def measure(ring, core, white_ref):
    """The three fractions the class is decided from."""
    return {
        "white": float(np.mean((ring[:, 1] < WHITE_SAT)
                               & (ring[:, 2] > WHITE_V * white_ref))),
        "dark": float(np.mean(core[:, 2] < DARK_V * white_ref)),
        "sat": float(np.mean(ring[:, 1] > 120)),
    }


def classify(stats):
    """Name the class from the measured fractions.

    Darkness settles the 8-ball before white is consulted, and the cue is taken
    before the stripe cut, because both would otherwise fall through to a cut
    they trivially pass.
    """
    if stats["dark"] > EIGHT_DARK:
        return "eight"
    if stats["white"] > CUE_WHITE and stats["sat"] < CUE_SAT:
        return "cue"
    if stats["white"] > STRIPE_WHITE:
        return "stripe"
    return "solid"


def classify_ball(hsv, x, y, r, white_ref):
    """(class, stats) for one ball, or (None, None) if it cannot be sampled."""
    pixels = sample(hsv, x, y, r)
    if pixels is None:
        return None, None
    stats = measure(pixels[0], pixels[1], white_ref)
    return classify(stats), stats
