"""Score a BPS run against a hand-written answer key.

The MVP's definition of done, on a game the system has not been tuned on:
  1. the break is detected in every game,
  2. at least 90% of pocketed balls are right - which ball AND which player,
  3. every game's winner is called correctly.
(Running live on the Pi is the fourth; bench.py answers that one.)

"Which ball" is judged by category - solid, stripe or the 8 - because that is
what the vision layer identifies; numbers written in the key are mapped to
their category. Scratches are counted separately and do not affect item 2.

Pots are matched IN ORDER within each game (longest common subsequence), so
the key does not need exact timestamps and one missed ball does not shift
every later one out of alignment. Balls a player pots in one turn are compared
as a group, since the order two balls drop on the same shot is arbitrary.

ANSWER KEY - a CSV, one row per thing that happened, in the order it happened:

    game,event,player,ball,time,note
    1,break,1,,00:04,
    1,pot,1,3,00:05,on the break
    1,pot,2,11,00:41,
    1,pot,2,cue,01:02,scratch
    1,win,2,,06:10,
    2,break,2,,07:30,

  game    1, 2, 3 ... in the order played in the video
  event   break | pot | win
  player  1 or 2 (Player 1 / Player 2 as main.py names them), or their names
  ball    1-15, cue, 8, solid or stripe (pot rows only)
  time    optional mm:ss into the video - shown next to misses, not scored
  note    optional, ignored

Usage:
    python evaluate.py answer_key.csv result_record.json
    python evaluate.py key.csv result_record_game1.json result_record_game2.json
    python evaluate.py key.csv result_record.json --json eval.json
"""

import argparse
import csv
import json
import sys

POT_TARGET = 0.90
OBJECT = ("solid", "stripe", "eight")


def ball_category(ball):
    b = str(ball).strip().lower()
    if b in ("cue", "c", "0", "white"):
        return "cue"
    if b in ("8", "eight", "black"):
        return "eight"
    if b in ("solid", "solids"):
        return "solid"
    if b in ("stripe", "stripes"):
        return "stripe"
    if b.isdigit() and 1 <= int(b) <= 15:
        return "solid" if int(b) < 8 else "stripe"
    raise ValueError(f"unknown ball {ball!r}")


def parse_time(text):
    text = (text or "").strip()
    if not text:
        return None
    secs = 0.0
    for part in text.split(":"):
        secs = secs * 60 + float(part)
    return secs


def clock(secs):
    if secs is None:
        return "--:--"
    secs = int(secs)
    return f"{secs // 60:02d}:{secs % 60:02d}"


def load_key(path, names=()):
    """{game number: {"breaker", "winner", "pots": [(player, cat, secs)], "scratches"}}"""
    lookup = {n.strip().lower(): i for i, n in enumerate(names, 1)}
    games = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for n, row in enumerate(csv.DictReader(fh), 2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if not any(row.values()):
                continue
            try:
                g = int(row["game"])
                event = row["event"].lower()
                who = row.get("player", "")
                player = (int(who) if who.isdigit()
                          else lookup.get(who.lower()) if who else None)
                if who and player not in (1, 2):
                    raise ValueError(f"unknown player {who!r}")
                game = games.setdefault(g, {"breaker": None, "winner": None,
                                            "pots": [], "scratches": 0})
                if event == "break":
                    game["breaker"] = player
                elif event == "win":
                    game["winner"] = player
                elif event == "pot":
                    cat = ball_category(row.get("ball", ""))
                    if cat == "cue":
                        game["scratches"] += 1
                    else:
                        if player is None:
                            raise ValueError("pot row needs a player")
                        game["pots"].append((player, cat, parse_time(row.get("time"))))
                else:
                    raise ValueError(f"unknown event {event!r}")
            except (KeyError, ValueError) as err:
                sys.exit(f"{path} line {n}: {err}")
    return [games[g] for g in sorted(games)]


def load_run(paths):
    """The games the system saw, from one or more match records, in order."""
    games, names = [], None
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        summary = rec.get("summary", {})
        names = names or summary.get("players")
        if "shots_log" not in summary:
            sys.exit(f"{path} has no shot log - it was written before evaluate.py "
                     f"existed. Re-run main.py on the video.")
        reverted = {e["seq"] for e in rec["events"] if e.get("reverted")}
        racks = {}
        for shot in summary["shots_log"]:
            # A shot whose every verdict was undone by a player never counted.
            if shot["events"] and all(s in reverted for s in shot["events"]):
                continue
            racks.setdefault(shot["rack"], {"shots": [], "winner": None})["shots"].append(shot)
        for e in rec["events"]:
            if e["type"] == "win" and e["seq"] not in reverted:
                racks.setdefault(e["rack"], {"shots": [], "winner": None})["winner"] = e["player"] + 1
        for r in sorted(racks):
            shots = racks[r]["shots"]
            breaks = [s for s in shots if s["break"]]
            games.append({
                "rack": r, "source": path,
                "breaker": breaks[0]["shooter"] + 1 if breaks else None,
                "break_at": breaks[0]["seconds"] if breaks else None,
                "winner": racks[r]["winner"],
                "pots": [(s["shooter"] + 1, c, s["seconds"])
                         for s in shots for c in s["pots"] if c in OBJECT or c == "unknown"],
                "scratches": sum(s["pots"].count("cue") + s["off_table"].count("cue")
                                 for s in shots),
            })
    return games, names or ["Player 1", "Player 2"]


def by_turn(pots):
    """Sort within each run of one player's pots: drop order on a shot is noise."""
    out, run = [], []
    for p in pots + [None]:
        if run and (p is None or p[0] != run[-1][0]):
            out.extend(sorted(run, key=lambda q: q[1]))
            run = []
        if p is not None:
            run.append(p)
    return out


def lcs(a, b, same):
    """Matched index pairs of the longest common subsequence of a and b."""
    n, m = len(a), len(b)
    t = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            t[i][j] = (t[i + 1][j + 1] + 1 if same(a[i], b[j])
                       else max(t[i + 1][j], t[i][j + 1]))
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if same(a[i], b[j]):
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif t[i + 1][j] >= t[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def evaluate(key, run):
    report = {"games": [], "key_games": len(key), "run_games": len(run)}
    total = correct = ball_only = extra = 0
    breaks_ok = winners_ok = True
    for g in range(max(len(key), len(run))):
        k = key[g] if g < len(key) else None
        s = run[g] if g < len(run) else None
        row = {"game": g + 1}
        if k is None:
            row["problem"] = "system saw a game that is not in the key"
            row["system_pots"] = len(s["pots"])
            extra += len(s["pots"])
            report["games"].append(row)
            continue
        kp = by_turn(k["pots"])
        sp = by_turn(s["pots"]) if s else []
        both = lcs(kp, sp, lambda x, y: x[:2] == y[:2])
        loose = lcs([p[1] for p in kp], [p[1] for p in sp], lambda x, y: x == y)
        hit_k = {i for i, _ in both}
        hit_s = {j for _, j in both}
        row.update({
            "key_pots": len(kp), "system_pots": len(sp),
            "correct": len(both), "ball_right": len(loose),
            "missed": [{"player": p[0], "ball": p[1], "time": clock(p[2])}
                       for i, p in enumerate(kp) if i not in hit_k],
            "wrong_or_extra": [{"player": p[0], "ball": p[1], "time": clock(p[2])}
                               for j, p in enumerate(sp) if j not in hit_s],
            "break_detected": bool(s and s["breaker"] is not None),
            "breaker": {"key": k["breaker"], "system": s and s["breaker"]},
            "winner": {"key": k["winner"], "system": s and s["winner"]},
            "scratches": {"key": k["scratches"], "system": s and s["scratches"]},
        })
        if s is None:
            row["problem"] = "the system never saw this game"
        total += len(kp)
        correct += len(both)
        ball_only += len(loose)
        extra += len(sp) - len(both)
        breaks_ok &= row["break_detected"] and (k["breaker"] is None
                                                or k["breaker"] == row["breaker"]["system"])
        winners_ok &= k["winner"] is not None and k["winner"] == row["winner"]["system"]
        report["games"].append(row)

    acc = correct / total if total else 0.0
    report.update({
        "pots_in_key": total, "pots_correct": correct,
        "pot_accuracy": round(acc, 4),
        "ball_accuracy_ignoring_player": round(ball_only / total, 4) if total else 0.0,
        "extra_or_wrong_pots": extra,
        "criteria": {
            "break detected (every game)": breaks_ok and len(run) >= len(key),
            f"pots >= {POT_TARGET:.0%} (ball + player)": acc >= POT_TARGET,
            "winner correct (every game)": winners_ok and len(run) >= len(key),
        },
    })
    report["pass"] = all(report["criteria"].values())
    return report


def show(report, names):
    who = lambda p: "-" if p is None else names[p - 1] if 1 <= p <= 2 else str(p)
    print(f"Games: key {report['key_games']}, system {report['run_games']}")
    for row in report["games"]:
        print(f"\nGame {row['game']}")
        if "problem" in row:
            print(f"  !! {row['problem']}")
        if "key_pots" not in row:
            continue
        print(f"  break    {'detected' if row['break_detected'] else 'NOT DETECTED'}"
              f"  breaker key {who(row['breaker']['key'])} / system "
              f"{who(row['breaker']['system'])}")
        print(f"  winner   key {who(row['winner']['key'])} / system "
              f"{who(row['winner']['system'])}")
        print(f"  pots     {row['correct']}/{row['key_pots']} right "
              f"(ball right, any player: {row['ball_right']}), system called "
              f"{row['system_pots']}")
        print(f"  scratch  key {row['scratches']['key']} / system "
              f"{row['scratches']['system']}")
        for m in row["missed"]:
            print(f"    missed      {m['time']}  {who(m['player'])} {m['ball']}")
        for m in row["wrong_or_extra"]:
            print(f"    wrong/extra {m['time']}  {who(m['player'])} {m['ball']}")
    print(f"\nPocketed balls: {report['pots_correct']}/{report['pots_in_key']} "
          f"= {report['pot_accuracy']:.1%}  (ball only: "
          f"{report['ball_accuracy_ignoring_player']:.1%}; "
          f"{report['extra_or_wrong_pots']} wrong or extra calls)")
    print("\nDefinition of done:")
    for name, ok in report["criteria"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("  ----  runs live on the Pi: see bench.py")
    print(f"\n{'PASS' if report['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key", help="answer key CSV")
    ap.add_argument("records", nargs="+",
                    help="match record JSON(s) from main.py, in the order played")
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args()

    run, names = load_run(args.records)
    key = load_key(args.key, names)
    if not key:
        sys.exit(f"{args.key} has no rows.")
    report = evaluate(key, run)
    show(report, names)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Saved:  {args.json}")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
