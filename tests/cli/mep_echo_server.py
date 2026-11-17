"""Minimal MEP/1 stdio server used by the `env check` tests (one JSON object per line).

Flags: ``--bad-spec`` omits the name, ``--die-on-step`` exits before answering step,
``--int-reference`` makes it a guessing game: reset fails unless the sample's reference is an
integer, and a step is rewarded 1.0 and ends the episode when its text is that integer.
Reset echoes the sample it was sent in ``info["sample"]``.
"""

import json
import sys

bad_spec = "--bad-spec" in sys.argv
die_on_step = "--die-on-step" in sys.argv
int_reference = "--int-reference" in sys.argv
turns: dict[str, int] = {}
secrets: dict[str, int] = {}

for line in sys.stdin:
    req = json.loads(line)
    op, ep = req.get("op"), req.get("episode_id")
    if op == "spec":
        out = {
            "version": "1",
            "protocol": "MEP/1",
            "max_turns": 3,
            "parallel_episodes": 2,
            "description": "echo",
        }
        if not bad_spec:
            out["name"] = "echo"
    elif op == "health":
        out = {"ok": True}
    elif op == "reset":
        turns[ep] = 0
        sample = req["sample"]
        if int_reference and type(sample["reference"]) is not int:
            message = f"reference must be an integer, got {sample['reference']!r}"
            out = {"error": {"type": "ValueError", "message": message}}
        else:
            secrets[ep] = sample["reference"]
            out = {"messages": sample["messages"], "info": {"seed": req.get("seed"), "sample": sample}}
    elif op == "step":
        if die_on_step:
            sys.exit(3)
        turns[ep] += 1
        text = req["action"]["text"]
        solved = int_reference and text == str(secrets[ep])
        done = solved or turns[ep] >= 3
        out = {
            "messages": [] if done else [{"role": "user", "content": f"echo:{text}"}],
            "reward": float(solved) if int_reference else float(len(text)),
            "done": done,
            "info": {"turn": turns[ep]},
        }
    elif op == "close":
        turns.pop(ep, None)
        secrets.pop(ep, None)
        out = {}
    else:
        out = {"error": {"type": "bad_op", "message": f"unknown op {op!r}"}}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
