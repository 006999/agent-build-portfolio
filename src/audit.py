"""The audit trail writer. Day 5: record what happened. No reading, no rotation.

This file writes. It never reads. src/agent.py does not import it - the caller
constructs a logger and passes it into run_agent as `on_event`, which is why the
loop has neither a read path nor a write path to its own log file. That is the
Day 4 rule stated the strong way: a file the agent writes and never reads back
is a log, not memory, and here the agent does not even write it.

The parser lives in scripts/read_log.py and is imported by nothing in src/.

One JSON object per line, appended. Chosen over a single JSON array because an
array has to be closed to be valid, and the one run that most needs a readable
log is the one that crashed before it could close anything.
"""

import datetime
import json
import uuid
from pathlib import Path


def ignore(event_type: str, **fields) -> None:
    """The default on_event. Logging is opt-in.

    Named without a leading underscore because src/agent.py imports it as a
    default argument, and a default that has to be reached into another module's
    private namespace is not a default anyone can use.
    """


class JsonlLogger:
    """Writes one line of JSON per event. Callable, so it is passed as on_event.

    Two fields are added to every event that the loop does not supply:

      run_id  - without it, a file holding two runs is not an audit trail. The
                caller may pass one in to correlate a run with its own records.
      seq     - the ordering instrument. Wall-clock timestamps can go backwards
                across an NTP correction; a counter cannot.

    `ts` is recorded because an auditor asks "when", not "in what order". It is
    the record, not the ordering.
    """

    def __init__(self, path, run_id: str | None = None):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event_type: str, **fields) -> None:
        self._seq += 1
        event = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": self.run_id,
            "seq": self._seq,
            "event": event_type,
        }
        event.update(fields)

        # Opened and closed per event rather than held open. A buffered handle
        # relies on garbage collection to flush, and the crash scenario is
        # exactly the case where relying on cleanup is how the last event -
        # the one naming the exception - goes missing.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    # Both defaults are load-bearing, and both are the strict choice.
    #
    # There is no `default=` fallback. An earlier version passed `default=str`
    # so that an object this file cannot serialise was recorded as its repr
    # rather than raising. That was removed: the claim this file makes is that a
    # run can be rebuilt from the log, and a repr cannot be rebuilt into the
    # value it came from. A log that quietly holds "<Invoice object at 0x...>"
    # where the run held an invoice passes every test and fails the only
    # question anyone will ask it. Raising is louder and earlier - and the
    # realistic trigger, a tool returning a non-string, was already going to end
    # the run a moment later at the API boundary, where tool_result content must
    # be a string.
    #
    # `allow_nan=False` closes the same hole from the other side. The default
    # emits bare NaN and Infinity, which Python reads back and nothing else
    # does - not jq, not another language, not a log pipeline. That is a file
    # only this program can audit, which is not an audit trail. A `default=`
    # fallback never caught it either way: float is a type the encoder handles,
    # so the fallback is never consulted.
    #
    # Both are recorded as known gaps in the private build's notes rather than
    # silent behaviour, and Day 8
    # (structured outputs) is when the question becomes real.
