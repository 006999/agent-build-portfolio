"""Read a run's log back and rebuild what happened. Not part of the agent.

This lives in scripts/ and nothing in src/ imports it, on purpose. src/agent.py
has no read path to its own log file - a file the agent writes and never reads
back is a log, not memory (Day 4). Putting the reader here is what enforces that
rather than merely asserting it.

The rebuild is done from the events, not from a dump. No single event carries the
transcript; the conversation is replayed:

    run_start        seeds it with the task
    reply            appends the assistant turn, blocks and all
    tool_result      appends the tool turn (all results of one cycle in one turn)
    completion_check appends the nudge, when there is one
    tool_failed /    Day 10: append nothing, and DROP the results collected for
    tool_unknown     that cycle - the loop never sent them, so they are not part
                     of the conversation. They stay in tool_calls.

That is the design decision the whole Day 5 test rests on. If the exit event
simply dumped `conversation`, comparing the log against the loop's return object
would be comparing the return object against a copy of itself, and the test could
not fail for the reason it exists.

Day 7 adds the split. A log file holds as many runs as have been appended to it -
the private build's workflow appends to one log file on every run - and until
today this
reader assumed one file meant one run. Pointed at the real four-run log it
produced one merged run that never happened: the first run's run_id, the last
run's stop_reason and cycles, every run's tokens concatenated, and a third run's
error. It did not raise.

`run_id` is on every event and `seq` restarts at 1 per run, so the file always
carried enough to separate them. `parse_all` does the separating; `parse` now
refuses a multi-run file rather than merging it, because a wrong answer that
looks right is worse than an error.

    .venv/bin/python scripts/read_log.py logs/run.jsonl
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunLog:
    """One run, rebuilt from its log. Field names match AgentResult where they overlap."""

    run_id: str = ""
    events: list[dict] = field(default_factory=list)

    # The same five things AgentResult reports.
    stop_reason: str | None = None
    cycles: int | None = None
    final_answer: str | None = None
    conversation: list[dict] = field(default_factory=list)
    blocked_by: str | None = None
    block_reason: str | None = None
    # Day 10. Composed exactly as AgentResult composes them, from the event.
    failed_tool: str | None = None
    failure_reason: str | None = None

    # Things AgentResult does not report and the log does.
    tool_calls: list[dict] = field(default_factory=list)
    tokens: list[tuple[int, int]] = field(default_factory=list)
    error: dict | None = None
    # Day 10. The tool_failed or tool_unknown event, minus the envelope fields.
    tool_failure: dict | None = None

    # Day 10. How many replies came back with each api_stop_reason, in the
    # order each reason first appeared. The loop's replies and the document
    # request's replies are counted apart, for the same reason log.tokens and
    # log.document are kept apart: they are different calls.
    #
    # A 29 Sep workflow run's log held "max_tokens" on 4 of 6 replies and no
    # reader surfaced it. A count printed beside the token line would have.
    api_stop_reasons: dict[str, int] = field(default_factory=dict)
    document_stop_reasons: dict[str, int] = field(default_factory=dict)

    # Day 8. The schema-enforced document request, which the private build's
    # workflow makes after run_agent has returned. That workflow is not in this
    # public copy, so no test here exercises the document-event branches. It is a real API call the loop never
    # makes, so without this the log showed a document appearing from nowhere.
    #
    # Read from document_request / document_reply and NOT from request / reply,
    # which are the loop's. Those two feed the cross-check below: every request
    # event must line up with a rebuilt turn, and this call appends no turn to
    # the rebuilt conversation. Folding it into them would report every healthy
    # run INCONSISTENT.
    document: dict | None = None
    draft: dict | None = None

    # Day 9. A document that failed the contract, attempt by attempt, and the
    # escalation if the last attempt failed too. `document` above keeps its Day
    # 8 meaning - the most recent document_reply - so on a retried run it is the
    # last attempt; the earlier ones are here, with the payload each returned.
    document_attempts: list[dict] = field(default_factory=list)
    escalation: dict | None = None

    # The internal cross-check. `message_count_at_request` is what the loop said
    # it was about to send; `rebuilt_length_at_request` is what replaying the
    # events actually produced. They must match at every cycle - a dropped event
    # shows up here even when nothing else notices.
    message_count_at_request: list[int] = field(default_factory=list)
    rebuilt_length_at_request: list[int] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        """False when the run raised. A crash writes no run_end."""
        return any(event["event"] == "run_end" for event in self.events)

    @property
    def event_types(self) -> list[str]:
        return [event["event"] for event in self.events]


def read_events(path) -> list[dict]:
    """Every event in the file, in the order the writer sequenced them.

    Sorted by `seq` rather than trusting line order. Line order is almost always
    the same thing, but `seq` is the field that promises it.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return sorted(events, key=lambda event: event["seq"])


def split_by_run(events: list[dict]) -> list[list[dict]]:
    """Group events by run_id, in the order each run first appears in the file.

    A dict rather than a sort or an itertools.groupby, and both alternatives are
    wrong here for the same reason: runs are not guaranteed to be contiguous.
    Two JsonlLoggers pointing at one path interleave, and groupby would then cut
    a single run into several pieces. Grouping on the key itself cannot.

    Order is first appearance rather than sorted, because run_id is a uuid4 hex
    and sorting it would order runs by nothing at all.
    """
    groups: dict[str, list[dict]] = {}
    for event in events:
        groups.setdefault(event["run_id"], []).append(event)
    return list(groups.values())


def parse_all(path) -> list[RunLog]:
    """Rebuild every run in one log file.

    This is the one to use on a real log. `parse` is for a file already known to
    hold a single run.
    """
    events = read_events(path)
    if not events:
        return []
    return [_rebuild(group) for group in split_by_run(events)]


def parse(path) -> RunLog:
    """Rebuild one run from a file holding exactly one run.

    Raises on a file holding more than one. Merging them is what this reader used
    to do, and the result was a run that never happened - reported with no error,
    which is the worst way for an audit tool to be wrong. Callers with a real log
    want `parse_all`.
    """
    events = read_events(path)
    if not events:
        return RunLog(events=events)

    groups = split_by_run(events)
    if len(groups) > 1:
        raise ValueError(
            f"{path} holds {len(groups)} runs, not one. parse() rebuilds a single "
            "run; use parse_all() for a file that has been appended to."
        )
    return _rebuild(events)


def _rebuild(events: list[dict]) -> RunLog:
    """Replay one run's events into a RunLog. Every event must share a run_id."""
    log = RunLog(events=events)
    if not events:
        return log

    log.run_id = events[0]["run_id"]

    task = ""
    # A run that never made a request has no conversation - the loop builds its
    # `messages` list after the input guardrails, so an input-blocked run really
    # does return an empty one. Seeding unconditionally would invent a turn that
    # was never sent.
    made_a_request = any(event["event"] == "request" for event in events)

    # Results of one cycle go into a single user turn, the way the loop batches
    # them, so they are collected and flushed rather than appended one by one.
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            log.conversation.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for event in events:
        kind = event["event"]

        # Day 10. A tool failure ends the cycle before its results are sent, so
        # results already collected that cycle are not in the conversation the
        # loop returns. Dropped from the rebuild here; tool_calls keeps them.
        if kind in ("tool_failed", "tool_unknown"):
            pending_results.clear()

        if kind != "tool_result":
            flush_results()

        if kind == "run_start":
            task = event["task"]
            if made_a_request:
                log.conversation.append({"role": "user", "content": task})

        elif kind == "request":
            log.message_count_at_request.append(event["message_count"])
            log.rebuilt_length_at_request.append(len(log.conversation))

        elif kind == "reply":
            log.conversation.append({"role": "assistant", "content": event["content"]})
            log.tokens.append((event["input_tokens"], event["output_tokens"]))
            reason = event["api_stop_reason"]
            log.api_stop_reasons[reason] = log.api_stop_reasons.get(reason, 0) + 1

        elif kind == "tool_result":
            log.tool_calls.append(
                {
                    "tool_use_id": event["tool_use_id"],
                    "name": event["name"],
                    "args": event["args"],
                    "result": event["result"],
                }
            )
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": event["tool_use_id"],
                    "content": event["result"],
                }
            )

        elif kind == "tool_failed":
            log.tool_failure = {
                "kind": kind,
                "tool_use_id": event["tool_use_id"],
                "name": event["name"],
                "args": event["args"],
                "exception_type": event["exception_type"],
                "message": event["message"],
                "cycle": event["cycle"],
            }
            log.failed_tool = event["name"]
            log.failure_reason = f"{event['exception_type']}: {event['message']}"

        elif kind == "tool_unknown":
            log.tool_failure = {
                "kind": kind,
                "tool_use_id": event["tool_use_id"],
                "name": event["name"],
                "args": event["args"],
                "registered": event["registered"],
                "cycle": event["cycle"],
            }
            log.failed_tool = event["name"]
            log.failure_reason = f"{event['name']!r} is not a registered tool"

        elif kind == "completion_check":
            if event["nudge"] is not None:
                log.conversation.append({"role": "user", "content": event["nudge"]})

        elif kind == "document_request":
            log.document = {
                "model": event["model"],
                "message_count": event["message_count"],
            }

        elif kind == "document_reply":
            # setdefault rather than assignment: a run that crashed between the
            # two events has a request and no reply, and that asymmetry is the
            # thing worth seeing rather than an AttributeError.
            log.document = dict(log.document or {})
            log.document.update(
                {
                    "api_stop_reason": event["api_stop_reason"],
                    "input_tokens": event["input_tokens"],
                    "output_tokens": event["output_tokens"],
                }
            )
            reason = event["api_stop_reason"]
            log.document_stop_reasons[reason] = (
                log.document_stop_reasons.get(reason, 0) + 1
            )

        elif kind == "document_attempt_failed":
            log.document_attempts.append(
                {
                    "attempt": event["attempt"],
                    "max_attempts": event["max_attempts"],
                    "problems": event["problems"],
                    "payload": event["payload"],
                }
            )

        elif kind == "document_escalated":
            log.escalation = {
                "path": event["path"],
                "attempts": event["attempts"],
                "problems": event["problems"],
                "payload": event["payload"],
            }

        elif kind == "draft_written":
            # Day 8 note: before today nothing here read this event at all. It
            # has been written since Day 6 with the full document splatted into
            # it, and _rebuild ignored it silently along with every other kind
            # it did not name. The audit-trail claim was true only because no
            # one asked the reader for the one thing the run produced.
            log.draft = {
                "slug": event["slug"],
                "path": event["path"],
                "confidence": event["confidence"],
            }

        elif kind == "blocked":
            log.blocked_by = event["guardrail"]
            log.block_reason = event["reason"]

        elif kind == "error":
            log.error = {
                "exception_type": event["exception_type"],
                "message": event["message"],
                "cycle": event["cycle"],
            }

        elif kind == "run_end":
            log.stop_reason = event["stop_reason"]
            log.cycles = event["cycles"]
            log.final_answer = event["final_answer"]

    flush_results()
    return log


def format_tally(counts: dict[str, int]) -> str:
    """{"tool_use": 2, "max_tokens": 4} as "tool_use=2, max_tokens=4", or "none"."""
    if not counts:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in counts.items())


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/read_log.py <logfile>")
        return 1

    logs = parse_all(sys.argv[1])
    print(f"{len(logs)} run(s) in {sys.argv[1]}")

    for index, log in enumerate(logs, start=1):
        print(f"\n--- run {index} of {len(logs)} ---")
        print(f"run_id:       {log.run_id}")
        print(f"stop reason:  {log.stop_reason if log.finished else '(never finished)'}")
        print(f"cycles:       {log.cycles}")
        print(f"turns:        {len(log.conversation)}")
        print(f"tokens:       {log.tokens}")
        print(
            f"api stops:    {format_tally(log.api_stop_reasons)}"
            f" | document: {format_tally(log.document_stop_reasons)}"
        )
        # The internal cross-check, printed rather than left to the tests. It is
        # the one line that says whether the rebuild can be trusted at all.
        rebuilt_ok = log.message_count_at_request == log.rebuilt_length_at_request
        print(f"rebuild:      {'consistent' if rebuilt_ok else 'INCONSISTENT'}")
        if log.document is not None:
            print(f"document:     {log.document}")
        if log.draft is not None:
            print(f"draft:        {log.draft['path']} ({log.draft['confidence']})")
        if log.document_attempts:
            print(f"failed tries: {len(log.document_attempts)}")
        if log.escalation is not None:
            print(
                f"escalated:    {log.escalation['path']} "
                f"(attempts={log.escalation['attempts']})"
            )
        if log.blocked_by is not None:
            print(f"blocked by:   {log.blocked_by}")
            print(f"because:      {log.block_reason}")
        if log.error is not None:
            print(f"error:        {log.error['exception_type']}: {log.error['message']}")
        if log.tool_failure is not None:
            print(f"tool failed:  {log.failed_tool} - {log.failure_reason}")
        print("tools called:")
        for call in log.tool_calls:
            print(f"  {call['name']}({call['args']}) -> {call['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
