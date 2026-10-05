"""Does the log say what actually happened? Day 5.

The shape of every test here is the same, and it is the point:

  1. run the loop, with a real logger writing a real file
  2. take ground truth from the loop's RETURN OBJECT, never from the log
  3. parse the log with scripts/read_log.py, which cannot see the return object
  4. assert the two agree field for field

A test that checked the log file was non-empty, or well-formed, or contained a
few expected strings would pass against a logger that dropped half the run. The
question is not "did it write something" but "is what it wrote true".

Five scenarios. Four of them return an AgentResult. The fifth crashes, so there
is no return object, and it needs different ground truth - see the crash section
at the bottom for what replaces it and why one crash scenario is not enough.

Day 10 moved two of the three crash sites into the returning group. A tool that
raises and a tool nobody registered now end the run with run_end, so they are
checked with the same field-for-field assertion as every other returning run.
Only the API call itself still crashes.
"""

import json

import pytest
from fakes import (
    FakeClient,
    FakeGuardrail,
    FakeResponse,
    FakeTextBlock,
    FakeTool,
    FakeToolUseBlock,
)

from scripts import read_log
from scripts.read_log import parse, parse_all
from src.agent import COMPLETION_SENTINEL, CONTINUE_NUDGE, StopReason, run_agent
from src.audit import JsonlLogger

# ---------------------------------------------------------------------------
# Ground truth, dug out of the return object. Nothing here reads the log.
# ---------------------------------------------------------------------------


def _tool_calls_from(conversation: list[dict]) -> list[dict]:
    """Every tool call in a transcript: name, args and result, paired by id.

    The pairing is the part that matters. A tool_use block holds the name and
    the arguments; the result arrives in a separate block in a later message,
    tied to it only by tool_use_id. Rebuilding that join here, from the return
    object, is what makes it independent of however the log chose to record it.
    """
    calls: dict[str, dict] = {}
    order: list[str] = []

    for message in conversation:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block["type"] == "tool_use":
                calls[block["id"]] = {
                    "tool_use_id": block["id"],
                    "name": block["name"],
                    "args": block["input"],
                    "result": None,
                }
                order.append(block["id"])
            elif block["type"] == "tool_result":
                calls[block["tool_use_id"]]["result"] = block["content"]

    return [calls[tool_use_id] for tool_use_id in order]


def _assert_log_matches(result, client, log) -> None:
    """The whole assertion, in one place, applied to all four returning scenarios."""
    assert log.stop_reason == result.stop_reason.value
    assert log.cycles == result.cycles
    assert log.final_answer == result.final_answer
    assert log.blocked_by == result.blocked_by
    assert log.block_reason == result.block_reason
    assert log.failed_tool == result.failed_tool
    assert log.failure_reason == result.failure_reason

    # Rebuilt from the events, not read from a dump.
    assert log.conversation == result.conversation

    # Day 10. A run that stops on a tool failure ends with a tool_use turn that
    # has no result turn: the failed call, plus any call that ran earlier in the
    # same cycle, whose result was logged and never sent. So only calls the
    # conversation actually answered are ground truth here. When no tool failed
    # the log must hold exactly those; when one did, it holds them first, and
    # the tests that script the failure check the rest against the tools' own
    # call records.
    answered = [
        call for call in _tool_calls_from(result.conversation) if call["result"] is not None
    ]
    if result.failed_tool is None:
        assert log.tool_calls == answered
    else:
        assert log.tool_calls[: len(answered)] == answered

    # Ground truth is what the fake recorded handing back, not a formula this
    # test re-derives.
    assert log.tokens == client.messages.usages

    # One API call per cycle, and one set of token counts per API call.
    assert len(log.tokens) == len(client.messages.calls)

    # The log's internal cross-check: what the loop said it was about to send
    # against what replaying the events actually rebuilt.
    assert log.message_count_at_request == log.rebuilt_length_at_request

    # Every run that returns writes exactly one run_end, and it is last.
    assert log.finished
    assert log.event_types.count("run_end") == 1
    assert log.event_types[-1] == "run_end"

    _assert_timestamps_are_sane(log)


def _assert_timestamps_are_sane(log) -> None:
    """Weaker than every other assertion here, and deliberately labelled so.

    The return object carries no timestamps, so there is nothing to compare
    against - only internal consistency. `seq` is the ordering instrument and is
    strictly increasing by construction; `ts` is the record. A backwards NTP
    correction mid-test would fail the non-decreasing check, which is a real if
    remote flake, and worth knowing about rather than discovering later.
    """
    assert all(event.get("ts") for event in log.events)
    assert [e["seq"] for e in log.events] == sorted(e["seq"] for e in log.events)
    timestamps = [event["ts"] for event in log.events]
    assert timestamps == sorted(timestamps)


def _run(tmp_path, name="run.jsonl", **kwargs):
    """Run the loop with a logger and hand back (result, log, path)."""
    path = tmp_path / name
    logger = JsonlLogger(path)
    result = run_agent(on_event=logger, **kwargs)
    return result, parse(path), path


# ---------------------------------------------------------------------------
# 1: normal completion, with a tool call in it.
# ---------------------------------------------------------------------------


def test_a_completed_run_is_recorded_exactly(tmp_path):
    """Deliberately not a one-cycle text-only run.

    A run with no tool call would let a logger that drops tool arguments pass,
    and that is the exact fault this suite has to catch.
    """
    client = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"sides": 20})],
            f"You rolled a 4.\n{COMPLETION_SENTINEL}",
        ]
    )
    result, log, _ = _run(
        tmp_path,
        task="Roll a die.",
        tools=[FakeTool(result="rolled a 4")],
        client=client,
    )

    assert result.stop_reason is StopReason.COMPLETE
    _assert_log_matches(result, client, log)

    # The arguments specifically, spelled out. The helper above would still pass
    # if both sides recorded an empty dict.
    assert log.tool_calls == [
        {
            "tool_use_id": "toolu_1",
            "name": "roll_dice",
            "args": {"sides": 20},
            "result": "rolled a 4",
        }
    ]


def test_two_text_blocks_in_one_reply_survive_the_round_trip(tmp_path):
    """Why the reply event carries `content` rather than a flattened string.

    The loop joins text blocks with a newline for its own purposes. If the log
    recorded that joined string, rebuilding would produce one block where the
    transcript has two, and the conversation assertion would fail. This pins it.
    """
    client = FakeClient(
        [
            [FakeTextBlock("First part."), FakeTextBlock(COMPLETION_SENTINEL)],
        ]
    )
    result, log, _ = _run(tmp_path, task="Say something.", client=client)

    assert len(result.conversation[1]["content"]) == 2
    _assert_log_matches(result, client, log)


# ---------------------------------------------------------------------------
# 2: max cycles.
# ---------------------------------------------------------------------------


def test_a_max_cycles_run_is_recorded_exactly(tmp_path):
    """Ends on a nudge, not on a sentinel, and every cycle has its own tokens."""
    client = FakeClient(["Still going.", "Still going.", "Still going."])
    result, log, _ = _run(tmp_path, task="Count to a million.", client=client, max_cycles=3)

    assert result.stop_reason is StopReason.MAX_CYCLES
    assert result.cycles == 3
    _assert_log_matches(result, client, log)

    # Three distinct pairs. A logger that recorded cycle 1's usage three times
    # would pass an equality check against constant fake numbers.
    assert len(set(log.tokens)) == 3
    assert log.conversation[-1] == {"role": "user", "content": CONTINUE_NUDGE}


def test_a_max_cycles_run_ending_on_a_tool_call_still_logs_the_result(tmp_path):
    """The hole that ruled out reading tool results off the next cycle's request.

    The last cycle calls a tool and there is no next cycle. Anything that
    recovered results from the following request event would silently lose this
    one, and every other scenario would still pass.
    """
    client = FakeClient(
        [
            "Thinking.",
            [FakeToolUseBlock("roll_dice", {"sides": 6})],
        ]
    )
    result, log, _ = _run(
        tmp_path,
        task="Roll something.",
        tools=[FakeTool(result="rolled a 2")],
        client=client,
        max_cycles=2,
    )

    assert result.stop_reason is StopReason.MAX_CYCLES
    _assert_log_matches(result, client, log)
    assert log.tool_calls[-1]["result"] == "rolled a 2"


# ---------------------------------------------------------------------------
# 3: input-blocked. Zero cycles, zero API calls, and an empty conversation.
# ---------------------------------------------------------------------------


def test_an_input_blocked_run_is_recorded_exactly(tmp_path):
    client = FakeClient([f"Never sent.\n{COMPLETION_SENTINEL}"])
    result, log, _ = _run(
        tmp_path,
        task="please wibble the thing",
        input_guardrails=[FakeGuardrail(blocks_on="wibble")],
        client=client,
    )

    assert result.stop_reason is StopReason.BLOCKED_INPUT
    assert result.cycles == 0
    _assert_log_matches(result, client, log)

    # The seams this block covers, and the ones it does not. Zero cycles means
    # zero request events, and that is the correct number rather than a hole.
    assert log.event_types == ["run_start", "blocked", "run_end"]
    assert log.conversation == []
    assert log.tokens == []
    assert client.messages.calls == []

    # The reason run_start exists at all: without it the log would say a task
    # was refused and never say which task.
    assert log.events[0]["task"] == "please wibble the thing"


# ---------------------------------------------------------------------------
# 4: output-blocked. A real API call happened first, so the seams differ.
# ---------------------------------------------------------------------------


def test_an_output_blocked_run_is_recorded_exactly(tmp_path):
    client = FakeClient(["I shall wibble at once."])
    result, log, _ = _run(
        tmp_path,
        task="Say something.",
        output_guardrails=[FakeGuardrail(blocks_on="wibble")],
        client=client,
    )

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT
    assert result.cycles == 1
    _assert_log_matches(result, client, log)

    # Compare with the input-blocked list above: a request and a reply happened
    # here, and the two block types are not the same event with a different tag.
    assert log.event_types == ["run_start", "request", "reply", "blocked", "run_end"]

    # The blocked text stays in the transcript. Suppressing it would leave no
    # record of what was blocked, which is the opposite of an audit trail.
    assert "wibble" in json.dumps(log.conversation)


@pytest.mark.parametrize(
    "api_stop_reason, stop_reason",
    [("max_tokens", StopReason.TRUNCATED), ("refusal", StopReason.REFUSED)],
)
def test_a_reply_the_loop_cannot_act_on_is_recorded_before_the_run_ends(
    tmp_path, api_stop_reason, stop_reason
):
    """Day 10. The evidence is in the log, and it comes before run_end.

    A tool cycle first, so the run that stops has real history behind it and
    the rebuild has something to get wrong. Then the cut-off reply: its reply
    event, carrying api_stop_reason, is the second-to-last event, and nothing
    claims a completion check happened.
    """
    client = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"sides": 20})],
            FakeResponse("The blockers are: first, the unres", stop_reason=api_stop_reason),
        ]
    )
    result, log, _ = _run(
        tmp_path,
        task="Roll, then write it up.",
        tools=[FakeTool(result="rolled a 4")],
        client=client,
    )

    assert result.stop_reason is stop_reason
    _assert_log_matches(result, client, log)

    assert log.event_types == [
        "run_start",
        "request",
        "reply",
        "tool_result",
        "request",
        "reply",
        "run_end",
    ]
    last_reply = [event for event in log.events if event["event"] == "reply"][-1]
    assert last_reply["api_stop_reason"] == api_stop_reason
    assert last_reply["content"] == [
        {"type": "text", "text": "The blockers are: first, the unres"}
    ]


def test_the_two_block_types_do_not_produce_the_same_events(tmp_path):
    """One assertion holding the two apart, so a later refactor cannot merge them."""
    blocked_input, input_log, _ = _run(
        tmp_path,
        name="input.jsonl",
        task="please wibble",
        input_guardrails=[FakeGuardrail(blocks_on="wibble")],
        client=FakeClient([]),
    )
    blocked_output, output_log, _ = _run(
        tmp_path,
        name="output.jsonl",
        task="Say something.",
        output_guardrails=[FakeGuardrail(blocks_on="wibble")],
        client=FakeClient(["I shall wibble."]),
    )

    assert blocked_input.stop_reason is StopReason.BLOCKED_INPUT
    assert blocked_output.stop_reason is StopReason.BLOCKED_OUTPUT
    assert "request" not in input_log.event_types
    assert "request" in output_log.event_types
    assert "reply" not in input_log.event_types
    assert "reply" in output_log.event_types


# ---------------------------------------------------------------------------
# 5: the crash path. No return object, so ground truth comes from elsewhere.
#
# Three crash sites, and they are not interchangeable. One scenario would
# exercise one of them and leave the other two unproven.
# ---------------------------------------------------------------------------


class _ExplodingMessages:
    """Stands in for the SDK failing: a dropped connection, a 429."""

    def __init__(self, error: Exception):
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # Recorded before raising, the way a real call that reaches the network
        # and then fails would be.
        self.calls.append(kwargs)
        raise self._error


class ExplodingClient:
    def __init__(self, error: Exception):
        self.messages = _ExplodingMessages(error)


def _raise_boom(**kwargs):
    raise RuntimeError("the tool exploded")


def _assert_crash_log(log, exc_info, expected_types, calls_made) -> None:
    """Ground truth without a return object.

    Three instruments, none of them the log:

      the exception from pytest.raises  - type and message
      client.messages.calls             - what was actually sent
      (the caller's own tool records)   - checked at each call site

    tests/test_memory.py already treats client.messages.calls this way, for the
    same reason: it is a record of what happened that the thing under test does
    not write.
    """
    assert log.error is not None
    assert log.error["exception_type"] == type(exc_info.value).__name__
    assert log.error["message"] == str(exc_info.value)

    # Which cycle, cross-checked against an independent instrument.
    assert log.error["cycle"] == calls_made

    assert log.event_types == expected_types

    # No run_end, because run_agent never returned. And specifically: the error
    # event is LAST. "No run_end" alone would pass a logger that wrote the error
    # and then quietly wrote something else.
    assert not log.finished
    assert log.event_types[-1] == "error"

    _assert_timestamps_are_sane(log)


def test_a_crash_inside_the_api_call_is_recorded(tmp_path):
    """5(a). The only test in the suite that pins the request event's position.

    It fires before client.messages.create, not after. In a run that succeeds,
    both orderings write a byte-identical log, so every other scenario passes
    either way. Move the request event below the call as a tidy-up and only this
    goes red. See "loop: the request event moves to after the API call" in
    scripts/mutate.py.
    """
    client = ExplodingClient(ConnectionError("connection reset by peer"))
    path = tmp_path / "run.jsonl"

    with pytest.raises(ConnectionError) as exc_info:
        run_agent("Do something.", client=client, on_event=JsonlLogger(path))

    log = parse(path)
    _assert_crash_log(
        log,
        exc_info,
        expected_types=["run_start", "request", "error"],
        calls_made=len(client.messages.calls),
    )

    # No reply came back, so there are no token counts for this cycle. An
    # absence, recorded as an absence.
    assert log.tokens == []


def test_an_unregistered_tool_name_is_recorded_exactly(tmp_path):
    """5(b), Day 10. The model asks for a tool that is not in the registry.

    It used to crash. It now returns UNKNOWN_TOOL, so the log is checked against
    a return object like any other returning run. The audit-relevant fact is
    still WHICH tool it asked for, and the tool_unknown event carries it along
    with what was registered.
    """
    client = FakeClient([[FakeToolUseBlock("summon_kraken", {"depth": 900})]])

    result, log, _ = _run(
        tmp_path,
        task="Do something.",
        tools=[FakeTool(name="roll_dice")],
        client=client,
    )

    assert result.stop_reason is StopReason.UNKNOWN_TOOL
    _assert_log_matches(result, client, log)

    assert log.event_types == ["run_start", "request", "reply", "tool_unknown", "run_end"]
    assert log.tool_calls == []
    assert log.tool_failure == {
        "kind": "tool_unknown",
        "tool_use_id": "toolu_1",
        "name": "summon_kraken",
        "args": {"depth": 900},
        "registered": ["roll_dice"],
        "cycle": 1,
    }
    assert log.error is None


def test_a_failure_in_the_second_of_two_tools_keeps_the_first_result(tmp_path):
    """5(c), Day 10. Two tools in one reply. The first succeeds, the second raises.

    Still the one a single-tool scenario would miss: the completed result has to
    be on the record before the failure, so the log says WHICH call ended the
    run. The exact event order is what pins that - see "logger: tool results
    are not flushed before a crash" in scripts/mutate.py.

    New on Day 10: the first result is in the log but NOT in the conversation.
    It was never sent to the model, and the rebuild has to agree.
    """
    client = FakeClient(
        [
            [
                FakeToolUseBlock("roll_dice", {"sides": 20}, block_id="toolu_1"),
                FakeToolUseBlock("explode", {"why": "boom"}, block_id="toolu_2"),
            ]
        ]
    )
    dice = FakeTool(name="roll_dice", result="rolled a 4")
    bomb = FakeTool(name="explode", run=_raise_boom)

    result, log, _ = _run(
        tmp_path,
        task="Do two things.",
        tools=[dice, bomb],
        client=client,
    )

    assert result.stop_reason is StopReason.TOOL_FAILED
    _assert_log_matches(result, client, log)

    assert log.event_types == [
        "run_start",
        "request",
        "reply",
        "tool_result",
        "tool_failed",
        "run_end",
    ]

    # Both tools really were reached - otherwise this test passes because the
    # second one was never called, which proves nothing about flushing.
    assert dice.calls == [{"sides": 20}]
    assert bomb.calls == [{"why": "boom"}]

    # The completed one is on the record, with its arguments and its result.
    assert log.tool_calls == [
        {
            "tool_use_id": "toolu_1",
            "name": "roll_dice",
            "args": {"sides": 20},
            "result": "rolled a 4",
        }
    ]
    # ...and not in the conversation, on either side.
    assert _tool_calls_from(result.conversation)[0]["result"] is None
    assert log.conversation[-1]["role"] == "assistant"

    assert log.tool_failure == {
        "kind": "tool_failed",
        "tool_use_id": "toolu_2",
        "name": "explode",
        "args": {"why": "boom"},
        "exception_type": "RuntimeError",
        "message": "the tool exploded",
        "cycle": 1,
    }
    assert log.error is None


# ---------------------------------------------------------------------------
# The contract that lets every Day 1-4 test keep passing unchanged.
# ---------------------------------------------------------------------------


def test_the_default_is_silence(tmp_path):
    """No on_event means nothing is written and nothing changes.

    111 tests written before today pass no logger. If the default did anything
    at all, they would be exercising a different loop than the one they were
    written against.
    """
    client = FakeClient([f"Done.\n{COMPLETION_SENTINEL}"])
    result = run_agent("Do something.", client=client)

    assert result.stop_reason is StopReason.COMPLETE
    assert list(tmp_path.iterdir()) == []


def test_the_loop_never_opens_the_log_itself(tmp_path):
    """src/agent.py has no read path AND no write path to its own log.

    The caller constructs the logger and hands it in. Day 4's rule was that a
    file the agent writes and never reads back is a log, not memory; this is the
    stronger form, where the agent does not write it either.
    """
    import inspect

    import src.agent

    source = inspect.getsource(src.agent)
    assert "open(" not in source
    assert "Path(" not in source
    assert "read_log" not in source


# ---------------------------------------------------------------------------
# 6: a guardrail that RAISES rather than refusing.
#
# Refusing is an ordinary outcome and Day 3 covers it. Raising is a crash, and
# until now only half of it was recorded: output guardrails run inside the loop
# body and were caught by the try/except there, while input guardrails run above
# the loop and produced a log that stopped at run_start with no explanation.
#
# Two scenarios rather than one. "The output side is already covered by the loop
# body" is structurally true and was, until this test existed, believed rather
# than measured - which is the shape of every CLAIMED-labelled-as-PROVEN entry
# this ledger exists to prevent.
# ---------------------------------------------------------------------------


def test_an_input_guardrail_that_raises_is_recorded(tmp_path):
    """6(a). The gap left open by the first Day 5 logging commit.

    Nothing has been sent, so the cross-check in _assert_crash_log compares the
    error's cycle against zero requests made. The log is run_start then error:
    no request, because the client is never even constructed on this path.
    """
    boom = RuntimeError("the guardrail's regex blew up")
    guard = FakeGuardrail(name="exploding_guard", raises=boom)
    client = FakeClient([f"Never reached.\n{COMPLETION_SENTINEL}"])
    path = tmp_path / "run.jsonl"

    with pytest.raises(RuntimeError) as exc_info:
        run_agent(
            "A perfectly ordinary task.",
            input_guardrails=[guard],
            client=client,
            on_event=JsonlLogger(path),
        )

    log = parse(path)
    _assert_crash_log(
        log,
        exc_info,
        expected_types=["run_start", "error"],
        calls_made=len(client.messages.calls),
    )

    # The guardrail really was reached, and the API really was not. Without
    # these two the test passes if the run died somewhere else entirely.
    assert guard.seen == ["A perfectly ordinary task."]
    assert client.messages.calls == []

    # The task is still on the record. That is the whole reason run_start exists
    # above the guardrail check rather than below it.
    assert log.events[0]["task"] == "A perfectly ordinary task."


def test_an_output_guardrail_that_raises_is_recorded(tmp_path):
    """6(b). The twin. Structurally covered by the loop body - now measured.

    A real API call happened first, so unlike 6(a) this one has a request and a
    reply on the record before the error, and the error's cycle is 1.
    """
    boom = ValueError("the banned-word list is not a list")
    guard = FakeGuardrail(name="exploding_output_guard", raises=boom)
    client = FakeClient(["Here is an ordinary reply."])
    path = tmp_path / "run.jsonl"

    with pytest.raises(ValueError) as exc_info:
        run_agent(
            "A perfectly ordinary task.",
            output_guardrails=[guard],
            client=client,
            on_event=JsonlLogger(path),
        )

    log = parse(path)
    _assert_crash_log(
        log,
        exc_info,
        expected_types=["run_start", "request", "reply", "error"],
        calls_made=len(client.messages.calls),
    )

    # One call really was made and paid for - the difference between this
    # scenario and 6(a), and the reason both exist.
    assert len(client.messages.calls) == 1
    assert log.error["cycle"] == 1
    assert guard.seen == ["Here is an ordinary reply."]

    # The reply that could not be checked is still on the record. Suppressing it
    # would leave no evidence of what the guardrail choked on.
    reply = [event for event in log.events if event["event"] == "reply"][0]
    assert reply["content"] == [{"type": "text", "text": "Here is an ordinary reply."}]


# ---------------------------------------------------------------------------
# The writer's serialisation contract. Two separate holes, two exceptions.
#
# Both are deliberate strictness, so both need a test or the strictness is an
# assertion in a comment. Neither is reachable from the current system - which
# is exactly why they are tested here directly rather than through the loop.
# ---------------------------------------------------------------------------


def test_the_writer_raises_on_a_value_it_cannot_serialise(tmp_path):
    """No repr fallback. A log holding a repr cannot be rebuilt from.

    The realistic trigger is a tool returning something that is not a string:
    run_agent's docstring constrains .run(**kwargs) but says nothing about what
    it returns, and src/tools.py returning strings is a convention, not a rule.
    """
    path = tmp_path / "run.jsonl"
    log = JsonlLogger(path)

    class Invoice:
        pass

    with pytest.raises(TypeError):
        log("tool_result", actor="agent", result=Invoice())


def test_the_writer_raises_on_a_non_finite_float(tmp_path):
    """allow_nan=False. The default writes bare NaN, which is not JSON.

    Python reads it back, so this file would round-trip through
    scripts/read_log.py and pass every other test here. jq, another language,
    or any log pipeline would reject the line. A file only this program can read
    is not an audit trail, and no `default=` fallback ever caught this - float
    is a type the encoder handles, so the fallback is never consulted.
    """
    path = tmp_path / "run.jsonl"
    log = JsonlLogger(path)

    with pytest.raises(ValueError):
        log("tool_result", actor="agent", result=float("nan"))


# ---------------------------------------------------------------------------
# 6, Day 7: one file, several runs. The shape production actually writes.
#
# Every scenario above writes one run to a fresh tmp_path file, and that is why
# all of them pass against a reader that cannot tell two runs apart. Production
# does not do that: the private build's workflow opens one log file on every run and
# JsonlLogger appends, so the real file holds every run ever made against it.
#
# Found on Day 7 by pointing scripts/read_log.py at that real workflow log
# and reading the output instead of trusting the suite. It reported run 1's
# run_id, run 4's stop_reason and cycles, every run's tokens concatenated, and
# run 2's error - one merged run that never happened. It did not raise. The gate
# says "a random run is reconstructable from the log alone", and against the
# only multi-run log this repo has ever written, it was not.
#
# The log format was never the problem: run_id is on every event and seq
# restarts at 1 per run, so the file always carried enough to separate them.
# Only the reader assumed one file meant one run.
# ---------------------------------------------------------------------------


def test_three_runs_in_one_file_are_read_back_as_three(tmp_path):
    """The test the Day 5 suite could not have failed, because it never appended.

    Three runs, three different shapes, one file - the third crashes, so it
    writes no run_end and its events run straight into the next line of the same
    file with nothing to close them. Each is asserted against its OWN return
    object, which is the same ground-truth rule as every scenario above: a
    reader that merged two runs would have to invent a result that matches
    neither.
    """
    path = tmp_path / "shared.jsonl"

    # Run one: completes, with a tool call in it.
    client_one = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"sides": 20})],
            f"You rolled a 4.\n{COMPLETION_SENTINEL}",
        ]
    )
    result_one = run_agent(
        "Roll a die.",
        tools=[FakeTool(result="rolled a 4")],
        client=client_one,
        on_event=JsonlLogger(path),
    )

    # Run two: never signals complete, so it runs out of cycles. A different
    # length and a different stop reason from run one, so a merge cannot pass
    # by coincidence.
    client_two = FakeClient(["Still thinking.", "Still thinking.", "Still thinking."])
    result_two = run_agent(
        "Do something unfinishable.",
        max_cycles=3,
        client=client_two,
        on_event=JsonlLogger(path),
    )

    # Run three: crashes. No run_end, and it is not the last thing in the file
    # only because nothing follows it - the reader still has to know where run
    # two ended and run three began without a closing event to tell it.
    client_three = ExplodingClient(ConnectionError("connection reset by peer"))
    with pytest.raises(ConnectionError) as exc_info:
        run_agent("Do something.", client=client_three, on_event=JsonlLogger(path))

    logs = parse_all(path)

    assert len(logs) == 3, (
        f"three runs went into one file and the reader found {len(logs)}"
    )

    # Order is first appearance in the file, not sorted, not a set.
    one, two, three = logs
    assert one.run_id != two.run_id != three.run_id
    assert len({one.run_id, two.run_id, three.run_id}) == 3

    # Each against its own return object, using the same assertion the
    # single-run scenarios use.
    _assert_log_matches(result_one, client_one, one)
    _assert_log_matches(result_two, client_two, two)

    assert one.stop_reason == StopReason.COMPLETE.value
    assert two.stop_reason == StopReason.MAX_CYCLES.value

    # The crashed run, by the crash-path rule: no return object, so ground truth
    # is the exception and the client's own call record.
    _assert_crash_log(
        three,
        exc_info,
        expected_types=["run_start", "request", "error"],
        calls_made=len(client_three.messages.calls),
    )

    # And the specific failures the merge produced. Each of these was wrong when
    # read through parse() on the real workflow log.
    assert one.cycles == result_one.cycles
    assert two.cycles == result_two.cycles
    assert one.tokens == client_one.messages.usages
    assert two.tokens == client_two.messages.usages
    assert three.tokens == []
    assert one.error is None and two.error is None
    assert three.error is not None


def test_every_event_in_the_file_lands_in_exactly_one_run(tmp_path):
    """No event is dropped and none is counted twice.

    The test above asserts each run is right. This asserts the split is total:
    a reader that quietly discarded the runs it could not place would satisfy
    every per-run assertion and lose part of the file. Nothing else here counts
    the whole file.
    """
    path = tmp_path / "shared.jsonl"

    for reply in ("First run.", "Second run.", "Third run."):
        run_agent(
            f"Task: {reply}",
            max_cycles=1,
            client=FakeClient([reply]),
            on_event=JsonlLogger(path),
        )

    on_disk = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    logs = parse_all(path)

    assert len(logs) == 3
    assert sum(len(log.events) for log in logs) == len(on_disk)

    # Not just the count - the same events. A split that dropped one event and
    # duplicated another would pass a count check.
    def canonical(events):
        return sorted(json.dumps(event, sort_keys=True) for event in events)

    split = [event for log in logs for event in log.events]
    assert canonical(split) == canonical(on_disk)

    # And every event in a run really carries that run's id.
    for log in logs:
        assert {event["run_id"] for event in log.events} == {log.run_id}


def test_parse_refuses_a_file_holding_more_than_one_run(tmp_path):
    """Loud beats wrong. This is the behaviour that replaced the silent merge.

    parse() could have kept merging and simply been documented as one-run-only.
    It does not, because the failure it used to produce was a plausible-looking
    run that never happened, reported with no error - and a caller who has not
    read the docstring gets the wrong answer either way. Raising is the only
    version where being wrong is visible.
    """
    path = tmp_path / "shared.jsonl"
    for reply in ("First run.", "Second run."):
        run_agent(
            f"Task: {reply}",
            max_cycles=1,
            client=FakeClient([reply]),
            on_event=JsonlLogger(path),
        )

    with pytest.raises(ValueError) as exc_info:
        parse(path)

    # Names the count and points at the way out. An error saying only "bad
    # input" would leave the caller where the merge did.
    assert "2 runs" in str(exc_info.value)
    assert "parse_all" in str(exc_info.value)


def test_a_single_run_file_still_reads_the_same_way(tmp_path):
    """parse() keeps its one-run contract, so nothing that already worked moves.

    parse_all is an addition, not a replacement. Every scenario above still
    calls parse() on a single-run file and must keep getting the same answer.
    """
    client = FakeClient([f"Done.\n{COMPLETION_SENTINEL}"])
    result, log, path = _run(tmp_path, task="Say something.", client=client)

    only = parse_all(path)
    assert len(only) == 1
    assert only[0].run_id == log.run_id
    assert only[0].stop_reason == log.stop_reason == result.stop_reason.value
    assert only[0].conversation == log.conversation == result.conversation


# ---------------------------------------------------------------------------
# 7, Day 10: api_stop_reason counted per run, and printed.
#
# A 29 Sep workflow run's log recorded "max_tokens" on 4 of 6 replies and no
# reader surfaced it. Counts above 1 on purpose, so a tally that kept only the
# last reply cannot pass.
# ---------------------------------------------------------------------------


def _spiral_then_cut_off():
    return FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"sides": 6})],
            "Still writing.",
            "Still writing.",
            FakeResponse("and the blockers are", stop_reason="max_tokens"),
        ]
    )


def test_api_stop_reasons_are_counted_per_run(tmp_path):
    path = tmp_path / "run.jsonl"
    run_agent(
        "Write it up.",
        tools=[FakeTool()],
        client=_spiral_then_cut_off(),
        on_event=JsonlLogger(path),
    )
    # A second run in the same file, the shape production writes. Its tally
    # must not include the first run's replies.
    run_agent("Say done.", client=FakeClient(["done\nTASK_COMPLETE"]), on_event=JsonlLogger(path))

    first, second = parse_all(path)

    assert first.api_stop_reasons == {"tool_use": 1, "end_turn": 2, "max_tokens": 1}
    assert second.api_stop_reasons == {"end_turn": 1}
    # No document request in a bare loop run.
    assert first.document_stop_reasons == {}


def test_the_tally_is_printed_beside_the_tokens(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.jsonl"
    run_agent(
        "Write it up.",
        tools=[FakeTool()],
        client=_spiral_then_cut_off(),
        on_event=JsonlLogger(path),
    )
    monkeypatch.setattr("sys.argv", ["read_log.py", str(path)])

    assert read_log.main() == 0

    lines = capsys.readouterr().out.splitlines()
    tokens_at = next(i for i, line in enumerate(lines) if line.startswith("tokens:"))
    assert lines[tokens_at + 1] == (
        "api stops:    tool_use=1, end_turn=2, max_tokens=1 | document: none"
    )
    assert "stop reason:  truncated" in lines


def test_a_tool_failure_is_printed(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.jsonl"
    run_agent(
        "Roll.",
        tools=[FakeTool(name="roll_dice", run=_raise_boom)],
        client=FakeClient([[FakeToolUseBlock("roll_dice", {})]]),
        on_event=JsonlLogger(path),
    )
    monkeypatch.setattr("sys.argv", ["read_log.py", str(path)])

    read_log.main()

    out = capsys.readouterr().out
    assert "tool failed:  roll_dice - RuntimeError: the tool exploded" in out
    assert "stop reason:  tool_failed" in out
