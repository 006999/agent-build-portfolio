"""The agent loop. Day 5: an audit trail. No retry or recovery logic of its own.

Corrected Day 10. This line used to read "No retries, no recovery", and that
was false one layer down. The loop writes no retry, but the Anthropic SDK client
it calls does: in the installed version (anthropic 0.120.1) the default is
max_retries=2, and connection errors, timeouts, 408, 409, 429 and 5xx responses
are retried with backoff before this loop ever sees an exception. Nothing here
changes that and nothing logs it, so one request event followed by one reply
event may stand for up to three HTTP requests. Retry policy is Day 15.

The loop is handed a list of tools and never learns what any of them do. It
reads a name off a tool_use block, finds it in a dict, and calls it. There is
no import from src/tools.py here, and there must never be one.

Day 3 adds the same arrangement for guardrails: the loop is handed lists of
them, runs them in order, and stops at the first refusal. It never asks what a
guardrail checks. There is no import from src/guardrails.py here either.

A guardrail that fires stops the run. It does not retry or rephrase - recovery
is Day 15.

Day 5 adds the same arrangement again for logging: the loop is handed one
callback and calls it. It does not know whether anything is listening, does not
open a file, and has no way to read one back. `src.audit` is imported for the
default no-op only; the loop never constructs a logger. That is what keeps a log
a log and not memory (Day 4) - the agent cannot read what it wrote, because it
did not write it.

The error event logs and RE-RAISES. Behaviour is unchanged: the caller still
gets a crash and no result object. Recording a failure is not handling one, and
handling is Day 10.

Day 10 stops the run on a reply the loop cannot act on. A reply whose
api_stop_reason is "max_tokens" was cut off, so it cannot carry the sentinel
and nudging it only buys another cut-off reply - measured twice, 21 Sep and
29 Sep, in the private build's run notes. A "refusal" is the same class. Both end the run on the
first occurrence with their own StopReason. Stopping is the whole of it: the
loop does not retry, rephrase or raise MAX_TOKENS.
"""

from dataclasses import dataclass, field
from enum import Enum

import anthropic

from src.audit import ignore as _ignore

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_CYCLES = 5
MAX_TOKENS = 1024

# A limit the caller can switch off is a default, not a guardrail. This catches
# an order-of-magnitude mistake - 50 typed for 5, a config value that arrived as
# a string - and nothing subtler. The number has two bounds readable off this
# repo: above the largest legitimate use anything here asks for, and below the
# smallest plausible typo (50, one extra zero on the default).
#
# Re-derived on Day 7, because the lower bound moved and the comment did not.
# Day 3 wrote it against a largest legitimate use of 6 (tests/test_integration.py)
# and a window of 7-49. Day 6 shipped a workflow that runs with max_cycles=16
# (in the private build, not in this copy), which is now the largest legitimate
# use, so the window
# is 17-49. The value is unchanged and has never been raised - what changed is
# the headroom above real usage, from 14 cycles to 4. A workflow one step more
# involved than Day 6's reaches this ceiling, and when it does, the answer is to
# come back to this comment rather than to edit the number in passing.
#
# It bounds steps, not spend. Day 2 measured input cost growing roughly with the
# square of the cycle count, so a 20-cycle run is bounded in steps and unbounded
# in money. Day 6 made that concrete and worse: each cycle may call search_web,
# which is its own model request with up to five server-side searches that this
# loop never sees, so one run can reach ~80 searches across two models. The
# honest instrument is a token budget, and the data is already captured -
# response.usage comes back on every call below and has been logged per cycle
# on the reply event since Day 5. Nothing reads it back to enforce a limit,
# which is the actual gap: the measurement exists, the budget does not.
MAX_CYCLES_CEILING = 4 * DEFAULT_MAX_CYCLES


class StopReason(str, Enum):
    """Why the loop stopped. A closed set on purpose.

    Day 16 turns these into categories in a failure list, so new reasons get
    added here as new members rather than as new free-text strings.

    The two BLOCKED members are separate rather than one because they are
    different failures with different owners: BLOCKED_INPUT means someone sent
    rubbish, BLOCKED_OUTPUT means the model produced something we suppressed.

    Day 8 considered and rejected members for the API's own "refusal" and
    "max_tokens" stop reasons, on the grounds that the loop could not stop for
    either. Day 10 reverses that, on the project owner's decision of 2 Oct, because
    the premise was wrong: a loop reply CAN come back "max_tokens", and on
    29 Sep one workflow run did so on 4 of 6 replies. TRUNCATED and REFUSED are the loop's
    own replies. The document request's refusal and truncation are still
    DocumentRefused and DocumentTruncated in the private build's workflow (not
    in this copy), because that request is made after run_agent has returned.
    """

    COMPLETE = "complete"
    MAX_CYCLES = "max_cycles"
    BLOCKED_INPUT = "blocked_input"
    BLOCKED_OUTPUT = "blocked_output"
    # Day 10. A loop reply with api_stop_reason "max_tokens" or "refusal".
    TRUNCATED = "truncated"
    REFUSED = "refused"
    # Day 10. Two members, not one, for the same reason as BLOCKED: different
    # failures with different owners. TOOL_FAILED means a registered tool raised
    # - the tool, the service behind it, or arguments it could not take.
    # UNKNOWN_TOOL means the model asked for a tool nobody registered - the
    # prompt or the registry.
    TOOL_FAILED = "tool_failed"
    UNKNOWN_TOOL = "unknown_tool"


# Day 10. The api_stop_reasons that end a run, and the reason each one maps to.
# Every other value ("end_turn", "tool_use", anything new) carries on as before.
API_STOP_REASONS = {
    "max_tokens": StopReason.TRUNCATED,
    "refusal": StopReason.REFUSED,
}


@dataclass(frozen=True)
class AgentResult:
    """What one run of the loop hands back.

    `final_answer` holds only text the model actually produced and a guardrail
    allowed, or "" when there is none. Never a string this codebase wrote. A
    blocked run reports itself through blocked_by and block_reason instead, so
    that a report on Day 26 cannot quote a placeholder back as if the agent had
    said it.

    The layering is deliberate: the category is typed (stop_reason), the
    identifier is stable (blocked_by), and only the prose is free text.

    Day 10 repeats it for a tool that failed. failed_tool is the name the model
    asked for; failure_reason is "ExceptionType: message" for TOOL_FAILED, and
    "'name' is not a registered tool" for UNKNOWN_TOOL. Both are None on every
    other stop reason.
    """

    stop_reason: StopReason
    cycles: int
    final_answer: str
    conversation: list[dict] = field(default_factory=list)
    blocked_by: str | None = None
    block_reason: str | None = None
    failed_tool: str | None = None
    failure_reason: str | None = None


COMPLETION_SENTINEL = "TASK_COMPLETE"


def _is_complete(text: str) -> bool:
    """True when the last non-empty line is exactly the sentinel.

    Deliberately not `COMPLETION_SENTINEL in text`. A model explaining the
    protocol back to us would trip that and stop the loop on cycle one.
    """
    lines = text.strip().splitlines()
    if not lines:
        return False
    return lines[-1].strip() == COMPLETION_SENTINEL


def _strip_sentinel(text: str) -> str:
    """Remove the sentinel line so the caller gets a clean final answer."""
    lines = text.strip().splitlines()
    if lines and lines[-1].strip() == COMPLETION_SENTINEL:
        lines = lines[:-1]
    return "\n".join(lines).strip()


# Appended to whatever system prompt the caller supplies. Says nothing about
# the task itself, which is what keeps this loop reusable on Day 6.
COMPLETION_PROTOCOL = (
    f"When the task is fully complete, end your reply with {COMPLETION_SENTINEL} "
    f"on a line by itself. Do not write {COMPLETION_SENTINEL} until the task is "
    "actually finished."
)

# The Messages API needs the conversation to end on a user turn, so something
# has to follow each assistant reply. This is that something.
CONTINUE_NUDGE = (
    "Continue working on the task. When it is fully complete, end your reply "
    f"with {COMPLETION_SENTINEL} on a line by itself."
)


def _text_from(response) -> str:
    """Join the text blocks of a response into one string."""
    parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(parts).strip()


def _tool_use_blocks(response) -> list:
    """Every tool call in one reply. The model may ask for more than one."""
    return [block for block in response.content if block.type == "tool_use"]


def _block_to_dict(block) -> dict:
    """Convert one response block to plain data.

    The conversation is stored as dicts rather than SDK objects so it stays
    printable and JSON-serializable. The API accepts either form.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    raise ValueError(f"Unsupported content block type: {block.type!r}")


def _first_block(guardrails, text: str) -> tuple[str, str] | None:
    """The name and reason of the first guardrail that refuses `text`.

    The only guardrail logic in the loop, and it has no `if` on which guardrail
    it is - it reads a name and calls a function, exactly the way tool dispatch
    does. Later guardrails are not consulted once one refuses: they would only
    produce a second reason nobody reads.
    """
    for guardrail in guardrails:
        verdict = guardrail.check(text)
        if not verdict.allowed:
            return guardrail.name, verdict.reason
    return None


def run_agent(
    task: str,
    *,
    tools=None,
    input_guardrails=None,
    output_guardrails=None,
    system_prompt: str = "",
    max_cycles: int = DEFAULT_MAX_CYCLES,
    model: str = DEFAULT_MODEL,
    client=None,
    on_event=_ignore,
) -> AgentResult:
    """Run the agent until the task is complete, blocked, cut off, or out of cycles.

    The loop does not read the task, the tools, or the guardrails. It sends
    them, runs whatever Claude asks for by name, checks the reply for the
    completion sentinel, and counts.

    `tools` is any sequence of objects with `.name`, `.to_api_format()` and
    `.run(**kwargs)`. A tool that raises ends the run as TOOL_FAILED, and a name
    that is not registered ends it as UNKNOWN_TOOL. Neither error is shown to
    the model (that is Day 13). An exception from the API call itself is still
    logged and re-raised: retry and backoff are Day 15.

    `input_guardrails` and `output_guardrails` are any sequence of objects with
    `.name` and `.check(text)` returning something with `.allowed` and
    `.reason`. Two lists rather than one because the stage is the caller's
    decision, not something a guardrail declares about itself.

    `on_event` is called as `on_event(event_type, **fields)` and its return value
    is ignored. Every event carries `actor`, which is "agent" or "guardrail" and
    nothing else. A tool is not a third actor - the agent called it.
    """
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")
    if max_cycles > MAX_CYCLES_CEILING:
        raise ValueError(f"max_cycles must be at most {MAX_CYCLES_CEILING}")

    # After the two raises above and before the input guardrails below. Those
    # raises are argument validation - a programmer error, not a run that
    # failed - so a bad max_cycles produces an empty log, which is correct:
    # no run started.
    #
    # This is the one call site Day 1 did not anticipate. The before-request
    # seam is inside the loop, and an input-blocked run returns before the loop
    # is entered, so without this the log would record that a task was refused
    # and never record which task.
    on_event(
        "run_start",
        actor="agent",
        task=task,
        system_prompt=system_prompt,
        tool_names=[tool.name for tool in (tools or [])],
        model=model,
        max_cycles=max_cycles,
    )

    input_guardrails = input_guardrails or []
    output_guardrails = output_guardrails or []

    # Before the client is built and before any request. A blocked task costs
    # nothing, and "zero API calls" is asserted in the tests rather than assumed.
    #
    # Its own try/except rather than a widening of the one around the loop body,
    # for two reasons. A guardrail that RAISES is a different event from one that
    # refuses, and this one happens above the loop where nothing else was
    # recording - the log used to end at run_start with no explanation. And kept
    # narrow so that deleting it breaks exactly one test: a wider try would be
    # killed by every mutation touching this path and would prove nothing about
    # this one.
    try:
        blocked = _first_block(input_guardrails, task)
    except Exception as exc:
        # cycle=0 for the same reason the blocked and run_end events below use
        # it - no cycle has started. It also keeps the crash tests' cross-check
        # (the error's cycle equals the number of requests actually made) true
        # at zero, rather than needing a special case for this path.
        on_event(
            "error",
            actor="agent",
            cycle=0,
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise

    if blocked is not None:
        name, reason = blocked
        on_event(
            "blocked",
            actor="guardrail",
            stage="input",
            guardrail=name,
            reason=reason,
            cycle=0,
        )
        # Zero cycles is the correct number, not a hole. Nothing was sent.
        on_event(
            "run_end",
            actor="agent",
            stop_reason=StopReason.BLOCKED_INPUT.value,
            cycles=0,
            final_answer="",
            conversation_length=0,
        )
        return AgentResult(
            stop_reason=StopReason.BLOCKED_INPUT,
            cycles=0,
            # Nothing was sent, so the model said nothing and there is no
            # conversation to report.
            final_answer="",
            conversation=[],
            blocked_by=name,
            block_reason=reason,
        )

    if client is None:
        client = anthropic.Anthropic()

    tools = tools or []
    registry = {tool.name: tool for tool in tools}
    api_tools = [tool.to_api_format() for tool in tools]

    system = f"{system_prompt}\n\n{COMPLETION_PROTOCOL}".strip()
    messages: list[dict] = [{"role": "user", "content": task}]

    # Default to MAX_CYCLES: if we fall out of the loop, that is why we stopped.
    stop_reason = StopReason.MAX_CYCLES
    final_answer = ""
    cycles = 0

    for cycle in range(1, max_cycles + 1):
        cycles = cycle

        request = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        # Only sent when there are tools, so a run without them is byte-for-byte
        # the Day 1 request.
        if api_tools:
            request["tools"] = api_tools

        # Seam 1, Day 1. Fires BEFORE the API call and never after. In a run
        # that succeeds, both orderings write an identical log - only a crash
        # inside client.messages.create can tell them apart, which is why
        # tests/test_logging.py scripts one and scripts/mutate.py mutates it.
        on_event(
            "request",
            actor="agent",
            cycle=cycle,
            model=model,
            message_count=len(messages),
        )

        # Wraps the loop body so `cycles` is accurate when the event is written.
        # An input guardrail that raises happens above this and is not covered -
        # a stated boundary, not a fix.
        try:
            response = client.messages.create(**request)

            # The whole reply is kept, blocks and all. Day 1 flattened it to a
            # string, which would throw away the tool calls.
            content = [_block_to_dict(block) for block in response.content]
            messages.append({"role": "assistant", "content": content})

            reply = _text_from(response)

            # Seam 2, Day 1, which already asked for token usage here. Per cycle
            # rather than a run total: Day 2 measured 86% of input tokens as tool
            # schema, and a total cannot show that.
            #
            # `content` is the authoritative field. It is the exact block list,
            # so the conversation can be rebuilt from the log rather than
            # approximated - a reply with two text blocks would not survive
            # being flattened to one string and rebuilt.
            on_event(
                "reply",
                actor="agent",
                cycle=cycle,
                api_stop_reason=response.stop_reason,
                content=content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            # Checked here, before three things, and the order is the design.
        #
            # Before `final_answer = reply` below, which is the whole mechanism
            # that keeps blocked text out of the returned answer without the loop
            # writing a placeholder. Before the tool branch, so a reply that trips
            # the filter cannot also run a tool. And before the sentinel, so a
            # reply that trips the filter and claims completion is blocked rather
            # than complete.
            #
            # Carrying on is not an option: the offending text is already in
            # `messages` and would be re-sent to the model on every later cycle.
            #
            # Guardrails are called even when `reply` is "" - a cycle that is
            # nothing but a tool call. Skipping those would be the loop deciding
            # something about what a guardrail means, and the loop decides nothing.
            blocked = _first_block(output_guardrails, reply)
            if blocked is not None:
                name, reason = blocked
                on_event(
                    "blocked",
                    actor="guardrail",
                    stage="output",
                    guardrail=name,
                    reason=reason,
                    cycle=cycles,
                )
                # A real API call happened before this fired, so unlike an input
                # block this run has a request and a reply in its log.
                on_event(
                    "run_end",
                    actor="agent",
                    stop_reason=StopReason.BLOCKED_OUTPUT.value,
                    cycles=cycles,
                    final_answer=final_answer,
                    conversation_length=len(messages),
                )
                return AgentResult(
                    stop_reason=StopReason.BLOCKED_OUTPUT,
                    cycles=cycles,
                    # The last reply that passed, or "" if none did. Never the
                    # blocked text, and never prose we wrote.
                    final_answer=final_answer,
                    # The offending turn stays in the transcript on purpose: Day 5
                    # logs it and Day 17 reads it. Suppressing it here would leave
                    # no record of what was blocked.
                    conversation=messages,
                    blocked_by=name,
                    block_reason=reason,
                )

            # Day 10. A reply cut off at MAX_TOKENS, or a refusal, ends the run on
            # the first occurrence. The position is the design, same as above:
            #
            # After the reply event, so the evidence - the reply and its
            # api_stop_reason - is in the log before the run ends.
            #
            # After the output guardrail, so the Day 3 claim that it runs on
            # every reply stays true. A reply that is both truncated and trips
            # the filter is BLOCKED_OUTPUT: the safety finding is the one with
            # an owner who has to act.
            #
            # Before `final_answer = reply`, so cut-off text never becomes the
            # answer - same rule as a blocked reply. Before tool dispatch, so a
            # tool_use block cut off at the cap is never run. Before the
            # sentinel, so a truncated reply cannot claim completion.
            stopped_by_api = API_STOP_REASONS.get(response.stop_reason)
            if stopped_by_api is not None:
                # No completion_check event: no check happened on this cycle.
                on_event(
                    "run_end",
                    actor="agent",
                    stop_reason=stopped_by_api.value,
                    cycles=cycles,
                    final_answer=final_answer,
                    conversation_length=len(messages),
                )
                return AgentResult(
                    stop_reason=stopped_by_api,
                    cycles=cycles,
                    # The last reply that passed, or "". The cut-off reply stays
                    # in the conversation, like a blocked one, so it is on record.
                    final_answer=final_answer,
                    conversation=messages,
                )

            # Only overwrite when the model actually said something. A cycle that
            # is nothing but a tool call would otherwise wipe the last real answer
            # and leave final_answer empty if the run then ran out of cycles.
            if reply:
                final_answer = reply

            requested_tools = _tool_use_blocks(response)

            if requested_tools:
                # Checked before the sentinel on purpose. A reply that asks for
                # data and claims completion at the same time cannot know the task
                # is done: it has not seen the answer yet.
                results = []
                for block in requested_tools:
                    # The only dispatch in the loop. No branching on which tool.
                    tool = registry.get(block.name)

                    # Day 10. The model asked for a tool nobody registered. The
                    # run stops before anything is called for this block. Tools
                    # earlier in the same reply have already run, in order, as
                    # they always did.
                    if tool is None:
                        failure_reason = f"{block.name!r} is not a registered tool"
                        on_event(
                            "tool_unknown",
                            actor="agent",
                            cycle=cycle,
                            tool_use_id=block.id,
                            name=block.name,
                            args=block.input,
                            registered=list(registry),
                        )
                        on_event(
                            "run_end",
                            actor="agent",
                            stop_reason=StopReason.UNKNOWN_TOOL.value,
                            cycles=cycles,
                            final_answer=final_answer,
                            conversation_length=len(messages),
                        )
                        return AgentResult(
                            stop_reason=StopReason.UNKNOWN_TOOL,
                            cycles=cycles,
                            final_answer=final_answer,
                            conversation=messages,
                            failed_tool=block.name,
                            failure_reason=failure_reason,
                        )

                    # Day 10. Narrow on purpose: only the tool's own call. An
                    # exception from the API call, or from a guardrail, is not a
                    # tool failure and still reaches the except at the bottom.
                    #
                    # The error is not sent to the model - that is Day 13. And
                    # results already collected this cycle are not appended to
                    # the conversation: they were never sent, so they are not
                    # part of what the model saw. They are in the log, each as
                    # its own tool_result event above this one.
                    try:
                        result = tool.run(**block.input)
                    except Exception as exc:
                        failure_reason = f"{type(exc).__name__}: {exc}"
                        on_event(
                            "tool_failed",
                            actor="agent",
                            cycle=cycle,
                            tool_use_id=block.id,
                            name=block.name,
                            args=block.input,
                            exception_type=type(exc).__name__,
                            message=str(exc),
                        )
                        on_event(
                            "run_end",
                            actor="agent",
                            stop_reason=StopReason.TOOL_FAILED.value,
                            cycles=cycles,
                            final_answer=final_answer,
                            conversation_length=len(messages),
                        )
                        return AgentResult(
                            stop_reason=StopReason.TOOL_FAILED,
                            cycles=cycles,
                            final_answer=final_answer,
                            conversation=messages,
                            failed_tool=block.name,
                            failure_reason=failure_reason,
                        )
                    # Logged here because this is the only place the result
                    # exists. None of Day 1's four seams can see it: it is a
                    # return value on the line above, and the next seam that
                    # could carry it is the following cycle's request event -
                    # which never comes on the last cycle of a max-cycles run.
                    on_event(
                        "tool_result",
                        actor="agent",
                        cycle=cycle,
                        tool_use_id=block.id,
                        name=block.name,
                        args=block.input,
                        result=result,
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

                # Every tool_use block needs its result in the very next message,
                # so all of them go in one user turn. That turn is what keeps the
                # conversation ending on the user, so no nudge is appended.
                messages.append({"role": "user", "content": results})
                continue

            # Seam 3, Day 1. A tool cycle never reaches this, because the branch
            # above continues - and that is correct rather than a gap: on a tool
            # cycle the completion check genuinely does not happen.
            matched = _is_complete(reply)
            on_event(
                "completion_check",
                actor="agent",
                cycle=cycle,
                sentinel_matched=matched,
                # The nudge text itself, not a flag. It becomes part of the next
                # prompt, and "log every prompt" means the log carries it rather
                # than expecting a reader to already know the constant.
                nudge=None if matched else CONTINUE_NUDGE,
            )

            if matched:
                final_answer = _strip_sentinel(reply)
                stop_reason = StopReason.COMPLETE
                break

            messages.append({"role": "user", "content": CONTINUE_NUDGE})

        except Exception as exc:
            # Logs and re-raises. The caller still gets a crash and no result
            # object, so behaviour is identical to Day 1 - recording a failure is
            # not handling one.
            #
            # Day 10 left this path as it was, on the project owner's decision. What still
            # arrives here: an exception from client.messages.create, from an
            # output guardrail, or from _block_to_dict on a block type the loop
            # does not know. A tool's own exception no longer does - see the
            # dispatch above. This is the one way left for a run to end without
            # a run_end, and that absence is how the log says the run died.
            # Retry and backoff are Day 15.
            #
            # BaseException is deliberately not caught: a KeyboardInterrupt is
            # not a run that failed.
            on_event(
                "error",
                actor="agent",
                cycle=cycle,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            raise

    # Seam 4, Day 1. The normal exit. The others are the returns above -
    # blocked, and Day 10's truncated, refused, tool_failed and unknown_tool -
    # and each writes its own run_end. Every run that returns writes
    # exactly one; a run that raises writes none, which is how the log records
    # that it never finished.
    on_event(
        "run_end",
        actor="agent",
        stop_reason=stop_reason.value,
        cycles=cycles,
        final_answer=final_answer,
        conversation_length=len(messages),
    )

    return AgentResult(
        stop_reason=stop_reason,
        cycles=cycles,
        final_answer=final_answer,
        conversation=messages,
    )
