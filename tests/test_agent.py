import os

import pytest

# Hand-written stand-ins for the client, a tool and a guardrail. They lived in
# this file until Day 4, when tests/test_memory.py needed the same ones.
from fakes import (
    FakeClient,
    FakeGuardrail,
    FakeResponse,
    FakeTextBlock,
    FakeTool,
    FakeToolUseBlock,
)

from src.agent import (
    COMPLETION_SENTINEL,
    CONTINUE_NUDGE,
    MAX_CYCLES_CEILING,
    AgentResult,
    StopReason,
    _is_complete,
    _strip_sentinel,
    run_agent,
)


def _blocks_of_type(conversation: list[dict], block_type: str) -> list[dict]:
    """Every block of one type across the whole conversation."""
    found = []
    for message in conversation:
        if isinstance(message["content"], list):
            found.extend(
                block for block in message["content"] if block["type"] == block_type
            )
    return found


def test_stop_reason_is_a_fixed_set_not_free_text():
    # Day 16 turns this into a failure category, so it must be an enum. Growing
    # this set is meant to be a deliberate act - if this test went red without
    # someone intending it, a reason was added by accident.
    assert {member.name for member in StopReason} == {
        "COMPLETE",
        "MAX_CYCLES",
        "BLOCKED_INPUT",
        "BLOCKED_OUTPUT",
        # Day 10, deliberately: a loop reply cut off at max_tokens, or refused.
        "TRUNCATED",
        "REFUSED",
        # Day 10, deliberately: a tool that raised, and a tool nobody registered.
        "TOOL_FAILED",
        "UNKNOWN_TOOL",
    }


def test_result_carries_all_four_facts():
    result = AgentResult(
        stop_reason=StopReason.COMPLETE,
        cycles=2,
        final_answer="done",
        conversation=[{"role": "user", "content": "hi"}],
    )
    assert result.stop_reason is StopReason.COMPLETE
    assert result.cycles == 2
    assert result.final_answer == "done"
    assert result.conversation == [{"role": "user", "content": "hi"}]


def test_sentinel_alone_on_the_last_line_means_complete():
    assert _is_complete("Here is the answer.\nTASK_COMPLETE") is True


def test_sentinel_mentioned_in_prose_does_not_mean_complete():
    # The bug this design invites: a `in reply` check stops on cycle 1 here.
    reply = f"I will write {COMPLETION_SENTINEL} once I have actually finished."
    assert _is_complete(reply) is False


def test_no_sentinel_means_not_complete():
    assert _is_complete("Still working on it.") is False


def test_empty_reply_is_not_complete():
    assert _is_complete("   ") is False


def test_stripping_removes_the_sentinel_line():
    assert _strip_sentinel("Here is the answer.\nTASK_COMPLETE") == "Here is the answer."


def test_stripping_leaves_text_alone_when_there_is_no_sentinel():
    assert _strip_sentinel("Still working.") == "Still working."


def test_stops_at_max_cycles_when_the_model_never_signals_complete():
    client = FakeClient()  # every reply is "Still working on it."
    result = run_agent("anything", max_cycles=3, client=client)

    assert result.stop_reason is StopReason.MAX_CYCLES
    assert result.cycles == 3
    # The call count is what catches an off-by-one in the range().
    assert len(client.messages.calls) == 3


def test_stops_as_soon_as_the_model_signals_complete():
    client = FakeClient(["Thinking...", "Here it is.\nTASK_COMPLETE", "never reached"])
    result = run_agent("anything", max_cycles=5, client=client)

    assert result.stop_reason is StopReason.COMPLETE
    assert result.cycles == 2
    assert len(client.messages.calls) == 2
    assert result.final_answer == "Here it is."  # sentinel stripped


def test_conversation_holds_every_turn():
    client = FakeClient(["Thinking...", "Here it is.\nTASK_COMPLETE"])
    result = run_agent("do the thing", max_cycles=5, client=client)

    roles = [message["role"] for message in result.conversation]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert result.conversation[0]["content"] == "do the thing"


def test_the_loop_never_rewrites_the_task():
    # Day 6 swaps in a real workflow. The loop must pass it through untouched.
    task = "Reconcile the invoice ledger for Q3."
    client = FakeClient(["done\nTASK_COMPLETE"])
    result = run_agent(task, client=client)

    assert client.messages.calls[0]["messages"][0]["content"] == task
    assert result.conversation[0]["content"] == task


def test_max_cycles_below_one_is_rejected():
    with pytest.raises(ValueError):
        run_agent("anything", max_cycles=0, client=FakeClient())


# ---------------------------------------------------------------------------
# Day 2: tool use. Every test below uses a made-up tool. The loop must not be
# able to tell the difference between `roll_dice` and a real client workflow.
# ---------------------------------------------------------------------------


def test_a_tool_use_reply_runs_the_named_tool_with_the_input_claude_sent():
    tool = FakeTool("roll_dice", result="rolled a 4")
    client = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"sides": 20})],
            "You rolled a 4.\nTASK_COMPLETE",
        ]
    )

    run_agent("roll something", tools=[tool], client=client)

    assert tool.calls == [{"sides": 20}]


def test_the_result_comes_back_with_the_id_that_asked_for_it():
    tool = FakeTool("roll_dice", result="rolled a 4")
    client = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"sides": 20}, block_id="toolu_ABC")],
            "done\nTASK_COMPLETE",
        ]
    )

    result = run_agent("roll something", tools=[tool], client=client)

    results = _blocks_of_type(result.conversation, "tool_result")
    assert len(results) == 1
    # Mismatch the ids and the API rejects the next request outright.
    assert results[0]["tool_use_id"] == "toolu_ABC"
    assert results[0]["content"] == "rolled a 4"


def test_two_tool_calls_in_one_reply_produce_one_message_with_two_results():
    dice = FakeTool("roll_dice", result="rolled a 4")
    coin = FakeTool("flip_coin", result="heads")
    client = FakeClient(
        [
            [
                FakeToolUseBlock("roll_dice", {"sides": 6}, block_id="toolu_1"),
                FakeToolUseBlock("flip_coin", {}, block_id="toolu_2"),
            ],
            "done\nTASK_COMPLETE",
        ]
    )

    result = run_agent("do both", tools=[dice, coin], client=client)

    # Both results in ONE user turn. Splitting them across two messages, or
    # answering only the first, is an API error on the very next request.
    user_turns_with_results = [
        message
        for message in result.conversation
        if message["role"] == "user" and isinstance(message["content"], list)
    ]
    assert len(user_turns_with_results) == 1
    assert len(user_turns_with_results[0]["content"]) == 2

    ids = [block["tool_use_id"] for block in user_turns_with_results[0]["content"]]
    assert ids == ["toolu_1", "toolu_2"]


def test_a_reply_with_both_a_tool_call_and_the_sentinel_keeps_going():
    """The model cannot know it is done - it has not seen the answer yet."""
    tool = FakeTool("roll_dice", result="rolled a 4")
    client = FakeClient(
        [
            [
                FakeTextBlock("I have finished.\nTASK_COMPLETE"),
                FakeToolUseBlock("roll_dice", {"sides": 6}),
            ],
            "Actually it was a 4.\nTASK_COMPLETE",
        ]
    )

    result = run_agent("roll something", tools=[tool], client=client)

    assert tool.calls == [{"sides": 6}]
    assert result.cycles == 2
    assert result.final_answer == "Actually it was a 4."


def test_the_loop_dispatches_a_tool_it_has_never_heard_of():
    """The domain-agnostic rule, made executable.

    Day 6 swaps in a real client workflow. If this test ever needs changing to
    accommodate a new tool, the loop has grown knowledge it should not have.
    """
    tool = FakeTool("wibble", result="wibbled")
    client = FakeClient(
        [[FakeToolUseBlock("wibble", {"how_much": "lots"})], "ok\nTASK_COMPLETE"]
    )

    result = run_agent("wibble it", tools=[tool], client=client)

    assert tool.calls == [{"how_much": "lots"}]
    assert result.stop_reason is StopReason.COMPLETE


def test_no_continue_nudge_is_added_on_a_cycle_that_produced_tool_results():
    # The tool_result message is already a user turn. Adding the nudge as well
    # would send the model a second, contradictory instruction every cycle.
    tool = FakeTool()
    client = FakeClient(
        [[FakeToolUseBlock("roll_dice", {})], "done\nTASK_COMPLETE"]
    )

    result = run_agent("roll", tools=[tool], client=client)

    text_contents = [
        message["content"]
        for message in result.conversation
        if isinstance(message["content"], str)
    ]
    assert CONTINUE_NUDGE not in text_contents


def test_the_nudge_still_appears_when_no_tool_was_called():
    tool = FakeTool()
    client = FakeClient(["Still thinking.", "done\nTASK_COMPLETE"])

    result = run_agent("think", tools=[tool], client=client)

    text_contents = [
        message["content"]
        for message in result.conversation
        if isinstance(message["content"], str)
    ]
    assert CONTINUE_NUDGE in text_contents


def test_without_tools_the_request_carries_no_tools_key():
    client = FakeClient(["done\nTASK_COMPLETE"])

    run_agent("anything", client=client)

    assert "tools" not in client.messages.calls[0]


def test_with_tools_the_schemas_go_in_every_request():
    """The positive control for the two negative tests.

    Without this, a run that never wired the schemas in at all would pass every
    "it did not call a tool" test, brilliantly and for the wrong reason.
    """
    tool = FakeTool("roll_dice")
    client = FakeClient(["Still thinking.", "done\nTASK_COMPLETE"])

    run_agent("anything", tools=[tool], client=client)

    assert len(client.messages.calls) == 2
    for call in client.messages.calls:
        assert [schema["name"] for schema in call["tools"]] == ["roll_dice"]


def test_a_tool_is_not_called_when_the_model_does_not_ask_for_one():
    tool = FakeTool("roll_dice")
    client = FakeClient(["No dice needed here.\nTASK_COMPLETE"])

    result = run_agent("explain what a dice is", tools=[tool], client=client)

    assert tool.calls == []
    assert _blocks_of_type(result.conversation, "tool_result") == []
    assert result.stop_reason is StopReason.COMPLETE


def test_tool_calls_cost_one_cycle_each_and_no_more():
    tool = FakeTool()
    client = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {}, block_id="toolu_1")],
            [FakeToolUseBlock("roll_dice", {}, block_id="toolu_2")],
            "done\nTASK_COMPLETE",
        ]
    )

    result = run_agent("roll twice", tools=[tool], client=client)

    assert result.cycles == 3
    assert len(client.messages.calls) == 3


# --- The three ways a tool call goes wrong. ---------------------------------
#
# Days 2-9 let all three crash the run, on purpose, and these tests pinned that.
# Day 10 reverses it on the project owner's decision of 2 Oct: each one now ends the run
# with a typed reason and a run_end. The tests are rewritten, not deleted, so
# the three cases stay visibly separate.


def test_an_unknown_tool_name_stops_the_run_as_unknown_tool():
    # The model invented a capability. Fix lives in the prompt or the registry.
    client = FakeClient([[FakeToolUseBlock("send_email", {"to": "a@b.c"})]])

    result = run_agent("anything", tools=[FakeTool("roll_dice")], client=client)

    assert result.stop_reason is StopReason.UNKNOWN_TOOL
    assert result.failed_tool == "send_email"
    assert result.failure_reason == "'send_email' is not a registered tool"
    assert len(client.messages.calls) == 1


def test_the_right_tool_with_a_wrong_argument_stops_the_run_as_tool_failed():
    # The model read the schema wrong. Fix lives in input_schema. It surfaces
    # as the tool raising TypeError, so it is a tool failure, not a new kind.
    def strict_run(sides):
        return f"rolled with {sides}"

    tool = FakeTool("roll_dice", run=strict_run)
    client = FakeClient([[FakeToolUseBlock("roll_dice", {"sides": 6, "colour": "red"})]])

    result = run_agent("anything", tools=[tool], client=client)

    assert result.stop_reason is StopReason.TOOL_FAILED
    assert result.failed_tool == "roll_dice"
    assert result.failure_reason.startswith("TypeError: ")


def test_a_tool_that_raises_stops_the_run_as_tool_failed():
    def explode(**kwargs):
        raise RuntimeError("the invoice file is not there")

    tool = FakeTool("roll_dice", run=explode)
    client = FakeClient([[FakeToolUseBlock("roll_dice", {})]])

    result = run_agent("anything", tools=[tool], client=client)

    assert result.stop_reason is StopReason.TOOL_FAILED
    assert result.cycles == 1
    assert result.failed_tool == "roll_dice"
    assert result.failure_reason == "RuntimeError: the invoice file is not there"
    assert result.blocked_by is None


def test_a_tool_failure_is_not_fed_back_to_the_model():
    """Rule 2: the error does not go back to the model. That is Day 13.

    One API call, and no tool_result block anywhere in the conversation - the
    failed call has no result, and nothing was sent after it.
    """

    def explode(**kwargs):
        raise RuntimeError("upstream timed out")

    client = FakeClient([[FakeToolUseBlock("roll_dice", {})], "done\nTASK_COMPLETE"])

    result = run_agent("roll", tools=[FakeTool("roll_dice", run=explode)], client=client)

    assert len(client.messages.calls) == 1
    assert _blocks_of_type(result.conversation, "tool_result") == []


def test_a_tool_that_succeeds_is_fed_back_on_the_next_call():
    """Positive control for the test above. Same script, a tool that works."""
    client = FakeClient([[FakeToolUseBlock("roll_dice", {})], "done\nTASK_COMPLETE"])

    result = run_agent("roll", tools=[FakeTool("roll_dice")], client=client)

    assert len(client.messages.calls) == 2
    second_request = client.messages.calls[1]["messages"]
    assert second_request[-1]["content"][0]["type"] == "tool_result"
    assert result.failed_tool is None
    assert result.failure_reason is None


# --- final_answer stays a clean string on every path -----------------------


def test_final_answer_is_a_string_even_when_the_run_dies_mid_tool_call():
    tool = FakeTool()
    client = FakeClient([[FakeToolUseBlock("roll_dice", {})]])  # then repeats

    result = run_agent("roll forever", tools=[tool], max_cycles=2, client=client)

    assert result.stop_reason is StopReason.MAX_CYCLES
    assert isinstance(result.final_answer, str)


def test_a_silent_tool_call_does_not_wipe_the_last_thing_the_model_said():
    """The quiet bug.

    Models routinely emit a tool_use block with no text beside it. Assigning
    final_answer from every reply returns "" here, which reads as "the agent
    had nothing to say" rather than "it was mid-reach when cycles ran out".
    """
    tool = FakeTool()
    client = FakeClient(
        [
            [
                FakeTextBlock("Let me look that invoice up."),
                FakeToolUseBlock("roll_dice", {}, block_id="toolu_1"),
            ],
            [FakeToolUseBlock("roll_dice", {}, block_id="toolu_2")],  # no text
        ]
    )

    result = run_agent("look it up", tools=[tool], max_cycles=2, client=client)

    assert result.stop_reason is StopReason.MAX_CYCLES
    assert result.final_answer == "Let me look that invoice up."


def test_the_conversation_stays_plain_data():
    # Day 5 logging and Day 12 resume both read this. SDK objects would not
    # survive a round trip through a file.
    import json

    tool = FakeTool()
    client = FakeClient(
        [[FakeToolUseBlock("roll_dice", {"sides": 6})], "done\nTASK_COMPLETE"]
    )

    result = run_agent("roll", tools=[tool], client=client)

    json.dumps(result.conversation)  # raises TypeError if anything is an object


# ---------------------------------------------------------------------------
# Day 3: guardrails. Every test below uses an invented guardrail, for the same
# reason the tool tests use roll_dice. The loop must not be able to tell the
# difference between wibble_guard and a client's compliance rules.
# ---------------------------------------------------------------------------


def test_a_blocked_task_never_reaches_the_api():
    guard = FakeGuardrail(name="no_wibbling", blocks_on="wibble", reason="no wibbling")
    client = FakeClient(["done\nTASK_COMPLETE"])

    result = run_agent("please wibble", input_guardrails=[guard], client=client)

    assert result.stop_reason is StopReason.BLOCKED_INPUT
    assert result.cycles == 0
    assert result.blocked_by == "no_wibbling"
    assert result.block_reason == "no wibbling"
    # The sharp one. A guardrail that runs after the first request has already
    # spent the money it exists to save.
    assert client.messages.calls == []
    assert result.conversation == []
    assert result.final_answer == ""


def test_input_guardrails_stop_at_the_first_refusal():
    first = FakeGuardrail(name="first", blocks_on="bad", reason="first said no")
    second = FakeGuardrail(name="second", blocks_on="bad", reason="second said no")

    result = run_agent("bad task", input_guardrails=[first, second], client=FakeClient())

    assert result.blocked_by == "first"
    assert second.seen == []  # never consulted


def test_an_allowed_task_reaches_the_guardrail_unchanged():
    guard = FakeGuardrail(name="watcher")
    task = "Reconcile the invoice ledger for Q3."

    run_agent(task, input_guardrails=[guard], client=FakeClient(["ok\nTASK_COMPLETE"]))

    assert guard.seen == [task]


def test_a_blocked_reply_stops_the_run():
    guard = FakeGuardrail(name="no_wibbling", blocks_on="wibble", reason="it wibbled")
    client = FakeClient(["that is wibble", "never reached"])

    result = run_agent("anything", output_guardrails=[guard], client=client)

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT
    assert result.cycles == 1
    assert result.blocked_by == "no_wibbling"
    assert result.block_reason == "it wibbled"
    assert len(client.messages.calls) == 1  # stopped rather than carried on


def test_the_offending_reply_stays_in_the_conversation():
    """Day 5 logs this and Day 17 reads it.

    Dropping it here would leave a blocked run with no record of what was
    blocked, which is the one thing you need when a client asks why.
    """
    guard = FakeGuardrail(blocks_on="wibble")
    client = FakeClient(["that is wibble"])

    result = run_agent("anything", output_guardrails=[guard], client=client)

    texts = [block["text"] for block in _blocks_of_type(result.conversation, "text")]
    assert texts == ["that is wibble"]


def test_a_block_on_cycle_one_leaves_final_answer_empty():
    """final_answer holds words the model said and a guardrail allowed.

    On cycle 1 there is no such text, so it is "". Never the blocked text, and
    never a refusal message this codebase wrote - Day 26 reporting would quote
    that back as if the agent had produced it.
    """
    guard = FakeGuardrail(blocks_on="wibble", reason="it wibbled")
    client = FakeClient(["that is wibble"])

    result = run_agent("anything", output_guardrails=[guard], client=client)

    assert result.final_answer == ""


def test_a_block_on_cycle_two_keeps_the_reply_that_passed():
    guard = FakeGuardrail(blocks_on="wibble", reason="it wibbled")
    client = FakeClient(["A clean first answer.", "now it is wibble"])

    result = run_agent("anything", output_guardrails=[guard], client=client)

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT
    assert result.cycles == 2
    assert result.final_answer == "A clean first answer."
    assert "wibble" not in result.final_answer
    assert result.final_answer != result.block_reason


def test_a_block_beats_the_completion_sentinel():
    # Blocked, not complete. The filter does not get overruled by the model
    # announcing it has finished.
    guard = FakeGuardrail(blocks_on="wibble")
    client = FakeClient(["all done with wibble\nTASK_COMPLETE"])

    result = run_agent("anything", output_guardrails=[guard], client=client)

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT


def test_a_block_beats_a_tool_call_and_the_tool_never_runs():
    tool = FakeTool("roll_dice")
    guard = FakeGuardrail(blocks_on="wibble")
    client = FakeClient(
        [[FakeTextBlock("about to wibble"), FakeToolUseBlock("roll_dice", {})]]
    )

    result = run_agent(
        "anything", tools=[tool], output_guardrails=[guard], client=client
    )

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT
    assert tool.calls == []


def test_output_guardrails_see_the_raw_reply_including_a_tool_only_cycle():
    """Two decisions pinned at once.

    The guardrail is handed "" on a cycle that was nothing but a tool call -
    skipping those would be the loop deciding what a guardrail means. And it is
    handed the reply with the sentinel still on it, so a guardrail never has to
    know the completion protocol exists.
    """
    tool = FakeTool()
    guard = FakeGuardrail(name="watcher")
    client = FakeClient([[FakeToolUseBlock("roll_dice", {})], "done\nTASK_COMPLETE"])

    run_agent("roll", tools=[tool], output_guardrails=[guard], client=client)

    assert guard.seen == ["", "done\nTASK_COMPLETE"]


def test_the_loop_runs_a_guardrail_it_has_never_heard_of():
    """The domain-agnostic rule, extended from tools to guardrails.

    Day 6 swaps in a client's own rules. If this test ever needs changing to
    accommodate a real guardrail, the loop has grown knowledge it should not
    have.
    """
    guard = FakeGuardrail(name="wibble_guard", blocks_on="zzz")
    client = FakeClient(["a clean answer\nTASK_COMPLETE"])

    result = run_agent(
        "anything",
        input_guardrails=[guard],
        output_guardrails=[guard],
        client=client,
    )

    assert result.stop_reason is StopReason.COMPLETE
    assert guard.seen == ["anything", "a clean answer\nTASK_COMPLETE"]


def test_an_unblocked_result_carries_no_block_fields():
    result = AgentResult(stop_reason=StopReason.COMPLETE, cycles=1, final_answer="done")

    assert result.blocked_by is None
    assert result.block_reason is None


def test_the_block_fields_are_none_on_every_other_stop_reason():
    complete = run_agent("x", client=FakeClient(["done\nTASK_COMPLETE"]))
    assert complete.stop_reason is StopReason.COMPLETE
    assert complete.blocked_by is None
    assert complete.block_reason is None

    exhausted = run_agent("x", max_cycles=2, client=FakeClient())
    assert exhausted.stop_reason is StopReason.MAX_CYCLES
    assert exhausted.blocked_by is None
    assert exhausted.block_reason is None


def test_max_cycles_at_the_ceiling_is_allowed():
    client = FakeClient(["done\nTASK_COMPLETE"])

    result = run_agent("x", max_cycles=MAX_CYCLES_CEILING, client=client)

    assert result.stop_reason is StopReason.COMPLETE


def test_max_cycles_above_the_ceiling_is_rejected():
    # A limit the caller can switch off is a default, not a guardrail.
    with pytest.raises(ValueError, match="at most"):
        run_agent("x", max_cycles=MAX_CYCLES_CEILING + 1, client=FakeClient())


def test_installing_guardrails_does_not_change_a_clean_run():
    """The false positive that actually costs a client.

    Same replies, same tool, once bare and once with permissive guardrails on
    both sides. Everything observable must match.
    """

    def replies():
        return [[FakeToolUseBlock("roll_dice", {})], "done\nTASK_COMPLETE"]

    bare_client = FakeClient(replies())
    bare = run_agent("roll", tools=[FakeTool()], client=bare_client)

    guarded_client = FakeClient(replies())
    guarded = run_agent(
        "roll",
        tools=[FakeTool()],
        input_guardrails=[FakeGuardrail(name="a")],
        output_guardrails=[FakeGuardrail(name="b")],
        client=guarded_client,
    )

    assert guarded.stop_reason is bare.stop_reason
    assert guarded.cycles == bare.cycles
    assert guarded.final_answer == bare.final_answer
    assert guarded.conversation == bare.conversation
    assert len(guarded_client.messages.calls) == len(bare_client.messages.calls)


# --- Day 10: a reply the loop cannot act on ends the run ---------------------
#
# Each "does not happen" test below has its positive control next to it: the
# same scripted text with an ordinary stop_reason, showing the thing that the
# hard stop prevents really does happen otherwise. Without the control, "one
# API call" is satisfied by a loop that never makes a second call at all.

# The shape of the 29 Sep replies: a document being written out inline and
# stopped mid-sentence by the cap.
CUT_OFF_REPLY = "The blockers are: first, the unresolved 2026-04-21 to"


def test_a_max_tokens_reply_stops_the_run_on_the_first_occurrence():
    client = FakeClient(
        [
            FakeResponse(CUT_OFF_REPLY, stop_reason="max_tokens"),
            "never reached\nTASK_COMPLETE",
        ]
    )

    result = run_agent("research", client=client)

    assert result.stop_reason is StopReason.TRUNCATED
    assert result.cycles == 1
    assert len(client.messages.calls) == 1
    # Not nudged. The nudge is what fed the 21 Sep and 29 Sep spirals.
    assert CONTINUE_NUDGE not in [m["content"] for m in result.conversation]


def test_the_same_reply_ending_normally_is_nudged_and_carries_on():
    """Positive control for the test above. Same text, end_turn."""
    client = FakeClient(
        [
            FakeResponse(CUT_OFF_REPLY, stop_reason="end_turn"),
            "done\nTASK_COMPLETE",
        ]
    )

    result = run_agent("research", client=client)

    assert result.stop_reason is StopReason.COMPLETE
    assert len(client.messages.calls) == 2
    assert CONTINUE_NUDGE in [m["content"] for m in result.conversation]


def test_a_refusal_reply_stops_the_run_on_the_first_occurrence():
    client = FakeClient(
        [
            FakeResponse("I can't help with that.", stop_reason="refusal"),
            "never reached\nTASK_COMPLETE",
        ]
    )

    result = run_agent("research", client=client)

    assert result.stop_reason is StopReason.REFUSED
    assert result.cycles == 1
    assert len(client.messages.calls) == 1
    assert CONTINUE_NUDGE not in [m["content"] for m in result.conversation]


def test_a_truncated_reply_with_a_tool_call_does_not_run_the_tool():
    """A tool_use block cut off at the cap may hold half its arguments."""
    tool = FakeTool("roll_dice")
    client = FakeClient(
        [
            FakeResponse(
                [FakeToolUseBlock("roll_dice", {"sides": 6})], stop_reason="max_tokens"
            )
        ]
    )

    result = run_agent("roll", tools=[tool], client=client)

    assert result.stop_reason is StopReason.TRUNCATED
    assert tool.calls == []


def test_the_same_tool_call_ending_normally_does_run_the_tool():
    """Positive control for the test above."""
    tool = FakeTool("roll_dice")
    client = FakeClient(
        [
            FakeResponse(
                [FakeToolUseBlock("roll_dice", {"sides": 6})], stop_reason="tool_use"
            ),
            "done\nTASK_COMPLETE",
        ]
    )

    run_agent("roll", tools=[tool], client=client)

    assert tool.calls == [{"sides": 6}]


def test_a_truncated_reply_never_becomes_the_final_answer():
    client = FakeClient(
        [
            "A clean first answer.",
            FakeResponse("and then it was cut off mid", stop_reason="max_tokens"),
        ]
    )

    result = run_agent("research", client=client)

    assert result.stop_reason is StopReason.TRUNCATED
    assert result.cycles == 2
    assert result.final_answer == "A clean first answer."
    # Still in the transcript, so the record shows what was cut off.
    texts = [b["text"] for b in _blocks_of_type(result.conversation, "text")]
    assert texts[-1] == "and then it was cut off mid"


def test_a_truncated_reply_that_trips_the_output_guardrail_is_blocked_not_truncated():
    """Pins the order: the output guardrail runs first, on every reply."""
    guard = FakeGuardrail(name="no_wibbling", blocks_on="wibble")
    client = FakeClient(
        [FakeResponse("we shall wibble the bl", stop_reason="max_tokens")]
    )

    result = run_agent("anything", output_guardrails=[guard], client=client)

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT
    assert result.blocked_by == "no_wibbling"
    assert guard.seen == ["we shall wibble the bl"]


@pytest.mark.integration
def test_real_model_hits_max_cycles_on_an_unfinishable_task():
    """Unfinishable by length, not by capability.

    A task the model *cannot do* would get a confident "I can't do that" plus
    the sentinel — green for the wrong reason. A task that is simply too long
    gives the model no reason to claim completion.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = run_agent(
        "Count upward from 1. List at most 10 numbers per reply. "
        "Stop only when you reach 1000000.",
        max_cycles=3,
    )

    assert result.stop_reason is StopReason.MAX_CYCLES
    assert result.cycles == 3
