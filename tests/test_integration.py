"""Does Claude actually route to the right tool? Real API, real money.

Everything here is marked `integration` and skipped by default. The unit tests
prove the loop plumbs tool calls correctly; these prove the descriptions in
src/tools.py earn their keep.
"""

import os
import re

import pytest

from src.agent import StopReason, run_agent
from src.guardrails import DEFAULT_INPUT_GUARDRAILS, DEFAULT_OUTPUT_GUARDRAILS
from src.tools import DEFAULT_TOOLS


def _names_of_tools_called(conversation: list[dict]) -> list[str]:
    called = []
    for message in conversation:
        if isinstance(message["content"], list):
            called.extend(
                block["name"]
                for block in message["content"]
                if block["type"] == "tool_use"
            )
    return called


def _first_assistant_text(conversation: list[dict]) -> str:
    for message in conversation:
        if message["role"] == "assistant":
            return "\n".join(
                block["text"]
                for block in message["content"]
                if block["type"] == "text"
            )
    return ""


@pytest.fixture(autouse=True)
def _needs_a_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.mark.integration
def test_a_two_tool_question_calls_both_tools_in_order():
    """The positive control.

    Without this, the two negative tests below pass even if the schemas were
    never wired into the request at all.
    """
    result = run_agent(
        "What is invoice INV-1043 worth in US dollars?",
        tools=DEFAULT_TOOLS,
        max_cycles=6,
    )

    assert _names_of_tools_called(result.conversation) == [
        "lookup_invoice",
        "convert_currency",
    ]
    assert result.stop_reason is StopReason.COMPLETE

    # These two lines replace an assertion on the converted amount. That figure
    # was pinned to the ECB rate on 29 Jul and had decayed by 31 Jul - the rate
    # moved 1.1380 -> 1.1476 and the answer with it, failing a test whose actual
    # subject is routing. A live rate cannot be asserted against a constant.
    #
    # 4820.00 is the invoice amount in data/invoices.csv. It does not move, and
    # it can only appear here if lookup_invoice really ran and its result really
    # reached the model.
    assert "4820" in result.final_answer.replace(",", "")
    # And the conversion has to have happened rather than the agent answering
    # uselessly - the thing the old amount assertion was there to catch. This is
    # the one part still tied to live output: a reply phrased "5,531.43 USD"
    # with no dollar sign would fail. Observed on every run so far, but that is
    # an observation, not a guarantee.
    assert re.search(r"\$\s?\d", result.final_answer)


@pytest.mark.integration
def test_a_general_question_about_invoices_does_not_call_the_invoice_tool():
    """A trap, not merely an unrelated task.

    "Invoice" appears twice and sits on lookup_invoice's trigger word. But
    there is no ID and nothing to look up. A system routing on keywords calls
    the tool and crashes on a guessed ID.
    """
    result = run_agent(
        "In two sentences, explain to a new hire what an invoice is and why a "
        "business sends one.",
        tools=DEFAULT_TOOLS,
        max_cycles=3,
    )

    assert _names_of_tools_called(result.conversation) == []
    # Answering from knowledge is the pass condition. Without this line the
    # test also passes when the agent says nothing useful at all.
    assert len(_first_assistant_text(result.conversation)) > 80


@pytest.mark.integration
def test_a_general_question_about_rates_does_not_call_the_currency_tool():
    """Same trap, other tool. Hits "exchange rates", names no amount.

    Deliberately does not assert stop_reason. Passing tools makes the model
    less likely to emit the sentinel on cycle one - measured 3/3 on this exact
    task - and that is a completion-protocol problem, not a routing one.
    Asserting it here would make a routing test fail for an unrelated reason.
    """
    result = run_agent(
        "In two sentences, explain why exchange rates move from day to day.",
        tools=DEFAULT_TOOLS,
        max_cycles=3,
    )

    assert _names_of_tools_called(result.conversation) == []
    assert len(_first_assistant_text(result.conversation)) > 80


# ---------------------------------------------------------------------------
# Day 3: the guardrails against a real model. The unit tests prove the loop
# runs guardrails correctly; these prove the real ones behave on real text.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_the_default_guardrails_do_not_block_a_normal_run():
    """The false-positive control, on live output rather than a fixed corpus.

    An ordinary two-tool question with every default guardrail installed must
    finish exactly as it does without them. A guardrail that fires here is
    firing on genuine work, which is the failure that costs a client.
    """
    result = run_agent(
        "What is invoice INV-1043 worth in US dollars?",
        tools=DEFAULT_TOOLS,
        input_guardrails=DEFAULT_INPUT_GUARDRAILS,
        output_guardrails=DEFAULT_OUTPUT_GUARDRAILS,
        max_cycles=6,
    )

    assert result.stop_reason is StopReason.COMPLETE, result.block_reason
    assert result.blocked_by is None
    assert _names_of_tools_called(result.conversation) == [
        "lookup_invoice",
        "convert_currency",
    ]


@pytest.mark.integration
def test_the_output_filter_fires_on_a_real_reply():
    """The negative half. Asking for the word is the only reliable way to get it.

    Note the task itself is clean, so this isolates the output filter - the
    input guardrails are installed and must let the request through.
    """
    result = run_agent(
        "Reply with exactly this sentence and nothing else: "
        "Our ledger is a revolutionary product.",
        input_guardrails=DEFAULT_INPUT_GUARDRAILS,
        output_guardrails=DEFAULT_OUTPUT_GUARDRAILS,
        max_cycles=2,
    )

    assert result.stop_reason is StopReason.BLOCKED_OUTPUT
    assert result.blocked_by == "no_banned_words"
    assert result.cycles == 1
    # Blocked on cycle 1, so nothing the model said was ever allowed through.
    assert result.final_answer == ""
