"""Break things on purpose and report which tests noticed. Not part of the agent.

A passing suite proves the tests agree with the code. It does not prove the
tests would have caught the code being wrong. This runs that second check:
apply one deliberate fault, run the suite, list what went red.

A mutation that kills nothing means the tests covering it are decoration. A
mutation that kills the wrong tests means they are coupled to something other
than the behaviour they claim to check.

Nothing in src/ imports this, and it imports nothing from src/. It shells out to
pytest against a throwaway copy of the repo, so the working tree is never
touched - which matters, because the whole point is to leave the code broken
long enough to run a suite against it.

    .venv/bin/python scripts/mutate.py            # every mutation
    .venv/bin/python scripts/mutate.py always     # only ones matching "always"

A note on this harness's own history, since it is the same class of mistake it
exists to find. The first version built the replacement guardrail with
`name="NOT_EMPTY"` - the Python constant - rather than the real name,
"not_empty". Every one of the eight guardrail mutations then killed
test_every_shipped_guardrail_is_covered, which checks that every shipped
guardrail has a test corpus. That looked like eight findings and was one
harness bug: the runs were measuring a rename, not a mutation. It is fixed
below by reading `.name` off the original object. If a mutation ever kills the
same test across every unrelated run again, suspect this file before the code.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "bin" / "python"

# The constants in src/guardrails.py, not the guardrail names. The harness reads
# the real name off each object so a mutant keeps the identity of what it
# replaces - see the note in the module docstring.
GUARDRAIL_CONSTANTS = [
    "NOT_EMPTY",
    "NOT_OVERSIZED",
    "NO_KNOWN_INJECTION_PHRASES",
    "NO_BANNED_WORDS",
]


def fresh_copy(work: Path) -> None:
    """A clean copy of the repo, outside it, for one mutation to be applied to."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(
        REPO,
        work,
        ignore=shutil.ignore_patterns(".venv*", ".git", ".pytest_cache", "*.pyc"),
    )


def _tests_seen(stdout: str) -> int:
    """How many tests the run actually accounted for, from the summary line.

    Sums every "<n> <outcome>" pair pytest reports - passed, failed, error,
    skipped - excluding deselected, which were never going to run.
    """
    total = 0
    for count, word in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", stdout):
        total += int(count)
    return total


def run_tests(work: Path) -> tuple[list[str], str]:
    """Run the unit suite in the copy and return (failed test ids, summary line).

    The id pattern stops at " - " rather than at whitespace. Parametrised ids
    contain spaces, so a \\S+ match truncates them and two different cases
    collapse into one line - which undercounts the kills.

    ERROR is matched as well as FAILED. A mutant that breaks an import kills
    tests at collection, which prints ERROR and no FAILED at all - and this
    harness would otherwise score that as SURVIVED, which is the opposite of
    the truth.
    """
    result = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=work,
        capture_output=True,
        text=True,
    )
    failed = sorted(
        set(re.findall(r"^(?:FAILED|ERROR) (.+?)(?:\s+-\s|$)", result.stdout, re.M))
    )
    lines = result.stdout.strip().splitlines()
    return failed, lines[-1] if lines else "(no output)"


def mutate_guardrail(work: Path, constant: str, *, always_allow: bool) -> None:
    """Replace one guardrail with one that always allows, or always refuses.

    Appended to the end of the module so it runs after the real definitions.
    The name is read off the original object rather than retyped, and the two
    default lists are rebuilt, because they were built from the originals and
    would otherwise still hold them.
    """
    verdict = "ALLOWED" if always_allow else 'Verdict(allowed=False, reason="mutant")'
    path = work / "src" / "guardrails.py"
    path.write_text(
        path.read_text()
        + f"\n\n{constant} = Guardrail("
        + f"name={constant}.name, check=lambda text: {verdict})\n"
        + "DEFAULT_INPUT_GUARDRAILS = "
        + "[NOT_EMPTY, NOT_OVERSIZED, NO_KNOWN_INJECTION_PHRASES]\n"
        + "DEFAULT_OUTPUT_GUARDRAILS = [NO_BANNED_WORDS]\n"
    )


def mutate_source(work: Path, filename: str, old: str, new: str) -> None:
    """Swap one exact line in one source file. Loud if the anchor has moved."""
    path = work / "src" / filename
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"anchor not found in {filename}: {old!r}")
    path.write_text(text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# Day 5 logger mutations.
#
# These wrap JsonlLogger.__call__ from outside rather than editing its body,
# because the fault being modelled is a logger that writes a plausible, complete
# looking file with one thing missing. Appending the patch keeps the original
# method intact and reachable, so the mutant behaves like a real bug rather than
# like a broken file.
# ---------------------------------------------------------------------------


def _patch_audit(work: Path, body: str) -> None:
    path = work / "src" / "audit.py"
    path.write_text(path.read_text() + "\n\n" + body)


def mutate_logger_drops_field(work: Path, field: str) -> None:
    """Every event still written, one field silently missing from all of them."""
    _patch_audit(
        work,
        "_real_call = JsonlLogger.__call__\n"
        "def _dropping(self, event_type, **fields):\n"
        f"    fields.pop({field!r}, None)\n"
        "    _real_call(self, event_type, **fields)\n"
        "JsonlLogger.__call__ = _dropping\n",
    )


def mutate_logger_skips_event(work: Path, event_type: str) -> None:
    """One kind of event never written at all. The rest of the log looks fine."""
    _patch_audit(
        work,
        "_real_call = JsonlLogger.__call__\n"
        "def _skipping(self, event_type, **fields):\n"
        f"    if event_type == {event_type!r}:\n"
        "        return\n"
        "    _real_call(self, event_type, **fields)\n"
        "JsonlLogger.__call__ = _skipping\n",
    )


def mutate_logger_buffers_tool_results(work: Path) -> None:
    """Tool results held back and flushed on the next event - unless it ends the run.

    Day 10 retargeted this. It used to hold results back past an `error` event
    only, and the scenario that killed it was a tool raising mid-cycle. Day 10
    turned that crash into a `tool_failed` event, so no crash path had pending
    results any more and the scenario test stopped killing it - shown by a run
    before the retarget, where only the two writer tests that call the logger
    directly with a tool_result went red, and they go red for a different
    reason (the mutant swallows the value they expect to raise on).
    `tool_failed` is now treated the same as `error`, which is the fault this
    always modelled: a completed result not written before the event that
    explains why the run ended.
    """
    _patch_audit(
        work,
        "_real_call = JsonlLogger.__call__\n"
        "def _buffering(self, event_type, **fields):\n"
        "    pending = getattr(self, '_pending', [])\n"
        "    if event_type == 'tool_result':\n"
        "        pending.append((event_type, fields))\n"
        "        self._pending = pending\n"
        "        return\n"
        "    if event_type not in ('error', 'tool_failed'):\n"
        "        for held_type, held_fields in pending:\n"
        "            _real_call(self, held_type, **held_fields)\n"
        "        self._pending = []\n"
        "    _real_call(self, event_type, **fields)\n"
        "JsonlLogger.__call__ = _buffering\n",
    )


MUTATIONS = []

for _constant in GUARDRAIL_CONSTANTS:
    MUTATIONS.append(
        (
            f"{_constant}: always refuses",
            lambda work, c=_constant: mutate_guardrail(work, c, always_allow=False),
        )
    )
    MUTATIONS.append(
        (
            f"{_constant}: always allows",
            lambda work, c=_constant: mutate_guardrail(work, c, always_allow=True),
        )
    )

def mutate_audit(work: Path, old: str, new: str) -> None:
    """Swap one exact line in src/audit.py. Same contract as mutate_source."""
    mutate_source(work, "audit.py", old, new)


def mutate_script(work: Path, filename: str, old: str, new: str) -> None:
    """Swap one exact line in a scripts/ file.

    mutate_source only reaches src/. scripts/read_log.py is the instrument the
    audit-trail claim is measured with, so it is mutated too.
    """
    path = work / "scripts" / filename
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"anchor not found in scripts/{filename}: {old!r}")
    path.write_text(text.replace(old, new, 1))


MUTATIONS += [
    (
        "loop: input guardrails never run",
        lambda work: mutate_source(
            work,
            "agent.py",
            "blocked = _first_block(input_guardrails, task)",
            "blocked = None",
        ),
    ),
    (
        "loop: sentinel checked before the output filter",
        lambda work: mutate_source(
            work,
            "agent.py",
            "blocked = _first_block(output_guardrails, reply)",
            "blocked = None if _is_complete(reply)"
            " else _first_block(output_guardrails, reply)",
        ),
    ),
    (
        "loop: tool call checked before the output filter",
        lambda work: mutate_source(
            work,
            "agent.py",
            "blocked = _first_block(output_guardrails, reply)",
            "blocked = None if _tool_use_blocks(response)"
            " else _first_block(output_guardrails, reply)",
        ),
    ),
    (
        "loop: final_answer assigned before the output filter",
        # The anchor carries its comment line. Day 5 indented the loop body into
        # a try/except and added a run_end event that also passes
        # `final_answer=final_answer` at the same indentation, so the bare line
        # is no longer unique - and `replace(..., 1)` would have silently
        # mutated the log event instead of the returned result. The comment is
        # what makes it unambiguous.
        lambda work: mutate_source(
            work,
            "agent.py",
            "                    # blocked text, and never prose we wrote.\n"
            "                    final_answer=final_answer,",
            "                    # blocked text, and never prose we wrote.\n"
            "                    final_answer=reply or final_answer,",
        ),
    ),
    # Day 4. Three ways state can survive a run, and they need three different
    # tests. Each mutation below makes state survive in one of those ways.
    #
    # The tuple-returning lambdas apply two edits: an import, then the change
    # that needs it. Written this way rather than with `__import__(...)` inline
    # so that a mutant that fails to compile is impossible - a SyntaxError makes
    # every test red at collection, which reads as "killed by everything" and
    # would be this harness lying to me again.
    (
        "loop: the conversation survives between runs",
        lambda work: mutate_source(
            work,
            "agent.py",
            '    messages: list[dict] = [{"role": "user", "content": task}]',
            "    messages: list[dict] = getattr(run_agent, \"_leak\", []) + [\n"
            '        {"role": "user", "content": task}\n'
            "    ]\n"
            "    run_agent._leak = messages",
        ),
    ),
    (
        "loop: the task is written to disk",
        lambda work: (
            mutate_source(work, "agent.py", "import anthropic", "import pathlib\n\nimport anthropic"),
            mutate_source(
                work,
                "agent.py",
                '    messages: list[dict] = [{"role": "user", "content": task}]',
                '    pathlib.Path("memory.json").write_text(task)\n'
                '    messages: list[dict] = [{"role": "user", "content": task}]',
            ),
        ),
    ),
    (
        "tools: the invoice file is read once and cached",
        lambda work: (
            mutate_source(work, "tools.py", "import csv", "import csv\nimport functools"),
            mutate_source(
                work,
                "tools.py",
                "def _read_invoices() -> list[dict]:",
                "@functools.lru_cache(maxsize=None)\n"
                "def _read_invoices() -> list[dict]:",
            ),
        ),
    ),
    (
        "loop: the cycle ceiling is not enforced",
        lambda work: mutate_source(
            work,
            "agent.py",
            "    if max_cycles > MAX_CYCLES_CEILING:",
            "    if False:",
        ),
    ),
    # Day 5. A log is only worth having if it is true, and "the file is not
    # empty" is not a test. Each of these makes the log lie in one specific way.
    (
        "logger: tool arguments are dropped",
        lambda work: mutate_logger_drops_field(work, "args"),
    ),
    (
        "logger: the error event drops the exception message",
        lambda work: mutate_logger_drops_field(work, "message"),
    ),
    (
        "logger: tool results are dropped",
        lambda work: mutate_logger_skips_event(work, "tool_result"),
    ),
    (
        "logger: the completion check is not logged",
        lambda work: mutate_logger_skips_event(work, "completion_check"),
    ),
    (
        "logger: token counts are always zero",
        lambda work: mutate_source(
            work,
            "agent.py",
            "                input_tokens=response.usage.input_tokens,",
            "                input_tokens=0,",
        ),
    ),
    (
        "loop: on_event is never called at the exit seam",
        # Four-space indent picks out the normal exit. The two blocked run_end
        # calls are deeper, so this leaves them alone - the point is a run that
        # ends normally and says nothing about it.
        lambda work: mutate_source(
            work,
            "agent.py",
            '    on_event(\n        "run_end",',
            '    _ignore(\n        "run_end",',
        ),
    ),
    (
        # Must be killed by the SDK-crash scenario and by nothing else. In a run
        # that succeeds, `messages` has not been appended to yet when the call
        # returns, so both orderings write a byte-identical log. If any other
        # test goes red here, the request event is carrying something it should
        # not be.
        "loop: the request event moves to after the API call",
        lambda work: (
            mutate_source(
                work,
                "agent.py",
                '        on_event(\n'
                '            "request",\n'
                '            actor="agent",\n'
                "            cycle=cycle,\n"
                "            model=model,\n"
                "            message_count=len(messages),\n"
                "        )",
                "        pass",
            ),
            mutate_source(
                work,
                "agent.py",
                "            response = client.messages.create(**request)",
                "            response = client.messages.create(**request)\n"
                '            on_event("request", actor="agent", cycle=cycle,'
                " model=model, message_count=len(messages))",
            ),
        ),
    ),
    (
        # Same shape as the one above: invisible unless the run crashes. Results
        # are held and flushed on the next event, which preserves order exactly
        # in every run that finishes - but an error event does not flush, so a
        # crash loses the tool calls that had already completed.
        "logger: tool results are not flushed before a crash",
        lambda work: mutate_logger_buffers_tool_results(work),
    ),
    # Day 5, second pass. Two guardrail-crash sites and two writer defaults.
    #
    # The first pair is the point of the exercise: an input guardrail that
    # raises was unlogged at the first Day 5 logging commit, and the output side was believed
    # covered rather than measured. Each mutation must be killed by exactly one
    # scenario. If either is killed by more, the try/except was widened past the
    # one call it is meant to guard.
    (
        "loop: an input guardrail that raises is not logged",
        # `raise` as the first statement of the except block. The on_event call
        # below it becomes unreachable, so the exception still propagates and
        # behaviour is unchanged - only the record disappears. That is exactly
        # the state the tree was in at that commit.
        #
        # The four-space indent is what makes this the input-guardrail handler
        # and not the loop body's, which sits at eight.
        lambda work: mutate_source(
            work,
            "agent.py",
            "\n    except Exception as exc:\n",
            "\n    except Exception as exc:\n        raise\n",
        ),
    ),
    (
        "loop: a failing output guardrail fails open",
        # The realistic bad change, not a contrived one: someone on Day 10
        # decides a guardrail that errors should not kill the run, wraps the
        # call, and treats the failure as "nothing to report". The text then
        # goes out unchecked and no error is recorded - a safety check that
        # fails open and says nothing.
        #
        # An earlier version of this mutation raised KeyboardInterrupt to escape
        # the handler. That aborted pytest itself: 23 tests never ran, no FAILED
        # line was printed, and the harness scored it SURVIVED. See the count
        # guard in run_tests below, which now catches that class of lie.
        lambda work: mutate_source(
            work,
            "agent.py",
            "            blocked = _first_block(output_guardrails, reply)",
            "            try:\n"
            "                blocked = _first_block(output_guardrails, reply)\n"
            "            except Exception:\n"
            "                blocked = None",
        ),
    ),
    (
        "audit: the repr fallback is restored",
        lambda work: mutate_audit(
            work,
            "json.dumps(event, allow_nan=False)",
            "json.dumps(event, allow_nan=False, default=str)",
        ),
    ),
    (
        "audit: non-finite floats are written as bare NaN",
        lambda work: mutate_audit(
            work,
            "json.dumps(event, allow_nan=False)",
            "json.dumps(event)",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Day 7. The reader that the whole audit-trail claim is measured with.
#
# This is the mutation that reproduces the bug Day 7 found rather than one that
# invents a new way to break things: until today, read_log.py did exactly what
# the mutation below restores - treated every event in a file as one run. It
# passed all 174 tests, because every test wrote one run to a fresh tmp_path
# file. Against a real four-run workflow log it reported a merged run
# that never happened, with no error.
#
# So the mutation's job is to prove the new tests would have caught it. If it
# survives, the tests are decoration and the audit-trail row does not tick.
# ---------------------------------------------------------------------------

MUTATIONS += [
    (
        "read_log: every event in a file is treated as one run",
        # The exact behaviour that shipped from Day 5 to Day 7.
        lambda work: mutate_script(
            work,
            "read_log.py",
            "    return list(groups.values())",
            "    return [events]",
        ),
    ),
    (
        "read_log: parse merges a multi-run file instead of refusing it",
        # The tempting softer fix: add parse_all, leave parse merging quietly.
        # Every existing caller keeps working and the silent-wrong-answer path
        # survives untouched for the next person who calls parse().
        lambda work: mutate_script(
            work,
            "read_log.py",
            "    if len(groups) > 1:",
            "    if False:",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Day 10. Failure modes. Each fault below makes a run that should stop cleanly
# either carry on, crash, or stop with the evidence missing from the log.
# ---------------------------------------------------------------------------

MUTATIONS += [
    (
        "stop: a max_tokens reply is nudged, not stopped",
        lambda work: mutate_source(
            work, "agent.py", '    "max_tokens": StopReason.TRUNCATED,\n', ""
        ),
    ),
    (
        "stop: a refusal reply is nudged, not stopped",
        lambda work: mutate_source(
            work, "agent.py", '    "refusal": StopReason.REFUSED,\n', ""
        ),
    ),
    (
        "stop: the check runs before the output guardrail",
        # The guardrail is skipped for a reply the check would stop - which is
        # what moving the check above it means.
        lambda work: mutate_source(
            work,
            "agent.py",
            "            blocked = _first_block(output_guardrails, reply)",
            "            blocked = None if response.stop_reason in API_STOP_REASONS"
            " else _first_block(output_guardrails, reply)",
        ),
    ),
    (
        "stop: the cut-off text becomes the final answer",
        lambda work: mutate_source(
            work,
            "agent.py",
            "            stopped_by_api = API_STOP_REASONS.get(response.stop_reason)",
            "            stopped_by_api = API_STOP_REASONS.get(response.stop_reason)\n"
            "            if stopped_by_api is not None:\n"
            "                final_answer = reply",
        ),
    ),
    (
        "stop: the run ends before the reply event is logged",
        # An early copy of the stop, above the reply event. The run still
        # stops with the right reason; only the evidence is missing.
        lambda work: mutate_source(
            work,
            "agent.py",
            "            reply = _text_from(response)\n",
            "            reply = _text_from(response)\n"
            "            if response.stop_reason in API_STOP_REASONS:\n"
            "                on_event('run_end', actor='agent',"
            " stop_reason=API_STOP_REASONS[response.stop_reason].value,"
            " cycles=cycles, final_answer=final_answer,"
            " conversation_length=len(messages))\n"
            "                return AgentResult(stop_reason="
            "API_STOP_REASONS[response.stop_reason], cycles=cycles,"
            " final_answer=final_answer, conversation=messages)\n",
        ),
    ),
]


MUTATIONS += [
    (
        "tool failure: re-raises instead of stopping",
        lambda work: mutate_source(
            work,
            "agent.py",
            "                    except Exception as exc:\n",
            "                    except Exception as exc:\n"
            "                        raise\n",
        ),
    ),
    (
        "tool failure: the tool_failed event is not logged",
        lambda work: mutate_source(
            work,
            "agent.py",
            '                        on_event(\n                            "tool_failed",',
            '                        _ignore(\n                            "tool_failed",',
        ),
    ),
    (
        "tool failure: the error is fed back to the model and the loop carries on",
        # The Day 13 shape, arriving early: the error becomes the tool result.
        # The original handler body is kept under `if False:` so the mutant
        # compiles and behaves exactly like someone wrapping the call.
        lambda work: mutate_source(
            work,
            "agent.py",
            "                    except Exception as exc:\n",
            "                    except Exception as exc:\n"
            "                        result = f'{type(exc).__name__}: {exc}'\n"
            "                    if False:\n",
        ),
    ),
    (
        "unknown tool: treated as a tool failure",
        # registry.get returns None, the check is skipped, and None.run raises
        # AttributeError inside the narrow try - so it is reported as
        # tool_failed, losing the distinction between the two owners.
        lambda work: mutate_source(
            work, "agent.py", "                    if tool is None:", "                    if False:"
        ),
    ),
    (
        "unknown tool: crashes the run again",
        lambda work: mutate_source(
            work,
            "agent.py",
            "tool = registry.get(block.name)",
            "tool = registry[block.name]",
        ),
    ),
]


MUTATIONS += [
    (
        "read_log: only the last reply's stop reason is kept",
        lambda work: mutate_script(
            work,
            "read_log.py",
            "            log.api_stop_reasons[reason] = log.api_stop_reasons.get(reason, 0) + 1",
            "            log.api_stop_reasons = {reason: 1}",
        ),
    ),
    (
        "read_log: one tally shared by every run in a file",
        lambda work: mutate_script(
            work,
            "read_log.py",
            "    api_stop_reasons: dict[str, int] = field(default_factory=dict)",
            "    api_stop_reasons: dict[str, int] = field(default_factory=lambda _d={}: _d)",
        ),
    ),
    (
        "read_log: the tally is never printed",
        lambda work: mutate_script(
            work,
            "read_log.py",
            '        print(\n            f"api stops:',
            '        (lambda *a: None)(\n            f"api stops:',
        ),
    ),
]


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    selected = [(label, apply) for label, apply in MUTATIONS if wanted in label]

    if not selected:
        print(f"no mutation matching {wanted!r}")
        return 1

    survivors = []
    miscounted = []
    root = Path(tempfile.mkdtemp(prefix="fde-mutate-"))

    try:
        work = root / "repo"

        # The clean run first. Every mutant is expected to account for exactly
        # this many tests - see the check below for why that matters.
        fresh_copy(work)
        _, clean_summary = run_tests(work)
        baseline = _tests_seen(clean_summary)
        print(f"baseline: {baseline} tests\n    {clean_summary}")

        for label, apply in selected:
            fresh_copy(work)
            apply(work)
            failed, summary = run_tests(work)
            seen = _tests_seen(summary)

            print(f"\n### {label}")
            print(f"    {summary}")
            for name in failed:
                print(f"    RED  {name}")
            if not failed:
                print("    RED  (nothing - this mutation SURVIVED)")
                survivors.append(label)

            # A mutant must not change how many tests RUN, only whether they
            # pass. A KeyboardInterrupt raised inside the loop aborts pytest
            # itself: the remaining tests never run, no FAILED line is printed,
            # and the count above reads SURVIVED when the mutant actually
            # destroyed the suite. Found on Day 5 by a mutation that scored
            # 0/1 while silently dropping 23 tests.
            #
            # This is the Day 3 rule applied to this file's own output: when a
            # tool reports two numbers about the same thing, make them agree
            # before believing either. The numbers here are "tests that ran"
            # and "tests there are".
            if seen != baseline:
                print(f"    WARN counted {seen} tests, expected {baseline}")
                miscounted.append((label, seen))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{len(selected) - len(survivors)}/{len(selected)} mutations killed.")
    for label in survivors:
        print(f"  SURVIVED: {label}")
    for label, seen in miscounted:
        print(f"  MISCOUNTED: {label} ran {seen}/{baseline} tests - result is not trustworthy")

    return 1 if (survivors or miscounted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
