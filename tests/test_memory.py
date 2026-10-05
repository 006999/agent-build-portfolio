"""Does anything survive a run? Day 4. The answer is meant to be no.

Nothing in this system persists, on purpose - the reasoning is in
the private build's Day 4 notes. These tests exist so that the day someone adds persistence,
it announces itself. Silent persistence never does.

State can survive a run in three ways, and they need three different
instruments:

  1. It changes what is sent.      -> compare the request bytes  (tests 1, 2)
  2. It lands on disk.             -> compare a tree snapshot    (test 5)
  3. It makes what is sent STALE   -> change the source and
     rather than different.           demand the change is seen  (test 6)

The third one is the one a byte comparison structurally cannot reach. A cache
does not change the bytes; it makes them plausible and wrong. That is the same
shape as the scripts/mutate.py bug from Day 3, where DEFAULT_INPUT_GUARDRAILS
still held the original objects - the list looked right and the contents were
stale.

Tests 1 to 5 assert on what was SENT, never on what came back. A leak reaches
the wire before it reaches an answer, and an answer is the model's, not ours.
"""

import json
from pathlib import Path

from fakes import FakeClient, FakeTool, FakeToolUseBlock

from src import tools
from src.agent import COMPLETION_SENTINEL, run_agent
from src.tools import lookup_invoice

REPO = Path(__file__).resolve().parent.parent

# A string that cannot occur anywhere else in this repo or in any reply the
# fake client is scripted to give. If it shows up in run B, it came from run A.
NONCE = "ZZQX-4471"


def _everything_sent(client: FakeClient) -> str:
    """Every byte of every request this client was asked to make.

    One flat string rather than a structure to walk, because the question is
    "did this text reach the model", and that question does not care which
    field it arrived in. `default=str` because the value under test is
    arbitrary - it must not be possible for an unserialisable object to make
    this quietly return less than it should.
    """
    return json.dumps(client.messages.calls, default=str, sort_keys=True)


# ---------------------------------------------------------------------------
# 1 and 2: nothing from run A reaches run B.
# ---------------------------------------------------------------------------


def _run_a() -> FakeClient:
    """A full run carrying the nonce through text, a tool call, and a result.

    Deliberately not a one-cycle text-only run. The leak surface includes tool
    inputs and tool results, and a run that never called a tool would not test
    them.
    """
    client = FakeClient(
        [
            [FakeToolUseBlock("roll_dice", {"note": NONCE})],
            f"The answer involves {NONCE}.\n{COMPLETION_SENTINEL}",
        ]
    )
    run_agent(
        f"Handle the matter of {NONCE}.",
        tools=[FakeTool(result=f"rolled a 4 for {NONCE}")],
        client=client,
    )
    return client


def test_a_second_run_sends_nothing_from_the_first():
    """The test that matters.

    Run A puts the nonce in the task, in a tool input, in a tool result and in
    a reply. Run B is a different task on a fresh client. If any store, cache or
    module-level list ever carries run A forward, the nonce arrives here.
    """
    client_a = _run_a()
    # The nonce really did travel in run A - otherwise this test passes because
    # nothing was ever there to leak, which is not the same result.
    assert NONCE in _everything_sent(client_a)

    client_b = FakeClient([f"Nothing to report.\n{COMPLETION_SENTINEL}"])
    run_agent("Summarise the position.", tools=[FakeTool()], client=client_b)

    assert NONCE not in _everything_sent(client_b)


def test_a_second_run_starts_from_exactly_the_new_task():
    """The stronger form of the test above.

    The nonce test catches one known string surviving. This catches anything at
    all surviving, including something a nonce would miss, by pinning the first
    request's messages byte for byte.
    """
    _run_a()

    client_b = FakeClient([f"Nothing to report.\n{COMPLETION_SENTINEL}"])
    run_agent("Summarise the position.", tools=[FakeTool()], client=client_b)

    assert client_b.messages.calls[0]["messages"] == [
        {"role": "user", "content": "Summarise the position."}
    ]


# ---------------------------------------------------------------------------
# 3 and 4: the caller can carry a fact forward, and only the caller can.
#
# One instrument, two opposite assertions. Either one alone proves nothing:
# test 3 alone passes if everything leaks, test 4 alone passes if nothing works.
# ---------------------------------------------------------------------------

PRIOR_FACT = "Placeholder Freight Ltd is always invoiced in EUR."
TASK = "Prepare the next invoice."


def test_a_prior_fact_reaches_the_model_when_the_caller_puts_it_in_the_task():
    """This is what "memory" means under the Day 4 design: the caller composes it.

    There is no store. The fact is a string the calling code chose to include,
    which is why it arrives inside `task` - and therefore why the input
    guardrails see it before any request is made.
    """
    client = FakeClient([f"Understood.\n{COMPLETION_SENTINEL}"])
    run_agent(f"{PRIOR_FACT}\n\n{TASK}", client=client)

    assert PRIOR_FACT in _everything_sent(client)


def test_a_prior_fact_does_not_reach_the_model_when_the_caller_does_not():
    """The default is forgetting. Same fact, same task, caller stays silent."""
    client = FakeClient([f"Understood.\n{COMPLETION_SENTINEL}"])
    run_agent(TASK, client=client)

    assert PRIOR_FACT not in _everything_sent(client)


# ---------------------------------------------------------------------------
# 5: nothing lands on disk.
# ---------------------------------------------------------------------------

# Hardcoded rather than read from .gitignore, because scripts/mutate.py copies
# the repo WITHOUT .git. A git-based check would error inside every mutant run
# and look like a finding in all of them - the exact harness bug that file's own
# docstring warns about.
IGNORED = (".git", ".venv", ".pytest_cache", "__pycache__", "logs", ".DS_Store")


def _tree_snapshot(root: Path) -> dict:
    """Every file under `root`, with its size and modification time.

    Size alone would miss an edit that keeps the length. mtime_ns is what
    catches a file being rewritten with different contents of the same size.
    """
    snapshot = {}
    for path in root.rglob("*"):
        if any(part.startswith(IGNORED) for part in path.parts):
            continue
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def test_a_run_leaves_no_trace_on_disk():
    """Catches persistence the request-byte tests cannot see.

    Honest limit: with a fake client this proves that run_agent and the tools it
    was handed write nothing. It does not prove the real SDK writes nothing.
    """
    before = _tree_snapshot(REPO)

    client = FakeClient([f"Done.\n{COMPLETION_SENTINEL}"])
    run_agent("Do something harmless.", tools=[FakeTool()], client=client)

    assert _tree_snapshot(REPO) == before


# ---------------------------------------------------------------------------
# 6: the third class - state that makes what is sent stale, not different.
# ---------------------------------------------------------------------------


def test_a_tool_reads_the_file_rather_than_remembering_it(tmp_path, monkeypatch):
    """src/tools.py:51, _read_invoices, says why it does not cache. Day 4's answer.

    An lru_cache on _read_invoices is module-level state that outlives every run
    in the process. The moment invoices.csv is per-client, that cache is a
    cross-client read - and no test above would notice, because a cached row and
    a fresh row are the same bytes on the wire.

    Verified by mutation: adding the cache passed all 105 tests before this one
    existed. See "tools: the invoice file is read once and cached" in
    scripts/mutate.py.

    Scope, stated plainly: this covers the one caching site that exists today.
    It is not a detector for the class. A second site appearing would not be
    noticed by anything here - that is a GAP line, not a claim.
    """
    csv_file = tmp_path / "invoices.csv"
    header = "invoice_id,client,amount,currency,issue_date,status\n"
    csv_file.write_text(header + "INV-0001,Acme,100.00,EUR,2026-07-31,unpaid\n")
    monkeypatch.setattr(tools, "INVOICE_FILE", csv_file)

    assert "100.00 EUR" in lookup_invoice("INV-0001")

    # The file changes underneath. A tool that reads sees this; a tool that
    # remembers does not.
    csv_file.write_text(header + "INV-0001,Acme,250.00,EUR,2026-07-31,paid\n")

    second = lookup_invoice("INV-0001")
    assert "250.00 EUR" in second
    assert "paid" in second
