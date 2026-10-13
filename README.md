# agent-build-portfolio

[![tests](https://github.com/006999/agent-build-portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/006999/agent-build-portfolio/actions/workflows/tests.yml)

A single tool-using agent loop written by hand in Python on the Anthropic SDK, with guardrails, an audit log, and a mutation harness that checks the tests catch real faults.

How it works: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

This is a public subset of a larger private build. The parts that touched real business data were left out; what is here runs on its own.

## What it demonstrates

- **A hand-written agent loop, no framework.** `src/agent.py` sends the request, runs any tools the model asks for, feeds the results back, and repeats until the model signals it is done or a limit is hit.
- **Every run that returns ends with a named reason.** `src/agent.py` defines `StopReason`: complete, max_cycles, blocked_input, blocked_output, truncated, refused, tool_failed, unknown_tool. `tests/test_agent.py` covers each path with a fake client. An error from the API call or a guardrail is not given a reason: it is logged and raised again.
- **Input and output guardrails, tested both ways.** `src/guardrails.py` checks the task before the model sees it (empty, oversized, known injection phrases) and checks replies afterwards (banned words). `tests/test_guardrails.py` tests every guardrail with text it must stop and text it must let through, and pins two known misses as tests.
- **Nothing carries over between runs.** `tests/test_memory.py` checks that a second run sends nothing from the first, writes nothing to disk, and that the invoice tool rereads its file rather than caching it.
- **An audit log you can rebuild a run from.** `src/audit.py` writes one JSON line per event. `scripts/read_log.py` splits a log file into runs and rebuilds each one. `tests/test_logging.py` compares the rebuild against what the loop actually returned, including for runs that crash.
- **Tests that are checked against deliberate faults.** `scripts/mutate.py` breaks the code in 43 specific ways, one at a time, and reports which tests went red.

## Run it

Needs Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Unit tests (no network, no API key, no cost):

```bash
.venv/bin/python -m pytest
```

Output on 5 Oct 2026:

```
====================== 145 passed, 6 deselected in 0.62s =======================
```

Mutation harness (28 seconds on the machine it was measured on, one run):

```bash
.venv/bin/python scripts/mutate.py
```

Output on 5 Oct 2026 (first two and last lines):

```
baseline: 145 tests
    145 passed, 6 deselected in 0.37s
...
43/43 mutations killed.
```

Integration tests call the real Anthropic API, so they cost money and are switched off by default (`pytest.ini` filters out the `integration` marker; these are the 6 "deselected"). To run them, put `ANTHROPIC_API_KEY` in a `.env` file and run:

```bash
.venv/bin/python -m pytest -m integration
```

To watch one run by hand (also calls the API): `.venv/bin/python -m src.run_demo`.

## How it is tested

- **145 unit tests**, all using fake clients, so no test touches the network.
- **Mutation testing.** A passing suite only shows the tests agree with the code. `scripts/mutate.py` copies the project to a temporary folder, applies one deliberate fault (for example "the input guardrails never run" or "the log drops tool arguments"), runs the unit tests, and records which tests failed. A fault no test notices is a gap in the tests. It also checks that every faulty copy ran the same number of tests as the clean one, so a fault that crashes the test run cannot be mistaken for one that was caught. Result on 5 Oct 2026: 43 of 43 faults caught.
- **6 integration tests** against the real model. They check that the model picks the right tool from the tool descriptions, that the guardrails work on real replies, and that an unfinishable task stops at the cycle limit. They were not run for this snapshot.

## Status and limits

- It runs locally, by hand. Nothing is deployed.
- The integration tests need an Anthropic API key and cost money per run.
- This is a learning build that is still in progress.
- The currency tool calls a public exchange-rate service (frankfurter.dev). The unit tests replace that call; a real run needs network access.
- The invoice data in `data/invoices.csv` is invented.
- `scripts/read_log.py` contains code for reading document events written by a workflow in the private build. That code is untested in this public copy.
- `src/guardrails.py` contains a draft-length guardrail (`DRAFT_NOT_OVERSIZED`, `DRAFT_GUARDRAILS`) used only by that same workflow. It is also untested in this public copy.
- The cycle limit bounds how many steps a run takes, not how much it costs. Token counts are logged per cycle, but nothing enforces a token budget yet.

## What I would do next

- Enforce a per-run token budget using the token counts the log already records.
- Add tests for the read_log document-event code and the draft guardrail, or remove that code from this copy.
- Run the integration tests on a schedule against a fixed small budget, and record their results next to the unit-test results.
