# Architecture

A short note on how the agent loop works, what stops it, and how the tests check it. It describes this public copy only. The code is in `src/agent.py`.

## The loop

1. Check the task with the input guardrails. If one refuses it, stop with `blocked_input`. The model never sees it.
2. Send the conversation to the model.
3. Check the reply with the output guardrails. This happens on every reply, including one that only asks for a tool. If one refuses it, stop with `blocked_output`.
4. If the reply was cut off at the token limit, stop with `truncated`. If the model refused, stop with `refused`.
   If the caller set `max_tokens_per_run`, add this reply's input and output tokens to the run total. If the total is now above the budget, stop with `budget_exceeded`. A total equal to the budget is allowed. If the reply has no usable token counts, raise (see below).
5. If the model asks for tools, run them in order. If it names a tool that is not registered, stop with `unknown_tool`. If a tool raises an error, stop with `tool_failed`. Otherwise add the results to the conversation and go back to step 2.
6. If the reply ends with the completion marker (`TASK_COMPLETE` on its own last line), stop with `complete`. If not, add a short "continue" message and go back to step 2.
7. If the cycle limit is reached first, stop with `max_cycles`.

There is no framework. The loop is one function, `run_agent`, so every path through it can be read and tested.

## Why every run ends with a named stop reason

A run that returns ends in one of nine ways: `complete`, `max_cycles`, `blocked_input`, `blocked_output`, `truncated`, `refused`, `tool_failed`, `unknown_tool`, `budget_exceeded`. Returning a name instead of free text means a caller, a test or the audit log can tell a finished task from a stopped one. Each reason has a test that triggers it with a fake model client.

One path does not return a reason. If the API call or a guardrail raises an error, the loop writes the error to the log and raises it again, so the caller gets an exception, not a result. With a budget set, a reply whose token counts are missing takes the same path: the loop raises ValueError rather than count the reply as free.

## Guardrails

Input checks catch empty or oversized tasks and known injection phrases. Output checks catch banned words. Each check is tested with text it must block and text it must allow. Two known misses are pinned as tests, so a fix later shows up as a changed test, not a silent change.

## Memory

Nothing carries over between runs. A second run starts clean, nothing is written to disk, and the invoice tool rereads its file on every call instead of caching it. A test checks this.

## Audit log

Each event is written as one JSON line. `scripts/read_log.py` splits the log into runs and rebuilds each one. Tests compare the rebuilt run with what the loop actually returned, including for runs that crashed.

## How the tests are checked

The unit tests use a fake model client, so they need no network and no API key. Passing tests only prove the tests pass, so `scripts/mutate.py` breaks the code on purpose in 47 ways, one at a time, in a temporary copy, and checks that the tests fail each time. It also checks that the broken copy runs the same number of tests as the clean one, and flags any mismatch, so a fault that crashes the test run is not mistaken for one that was caught. Result on 17 November 2026: 47 of 47 caught.

A real gap showed up in a run log, not in the tests. A workflow run in the private build had 4 of 6 replies cut off at the token limit, and the loop kept nudging the model instead of stopping. The fake client could not return a cut-off reply, so no test could see this. The fake was fixed first so it could return one. Then the loop was changed to stop with `truncated`. The harness now includes a mutation that brings back the old behaviour, and the tests catch it.

## Known limits

* Runs locally only. Not deployed.
* The cycle limit bounds steps, not cost. The token budget bounds cost only when a caller sets one, and is checked after each reply, so a run can go over by up to one reply. It has been tested only against the fake client.
* Integration tests call the real API and were not run for this snapshot.
* The document-event code in the log reader and the draft-length guardrail have no tests in this copy.
* The invoice data is invented. The currency tool calls a public exchange-rate service, which the unit tests replace.
