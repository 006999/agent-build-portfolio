"""Stand-ins for the API client, a tool, and a guardrail. Test support only.

These were written by hand rather than with unittest.mock so that the shape of
what the loop reads from a response stays visible on the page. They moved here
from tests/test_agent.py on Day 4, when tests/test_memory.py needed the same
ones. Two copies of the loop's response contract in the tree is exactly the
drift the Tool dataclass exists to prevent, so there is one copy.

Nothing here imports src.tools or src.guardrails, and that is the point: if the
loop tests used the real invoice tool or the real banned-word guardrail, a loop
that had quietly grown a special case for either would still pass.
"""

import copy


class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, name: str, tool_input: dict, block_id: str = "toolu_1"):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = tool_input


class FakeUsage:
    """Added on Day 5. The real response carries this and Day 1-4 ignored it.

    A Day 3 comment in src/agent.py said so - "response.usage comes back on
    every call below and is discarded" (since reworded; now at src/agent.py:74)
    - which is why adding it here is filling
    in a known gap rather than inventing a field.
    """

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeStopDetails:
    """Only present on a refusal. The real response has it as None otherwise."""

    def __init__(self, category: str | None):
        self.type = "refusal"
        self.category = category


class FakeResponse:
    def __init__(
        self,
        reply,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str | None = None,
        stop_details=None,
    ):
        """A reply is either a string of text or a ready-made list of blocks."""
        if isinstance(reply, str):
            self.content = [FakeTextBlock(reply)]
        else:
            self.content = list(reply)

        # Derived rather than passed in, the way the real API derives it.
        #
        # Day 8 adds the override, and keeps derivation as the default so every
        # test written before today sends a byte-identical response. Two of the
        # four stop reasons this fake now has to be able to express - "refusal"
        # and "max_tokens" - cannot be derived from content at all: a refusal is
        # an HTTP 200 whose body need not match the schema, and a truncation
        # looks exactly like a short reply. A fake that could not produce them
        # could not test the two paths that matter most on Day 8.
        if stop_reason is not None:
            self.stop_reason = stop_reason
        else:
            has_tool_use = any(block.type == "tool_use" for block in self.content)
            self.stop_reason = "tool_use" if has_tool_use else "end_turn"

        self.stop_details = stop_details
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeMessages:
    def __init__(self, replies: list, clock=None, seconds_per_call: float = 0):
        self._replies = list(replies)
        self.calls: list[dict] = []
        # Day 10. Optional: each call moves `clock` on by `seconds_per_call`, as
        # if the request took that long. This is how the search time limit is
        # tested without a single sleep.
        self._clock = clock
        self._seconds_per_call = seconds_per_call
        # What this fake actually handed back, recorded rather than recomputed.
        # tests/test_logging.py uses it as token ground truth, and a test that
        # re-derived the formula would be checking the formula against itself.
        self.usages: list[tuple[int, int]] = []

    def _usage_for_call(self) -> tuple[int, int]:
        """A different pair of numbers on every call, derived from the call index.

        Deliberately not a constant. A logger that recorded cycle 1's tokens for
        every cycle would pass against fixed numbers, and per-cycle counts are
        the whole reason Day 5 logs them at all.
        """
        n = len(self.calls)
        return 1000 + n, 10 + n

    def create(self, **kwargs) -> FakeResponse:
        # Deep-copied, not appended as handed over. The loop passes the same
        # `messages` list every cycle and appends to it, so storing the object
        # would make every recorded call point at the finished transcript - a
        # recorder that records a reference is not recording. Found on Day 4 by
        # a test that asserted what the first request contained and got the
        # last one. Same shape as the Day 3 mutate.py bug, where a rebuilt list
        # still held the original objects.
        self.calls.append(copy.deepcopy(kwargs))
        if self._clock is not None:
            self._clock.advance(self._seconds_per_call)
        input_tokens, output_tokens = self._usage_for_call()
        self.usages.append((input_tokens, output_tokens))
        if self._replies:
            reply = self._replies.pop(0)
            # A scripted entry may already be a FakeResponse, which is how a
            # test asks for a stop_reason this fake would not otherwise derive.
            # Its usage is filled in here so the token ground truth in
            # `self.usages` stays the single record of what was handed back.
            if isinstance(reply, FakeResponse):
                reply.usage = FakeUsage(input_tokens, output_tokens)
                return reply
            return FakeResponse(
                reply,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return FakeResponse(  # never the sentinel
            "Still working on it.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class FakeClient:
    """Stands in for anthropic.Anthropic(). Only `.messages.create` is used.

    `messages.calls` is the record of every request this client was asked to
    make. tests/test_memory.py reads it to assert on what was sent rather than
    on what came back, so it is part of this fake's contract, not an accident.
    """

    def __init__(self, replies: list | None = None, clock=None, seconds_per_call: float = 0):
        self.messages = FakeMessages(replies or [], clock, seconds_per_call)
        self.options: list[dict] = []

    def with_options(self, **options):
        """Day 10. The real client returns a copy with the options applied.

        This one records them and returns itself, so `messages.calls` stays the
        single record of what was sent - whichever handle sent it.
        """
        self.options.append(options)
        return self


class FakeClock:
    """Time that moves only when something moves it. No test sleeps.

    Called with no arguments, like time.monotonic. Starts away from zero so a
    bug that treats the clock reading itself as elapsed time shows up.
    """

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTool:
    """A tool the loop has never heard of, with a deliberately silly name."""

    def __init__(self, name: str = "roll_dice", result="rolled a 4", run=None):
        self.name = name
        self.calls: list[dict] = []
        self._result = result
        self._run = run

    def to_api_format(self) -> dict:
        return {
            "name": self.name,
            "description": f"does whatever {self.name} does",
            "input_schema": {"type": "object", "properties": {}},
        }

    def run(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if self._run is not None:
            return self._run(**kwargs)
        return self._result


class FakeVerdict:
    def __init__(self, allowed: bool, reason: str = ""):
        self.allowed = allowed
        self.reason = reason


class FakeGuardrail:
    """A guardrail the loop has never heard of.

    Note that `check` is a bound method here while the real Guardrail stores a
    plain function. Both are called as `guardrail.check(text)`, which is the
    point - the loop duck-types and does not care.
    """

    def __init__(
        self,
        name: str = "wibble_guard",
        blocks_on=None,
        reason="wibbled",
        raises: Exception | None = None,
    ):
        self.name = name
        self.seen: list[str] = []
        self._blocks_on = blocks_on
        self._reason = reason
        self._raises = raises

    def check(self, text: str) -> FakeVerdict:
        self.seen.append(text)
        # Refusing and raising are different events. A guardrail with a bad
        # regex, or one that calls out to a service that times out, does not
        # return a Verdict at all - it throws, and Day 5 has to record that
        # rather than losing the run. Mirrors FakeTool's `run=` for the same
        # reason: the fake has to be able to fail the way the real thing fails.
        if self._raises is not None:
            raise self._raises
        if self._blocks_on is not None and self._blocks_on in text:
            return FakeVerdict(allowed=False, reason=self._reason)
        return FakeVerdict(allowed=True)
