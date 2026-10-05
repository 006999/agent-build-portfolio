import inspect

import pytest

from src import tools
from src.tools import (
    CONVERT_CURRENCY_TOOL,
    DEFAULT_TOOLS,
    LOOKUP_INVOICE_TOOL,
    convert_currency,
    lookup_invoice,
)


# ---------------------------------------------------------------------------
# lookup_invoice
# ---------------------------------------------------------------------------


def test_lookup_returns_the_row_for_a_known_invoice():
    line = lookup_invoice("INV-1043")

    assert "INV-1043" in line
    assert "Placeholder Freight Ltd" in line
    assert "4820.00 EUR" in line
    assert "2026-06-14" in line
    assert "unpaid" in line


def test_lookup_raises_on_an_unknown_invoice_rather_than_returning_nothing():
    # The quiet version of this bug returns "" or None, the model treats the
    # absence as an answer, and the run reports on an invoice that never existed.
    with pytest.raises(KeyError):
        lookup_invoice("INV-9999")


def test_lookup_does_not_match_a_partial_id():
    with pytest.raises(KeyError):
        lookup_invoice("INV-104")


# ---------------------------------------------------------------------------
# convert_currency
# ---------------------------------------------------------------------------

CANNED_BODY = {
    "amount": 1.0,
    "base": "EUR",
    "date": "2026-07-29",
    "rates": {"USD": 1.138},
}


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the network call. Records the URL that was asked for."""
    calls = []

    def _fake(url: str) -> dict:
        calls.append(url)
        return CANNED_BODY

    monkeypatch.setattr(tools, "_fetch_json", _fake)
    return calls


def test_convert_reads_the_rate_and_does_its_own_arithmetic(fake_fetch):
    line = convert_currency("EUR", "USD", 4820.0)

    # 4820 * 1.138 = 5485.16. Asking the API to convert returns 5485 - it
    # rounds. The multiplication stays here so the cents survive.
    assert "5,485.16 USD" in line
    assert "4,820.00 EUR" in line
    assert "1.138" in line
    assert "2026-07-29" in line


def test_convert_asks_for_only_the_one_currency_it_needs(fake_fetch):
    convert_currency("EUR", "USD", 100.0)

    # Every character a tool returns is re-sent on every later cycle. The
    # unrestricted response lists about thirty currencies.
    assert "symbols=USD" in fake_fetch[0]
    assert "base=EUR" in fake_fetch[0]


def test_convert_accepts_lowercase_codes(fake_fetch):
    line = convert_currency("eur", "usd", 100.0)

    assert "100.00 EUR" in line
    assert "113.80 USD" in line


def test_convert_raises_when_the_response_lacks_the_requested_currency(monkeypatch):
    monkeypatch.setattr(
        tools, "_fetch_json", lambda url: {"amount": 1.0, "date": "x", "rates": {}}
    )

    with pytest.raises(KeyError):
        convert_currency("EUR", "USD", 100.0)


# ---------------------------------------------------------------------------
# The Tool shape itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", DEFAULT_TOOLS, ids=lambda tool: tool.name)
def test_the_schema_and_the_function_signature_agree(tool):
    """The drift the Tool dataclass exists to prevent.

    A renamed argument on one side and not the other is a TypeError at
    tool.run(**block.input) - at runtime, after Claude has already decided.
    """
    parameters = inspect.signature(tool.run).parameters
    without_defaults = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
    }

    assert set(tool.input_schema["properties"]) == set(parameters)
    assert set(tool.input_schema["required"]) == without_defaults


@pytest.mark.parametrize("tool", DEFAULT_TOOLS, ids=lambda tool: tool.name)
def test_every_tool_satisfies_the_contract_the_loop_relies_on(tool):
    # The loop tests use hand-written fakes to prove it does not know these
    # tools. This is the test that makes sure the real ones fit the same shape.
    assert isinstance(tool.name, str) and tool.name
    assert callable(tool.run)
    assert set(tool.to_api_format()) == {"name", "description", "input_schema"}


@pytest.mark.parametrize("tool", DEFAULT_TOOLS, ids=lambda tool: tool.name)
def test_every_description_says_when_not_to_use_the_tool(tool):
    # The clause that draws the boundary from both sides. Losing it in an edit
    # is silent: the tools still work, they just get picked wrongly.
    assert "Do not use this when" in tool.description


def test_the_two_tools_have_different_names():
    assert LOOKUP_INVOICE_TOOL.name != CONVERT_CURRENCY_TOOL.name
