"""The two tools the agent can call. Day 2: no retries, no error handling.

Nothing here imports the agent loop, and the loop imports nothing from here.
That one-way dependency is the point: on Day 6 this file gets swapped for a
real client workflow and src/agent.py should not need a single line changed.
"""

import csv
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

INVOICE_FILE = Path(__file__).resolve().parent.parent / "data" / "invoices.csv"

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"
FETCH_TIMEOUT_SECONDS = 10
USER_AGENT = "fde-build/0.1"


@dataclass(frozen=True)
class Tool:
    """One tool: the schema Claude reads and the function Python runs, together.

    Keeping these in one object is deliberate. Passing them as two parallel
    lists lets them drift, and drift does not show up until Claude sends an
    argument the function does not accept — at runtime, in front of a client.
    """

    name: str
    description: str
    input_schema: dict
    run: Callable[..., str]

    def to_api_format(self) -> dict:
        """The shape the Messages API wants. `run` is ours and does not go."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# Tool one: read one invoice from a file in this repo.
# ---------------------------------------------------------------------------


def _read_invoices() -> list[dict]:
    """Read the whole CSV, fresh, every call. Eight rows, so no caching.

    Day 4 asked whether to cache this and the answer was no. An lru_cache here
    is module-level state that outlives every run in the process, and the moment
    invoices.csv is per-client that is a cross-client read. It is also the one
    kind of leak a test on what gets sent to the model cannot see: a cached row
    and a fresh row are identical bytes, so the request looks correct and is
    stale. Pinned by test_a_tool_reads_the_file_rather_than_remembering_it in
    tests/test_memory.py, which is mutation-verified.
    """
    with INVOICE_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lookup_invoice(invoice_id: str) -> str:
    """Return one invoice as a single line of text.

    Raises KeyError when there is no such invoice. That crashes the run today,
    which is the intended behaviour: a guessed invoice ID should be loud.
    """
    for row in _read_invoices():
        if row["invoice_id"] == invoice_id:
            return (
                f"{row['invoice_id']} | {row['client']} | "
                f"{row['amount']} {row['currency']} | "
                f"issued {row['issue_date']} | {row['status']}"
            )

    raise KeyError(f"No invoice with id {invoice_id!r} in {INVOICE_FILE.name}")


LOOKUP_INVOICE_DESCRIPTION = (
    "Look up one invoice from the local invoice records by its invoice ID. "
    "Returns the client name, the amount, the currency the invoice was issued "
    "in, the issue date, and whether it has been paid.\n\n"
    "Use this whenever you need a fact about one specific invoice - who it is "
    "for, how much it is for, what currency it is in, or whether it is paid. "
    "Invoice IDs look like INV-1043.\n\n"
    "Do not use this when the question is about exchange rates, converting an "
    "amount into another currency, or a total across several invoices. This "
    "tool reads exactly one invoice and does no arithmetic. Do not guess an "
    "invoice ID - if you have not been given one, say so instead of inventing "
    "one."
)


# ---------------------------------------------------------------------------
# Tool two: fetch today's exchange rate from the public web.
# ---------------------------------------------------------------------------


def _fetch_json(url: str) -> dict:
    """Fetch and parse one JSON body.

    A separate function so tests can replace it and never touch the network.
    A non-200 response raises urllib.error.HTTPError, which today ends the run.

    The User-Agent is not optional. Left to itself urllib sends
    "Python-urllib/3.12", and the service answers that with 403.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def convert_currency(from_currency: str, to_currency: str, amount: float) -> str:
    """Convert an amount using today's European Central Bank reference rate."""
    from_code = from_currency.upper()
    to_code = to_currency.upper()

    # Only one symbol is requested. The unrestricted response lists about
    # thirty currencies, and every one of them would be re-sent to the model on
    # every later cycle of the run.
    query = urllib.parse.urlencode({"base": from_code, "symbols": to_code})
    body = _fetch_json(f"{FRANKFURTER_URL}?{query}")

    # The rate is fetched for one unit and multiplied here rather than asking
    # the API to convert. Passing an amount makes the API round its answer:
    # 4820 EUR came back as 5485 rather than 5485.16.
    rate = body["rates"][to_code]
    converted = amount * rate

    return (
        f"{amount:,.2f} {from_code} = {converted:,.2f} {to_code} "
        f"(rate {rate}, European Central Bank, {body['date']})"
    )


CONVERT_CURRENCY_DESCRIPTION = (
    "Convert an amount of money from one currency into another, using today's "
    "published European Central Bank reference rate fetched live over the "
    "internet. Returns the converted amount, the rate used, and the date that "
    "rate is from.\n\n"
    "Use this whenever an amount needs to be expressed in a different currency "
    "than the one it is stated in. Currencies are three-letter codes such as "
    "USD, EUR, GBP. The rate is fetched live, so use this instead of any rate "
    "you already know - a remembered rate is out of date.\n\n"
    "Do not use this when the two currencies are the same, when there is no "
    "specific amount in question, or when the question asks about a rate on a "
    "past date. This tool returns today's rate only. It does not know anything "
    "about invoices - if you need an amount from an invoice, get it first."
)


# ---------------------------------------------------------------------------
# The registry handed to run_agent. `additionalProperties: false` tells the
# model the argument list is closed, so it is less likely to invent an extra
# key. It shapes generation; it does not guarantee anything.
# ---------------------------------------------------------------------------

LOOKUP_INVOICE_TOOL = Tool(
    name="lookup_invoice",
    description=LOOKUP_INVOICE_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "The invoice ID, for example INV-1043.",
            }
        },
        "required": ["invoice_id"],
        "additionalProperties": False,
    },
    run=lookup_invoice,
)

CONVERT_CURRENCY_TOOL = Tool(
    name="convert_currency",
    description=CONVERT_CURRENCY_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {
            "from_currency": {
                "type": "string",
                "description": "Three-letter code of the currency to convert from, e.g. EUR.",
            },
            "to_currency": {
                "type": "string",
                "description": "Three-letter code of the currency to convert to, e.g. USD.",
            },
            "amount": {
                "type": "number",
                "description": "How much of from_currency to convert.",
            },
        },
        "required": ["from_currency", "to_currency", "amount"],
        "additionalProperties": False,
    },
    run=convert_currency,
)

DEFAULT_TOOLS = [LOOKUP_INVOICE_TOOL, CONVERT_CURRENCY_TOOL]
