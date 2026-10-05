"""Run the loop once by hand so you can see what it does. Not part of the loop."""

from dotenv import load_dotenv

from src.agent import run_agent
from src.guardrails import DEFAULT_INPUT_GUARDRAILS, DEFAULT_OUTPUT_GUARDRAILS
from src.tools import DEFAULT_TOOLS

load_dotenv()

# Needs both tools: the invoice is in EUR, the question asks for USD. Nothing
# tells the agent to use them in that order except the tool descriptions.
result = run_agent(
    "What is invoice INV-1043 worth in US dollars?",
    tools=DEFAULT_TOOLS,
    input_guardrails=DEFAULT_INPUT_GUARDRAILS,
    output_guardrails=DEFAULT_OUTPUT_GUARDRAILS,
)

print(f"stop reason:  {result.stop_reason.value}")
print(f"cycles used:  {result.cycles}")
print(f"turns:        {len(result.conversation)}")
print(f"final answer: {result.final_answer}")

if result.blocked_by is not None:
    print(f"blocked by:   {result.blocked_by}")
    print(f"because:      {result.block_reason}")

print("\ntools called:")
for message in result.conversation:
    if isinstance(message["content"], list):
        for block in message["content"]:
            if block["type"] == "tool_use":
                print(f"  {block['name']}({block['input']})")
