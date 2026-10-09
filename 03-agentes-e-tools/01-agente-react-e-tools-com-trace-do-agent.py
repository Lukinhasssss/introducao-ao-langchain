import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

@tool(name_or_callable="calculador")
def calculator(expression: str) -> str:
  """Evaluate a simple mathematical expression and returns the result."""
  try:
    result = eval(expression) # be careful with this because it's a security risk
  except Exception as ex:
    return f"Error: {ex}"
  return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C."
    }

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."

    return "I don't know the capital of that country."

llm = init_chat_model(
  model=os.getenv("OPENROUTER_BASE_MODEL","openrouter/free"),
  model_provider="openrouter",
  temperature=0,
  disable_streaming=True
)

tools = [calculator, web_search_mock]

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[ModelCallLimitMiddleware(run_limit=3)],
    system_prompt=(
        "Answer questions using only the information returned by the provided tools. "
        "If the tools do not provide the answer, say you don't know. "
        "Never search the internet; only use the tools provided. "
        "Format your final response with these sections: "
        "'Thought' as one brief, user-facing summary of your approach (never reveal "
        "private chain-of-thought), 'Action' and 'Action Input' for each tool used, "
        "'Observation' with the corresponding tool result, and 'Final Answer'. "
        "Only report actions and observations that actually occurred; do not invent them."
    ),
)


def run_agent(question: str) -> None:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    print(f"\nQuestion: {question}")
    print("Tool execution trace:")
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                print(f"Action: {tool_call['name']}")
                print(f"Action Input: {tool_call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"Observation: {message.content}")

    print("\nAgent response:")
    print(result["messages"][-1].content)


run_agent("What is the capital of Iran?")
run_agent("What is the capital of Brazil?")
run_agent("How much is 10 + 10?")