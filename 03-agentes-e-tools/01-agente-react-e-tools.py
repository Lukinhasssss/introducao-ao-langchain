import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

@tool(name_or_callable="calculador", return_direct=True)
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
        "Never search the internet; only use the tools provided."
    ),
)

print(
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is the capital of Iran?"}]}
    )["messages"][-1].content
)
print(
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is the capital of Brazil?"}]}
    )["messages"][-1].content
)
print(
    agent.invoke(
        {"messages": [{"role": "user", "content": "How much is 10 + 10?"}]}
    )["messages"][-1].content
)