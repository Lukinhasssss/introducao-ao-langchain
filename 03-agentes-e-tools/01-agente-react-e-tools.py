from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

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