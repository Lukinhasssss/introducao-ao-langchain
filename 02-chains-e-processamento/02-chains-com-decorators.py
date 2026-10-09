import math
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain

load_dotenv()

@chain
def square_root(input: dict) -> dict:
  x = input["x"]
  return {"square_result": math.sqrt(x)}

model = init_chat_model(model=os.getenv("OPENROUTER_BASE_MODEL","openrouter/free"), model_provider="openrouter", temperature=0)

question_template = PromptTemplate(
  input_variables=["square_result"],
  template="Tell me abot the number {square_result}"
)

chain = square_root | question_template | model

result = chain.invoke({"x":144})

print(result.content)
print(result.response_metadata['model_name'])