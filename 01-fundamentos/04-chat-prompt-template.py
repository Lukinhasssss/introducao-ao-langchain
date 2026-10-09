import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

system = ("system", "you're an assistant that answer questions in a {style} style")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="funny", question="Who is Clóvis Basílio?")

for message in messages:
  print(f"{message.type}: {message.content}")

model = init_chat_model(model=os.getenv("OPENROUTER_BASE_MODEL","openrouter/free"), model_provider="openrouter", temperature=0)
result = model.invoke(messages)

print(result.content)
print(result.response_metadata['model_name'])