import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model=os.getenv("OPENROUTER_BASE_MODEL","openrouter/free"), model_provider="openrouter")
message = model.invoke("Hello, how are you?")

print(message.content)
print(message.response_metadata['model_name'])