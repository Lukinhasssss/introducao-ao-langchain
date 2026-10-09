import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(model=os.getenv("OPENROUTER_BASE_MODEL","openrouter/free"), temperature=0.5)
message = model.invoke("Hello, how are you?")

print(message.content)
print(message.response_metadata['model_name'])