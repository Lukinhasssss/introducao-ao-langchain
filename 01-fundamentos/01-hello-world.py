from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(model="openrouter/free", temperature=0.5)
message = model.invoke("Hello, how are you?")

print(message.content)
print(message.response_metadata['model_name'])