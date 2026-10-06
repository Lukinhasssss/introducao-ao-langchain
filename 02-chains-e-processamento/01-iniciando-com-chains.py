from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

load_dotenv()

question_template = PromptTemplate(
  input_variables=["name"],
  template="Hi, I'm {name}! Tell me a joke with my name!"
)

model = init_chat_model(model="openrouter/free", model_provider="openrouter", temperature=0)

# A saída do question_template vai para a entrada do model
chain = question_template | model

result = chain.invoke({"name": "Lukinhas"})

print(result.content)
print(result.response_metadata['model_name'])