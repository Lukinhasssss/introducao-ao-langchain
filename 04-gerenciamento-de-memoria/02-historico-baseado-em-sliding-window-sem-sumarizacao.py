from typing import Callable, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (ModelRequest, ModelResponse,
                                         wrap_model_call)
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@wrap_model_call
def trim_context(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
  trimmed = trim_messages(
    request.messages,
    token_counter=len,   # conta mensagens, não tokens
    max_tokens=3,        # par anterior (human + AI) + mensagem atual
    strategy="last",
    start_on="human",
    allow_partial=False,
  )
  return handler(request.override(messages=cast(list[AnyMessage], trimmed)))


chat_model = init_chat_model(
  model="openrouter/free",
  model_provider="openrouter",
  temperature=0,
)

agent = create_agent(
  model=chat_model,
  system_prompt="You're a helpful assistant that answers with a short joke when possible.",
  middleware=[trim_context],
  checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "demo-session"}}


def ask(text: str) -> str:
  result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
  return result["messages"][-1].content


print("Assistant:", ask("My name is Lucas. Reply only with 'OK' and do not mention my name."))
print("Assistant:", ask("Tell me a one-sentence fun fact. Do not mention my name."))
print("Assistant:", ask("What is my name?"))

# Não use @before_model retornando RemoveMessage(REMOVE_ALL_MESSAGES) para esse caso.
# É o padrão que a doc mostra, mas ele apaga as mensagens do estado de forma permanente.
# O wrap_model_call é o que preserva o histórico completo.