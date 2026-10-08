from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

load_dotenv()

chat_model = init_chat_model(
  model="openrouter/free",
  model_provider="openrouter",
  temperature=0,
)

WINDOW_SIZE = 5


@before_model
def apply_sliding_window(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  _ = runtime
  messages = state["messages"]
  if len(messages) <= WINDOW_SIZE:
    return None

  # Antes da chamada ao modelo, a ultima mensagem costuma ser a nova pergunta.
  # Mantemos essa pergunta e as mensagens mais recentes dentro do limite.
  return {
    "messages": [
      RemoveMessage(id=REMOVE_ALL_MESSAGES),
      *messages[-WINDOW_SIZE:],
    ]
  }


agent = create_agent(
  model=chat_model,
  tools=[],
  system_prompt="You are a helpful assistant.",
  middleware=[apply_sliding_window],
  checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "sliding-window-session"}}

# Interactions
response1 = agent.invoke(
  {"messages": [{"role": "user", "content": "Hello, my name is Lucas. how are you?"}]},
  config=config,
)
print("Assistant: ", response1["messages"][-1].content)
print("-"*30)

response2 = agent.invoke(
  {"messages": [{"role": "user", "content": "Can you repeat my name?"}]},
  config=config,
)
print("Assistant: ", response2["messages"][-1].content)
print("-"*30)

response3 = agent.invoke(
  {"messages": [{"role": "user", "content": "Can you repeat my name in a motivation phrase?"}]},
  config=config,
)
print("Assistant: ", response3["messages"][-1].content)
print("-"*30)

response4 = agent.invoke(
  {"messages": [{"role": "user", "content": "Can you repeat my name again?"}]},
  config=config,
)
print("Assistant: ", response4["messages"][-1].content)
print("-"*30)

# Como funciona a janela de contexto (sliding window):
# O middleware e executado antes de cada chamada ao modelo. Se o historico
# ultrapassar WINDOW_SIZE, ele remove as mensagens antigas e mantem as mais
# recentes. Neste exemplo, o modelo recebe no maximo cinco mensagens por vez;
# mensagens removidas tambem deixam de fazer parte do estado salvo da conversa.
# InMemorySaver guarda esse estado atualizado durante o processo, mas nao o
# preserva depois que o processo termina.



# Não use @before_model retornando RemoveMessage(REMOVE_ALL_MESSAGES) para esse caso.
# É o padrão que a doc mostra, mas ele apaga as mensagens do estado de forma permanente.
# O wrap_model_call é o que preserva o histórico completo.