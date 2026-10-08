from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

chat_model = init_chat_model(
  model="openrouter/free",
  model_provider="openrouter",
  temperature=0,
)

agent = create_agent(
  model=chat_model,
  tools=[],
  system_prompt="You are a helpful assistant.",
  checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "demo-session"}}

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

# Como o historico e armazenado e carregado:
# InMemorySaver guarda o estado da conversa na memoria do processo, incluindo
# as mensagens enviadas pelo usuario e as respostas do assistente.
# O thread_id identifica essa conversa. Em cada invoke, o agente usa esse id
# para recuperar o estado anterior, acrescenta a nova mensagem e salva o estado
# atualizado. Assim, o modelo recebe o historico junto com a nova mensagem.
# Como o armazenamento e apenas em memoria, o historico desaparece quando o
# processo termina. Para preserva-lo entre execucoes, use um checkpointer
# persistente, como os oferecidos pelo LangGraph.