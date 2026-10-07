import math

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain

load_dotenv()

llm = init_chat_model(model="openrouter/free", model_provider="openrouter", temperature=0)

template_translate = PromptTemplate(
  input_variables=["initial_text"],
  template="Translate the following text to English:\n```{initial_text}```"
)

template_summary = PromptTemplate(
  input_variables=["text"],
  template="Translate the following text in 4 words:\n```{text}```\n\n"
)

"""
- template_translate: gera o prompt a partir do texto de entrada
- model: envia esse prompt para o modelo de IA
- StrOutputParser(): converte a resposta do modelo em uma string simples, pronto para ser utilizado no formato de Runnable, ou seja, pode ser utilizado dentro de uma pipeline de outra chamada
"""
text_translated = template_translate | llm | StrOutputParser()
pipeline = {"text": text_translated} | template_summary | llm | StrOutputParser()

result = pipeline.invoke({"initial_text": "LangChain é um framework para desenvolvimento de aplicacões de IA"})

print(f"Texto traduzido: {text_translated}\n")
print(f"Pipeline: {pipeline}\n")
print(result)