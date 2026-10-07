# Sequence descreve uma sequência de documentos recebida pela função abaixo.
from collections.abc import Sequence
from typing import Any

# Carrega as variáveis de ambiente, como a chave de API, do arquivo .env.
from dotenv import load_dotenv
# Inicializa o modelo de chat configurado para uso com o provedor OpenRouter.
from langchain.chat_models import init_chat_model
# Document é o tipo de objeto criado pelo divisor de texto do LangChain.
from langchain_core.documents import Document
# Converte a resposta do modelo em uma string simples.
from langchain_core.output_parsers import StrOutputParser
# Monta o prompt que será enviado ao modelo.
from langchain_core.prompts import PromptTemplate
# Adapta uma função Python comum para que ela possa participar de uma chain LCEL.
from langchain_core.runnables import RunnableLambda
# Divide textos longos em trechos menores, com uma sobreposição entre trechos.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Disponibiliza no processo as configurações definidas em .env.
load_dotenv()

# temperature=0 reduz a variação das respostas; o modelo é acessado via OpenRouter.
llm = init_chat_model(model="openrouter/free", model_provider="openrouter", temperature=0)

# Texto de exemplo que será dividido e resumido. O conteúdo pode ser substituído
# por qualquer texto que se queira processar.
long_text = """
Dawn threads a pale gold through the alley of glass.
The city yawns in a chorus of brakes and distant sirens.
Windows blink awake, one by one, like sleepy eyes.
Streetcloth of steam curls from manholes, a quiet river.
Coffee steam spirals above a newspaper's pale print.
Pedestrians sketch light on sidewalks, hurried, loud with umbrellas.
Buses swallow the morning with their loud yawns.
A sparrow perches on a steel beam, surveying the grid.
The subway sighs somewhere underground, a heartbeat rising.
Neon still glows in the corners where night refused to retire.
A cyclist cuts through the chorus, bright with chrome and momentum.
The city clears its throat, the air turning a little less electric.
Shoes hiss on concrete, a thousand small verbs of arriving.
Dawn keeps its promises in the quiet rhythm of a waking metropolis.
The morning light cascades through towering windows of steel and glass,
casting geometric shadows on busy streets below.
Traffic flows like rivers of metal and light,
while pedestrians weave through crosswalks with purpose.
Coffee shops exhale warmth and the aroma of fresh bread,
as commuters clutch their cups like talismans against the cold.
Street vendors call out in a symphony of languages,
their voices mixing with the distant hum of construction.
Pigeons dance between the feet of hurried workers,
finding crumbs of breakfast pastries on concrete sidewalks.
The city breathes in rhythm with a million heartbeats,
each person carrying dreams and deadlines in equal measure.
Skyscrapers reach toward clouds that drift like cotton,
while far below, subway trains rumble through tunnels.
This urban orchestra plays from dawn until dusk,
a endless song of ambition, struggle, and hope.
"""

# Cada trecho terá até 250 caracteres. A sobreposição de 70 caracteres ajuda a
# preservar o contexto que poderia se perder nas fronteiras entre trechos.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=70,
)

# Converte o texto em uma lista de Document, cada um contendo um trecho em
# page_content e podendo carregar também metadados.
parts = splitter.create_documents([long_text])

# O prompt de redução espera uma variável chamada "text", então esta função
# reúne os resumos produzidos na etapa map nesse formato.
def format_summaries(summaries: Sequence[str]) -> dict[str, str]:
    return {"text": "\n\n".join(summaries)}


# Imprime o valor produzido por uma etapa e o devolve sem modificá-lo,
# permitindo observar o fluxo sem interromper a execução da chain.
def log_stage(name: str) -> RunnableLambda:
    def log_value(value: Any) -> Any:
        print(f"\n--- {name} ---")

        if hasattr(value, "to_string"):
            print(value.to_string())
        elif hasattr(value, "content"):
            print(value.content)
        else:
            print(value)

        return value

    return RunnableLambda(log_value)


# Na etapa map, cada documento é resumido de forma independente.
map_prompt = PromptTemplate.from_template(
    "Summarize the following text:\n\n{text}"
)

chain_map = (
    map_prompt
    | log_stage("Prompt de um trecho")
    | llm
    | StrOutputParser()
    | log_stage("Resumo do trecho")
)

# Executa a etapa map para cada trecho; batch processa os documentos sem
# concatená-los antes da sumarização.
mapped_summaries = chain_map.batch(
    [{"text": document.page_content} for document in parts]
)

# Na etapa reduce, os resumos parciais são combinados e resumidos novamente.
reduce_prompt = PromptTemplate.from_template(
    "Create a concise summary from the following partial summaries. "
    "Preserve the main ideas and avoid repetition:\n\n{text}"
)

chain_reduce = (
    RunnableLambda(format_summaries)
    | log_stage("Resumos parciais reunidos")
    | reduce_prompt
    | log_stage("Prompt de redução")
    | llm
    | StrOutputParser()
    | log_stage("Resumo final")
)

# Consolida os resumos parciais em um único resumo final.
result = chain_reduce.invoke(mapped_summaries)