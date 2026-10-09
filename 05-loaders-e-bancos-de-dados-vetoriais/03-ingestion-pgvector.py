import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

# Carrega as configuracoes do arquivo .env para que os.getenv consiga le-las.
load_dotenv()

def require_env(name: str, default: str | None = None) -> str:
    """Retorna uma variavel de ambiente ou falha se ela for obrigatoria e estiver ausente."""
    value = os.getenv(name)
    if value:
        return value
    if default is not None:
        return default
    raise RuntimeError(f"Environment variable {name} is not set")

OPENROUTER_API_KEY = require_env("OPENROUTER_API_KEY")
PGVECTOR_URL = require_env("PGVECTOR_URL")
PGVECTOR_TABLE = require_env("PGVECTOR_TABLE")
EMBEDDING_MODEL = require_env("OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")



# O PDF fica na mesma pasta deste script, independentemente da pasta de onde
# o comando for executado.
current_dir = Path(__file__).parent
pdf_path = current_dir / "gpt5.pdf"

# Carrega o PDF como uma lista de Documentos. O loader normalmente cria um
# documento por pagina, com o texto em page_content e dados da pagina em metadata.
docs = PyPDFLoader(str(pdf_path)).load()

# Divide os textos em trechos menores (chunks), que serao vetorizados e
# pesquisados individualmente. A sobreposicao ajuda a manter contexto entre
# trechos consecutivos quando uma ideia fica na fronteira entre eles.
splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=False,
).split_documents(docs)

# Nao ha nada para inserir se o PDF nao produziu texto/trechos.
if not splits:
    raise SystemExit(0)

# Recria cada Document filtrando metadados vazios, para evitar guardar valores
# sem utilidade junto com o texto. O conteudo e os demais metadados sao mantidos.
enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
    )
    for d in splits
]

# Gera um UUID deterministico para cada trecho usando o nome do PDF e sua
# posicao na lista. O ID e reproduzivel quando nome e posicao se mantem; se o
# conteudo do PDF mudar sem mudar esses dados, o mesmo ID pode passar a identificar
# um texto diferente.
ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.name}-{i}")) for i in range(len(enriched))]
# ids = [f"doc-{i}" for i in range(len(enriched))]

# Configura o modelo remoto que transforma cada trecho de texto em um vetor
# numerico (embedding). A chave e enviada como segredo, e a URL aponta para
# a API compativel do OpenRouter.
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=SecretStr(OPENROUTER_API_KEY),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"X-Title": "introducao-ao-langchain"},  # ajuda a identificar a origem das requisições
    check_embedding_ctx_length=False, # evita erro de limite de tokens, pois o modelo de embedding nao tem limite
)

# Calcula o tamanho do embedding para garantir que a tabela do pgvector seja criada com a dimensao correta.
VECTOR_SIZE = len(embeddings.embed_query("teste"))
print(f"Embedding size: {VECTOR_SIZE}")

# Cria o mecanismo de acesso ao PostgreSQL/pgvector usando a URL de conexao.
engine = PGEngine.from_connection_string(url=PGVECTOR_URL)

# Inicializa a tabela vetorial com a dimensao compativel com o modelo configurado.
# Essa dimensao precisa corresponder ao tamanho dos embeddings gerados.
engine.init_vectorstore_table(
    table_name=PGVECTOR_TABLE,
    vector_size=VECTOR_SIZE,
    overwrite_existing=True,  # apaga e recria a tabela se já existir
)

# Associa a tabela ao modelo de embeddings; o vector store usa essa associacao
# tanto para converter os documentos em vetores quanto para futuras buscas.
store = PGVectorStore.create_sync(
    engine=engine,
    table_name=PGVECTOR_TABLE,
    embedding_service=embeddings,
)

# Insere os documentos enriquecidos. O vector store gera os embeddings e grava
# os vetores junto com o texto, os metadados e os IDs correspondentes.
store.add_documents(documents=enriched, ids=ids)