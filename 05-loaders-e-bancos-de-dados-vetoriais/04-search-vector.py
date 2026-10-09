import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from pydantic import SecretStr

load_dotenv()

def require_env(name: str, default: str | None = None) -> str:
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



query = "Tell me more about the gpt-5 thinking evaluation and performance results comparing to gpt-4"

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=SecretStr(OPENROUTER_API_KEY),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"X-Title": "introducao-ao-langchain"},
    check_embedding_ctx_length=False,
)

engine = PGEngine.from_connection_string(url=PGVECTOR_URL)

store = PGVectorStore.create_sync(
    engine=engine,
    table_name=PGVECTOR_TABLE,
    embedding_service=embeddings,
)

results = store.similarity_search_with_score(query, k=10)

for i, (doc, score) in enumerate(results, start=1):
    print("=" * 50)
    print(f"Resultado {i} (score: {score:.2f}):")
    print("=" * 50)

    print("\nTexto:\n")
    print(doc.page_content.strip())

    print("\nMetadados:\n")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")