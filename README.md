# Langchain

## Criação

- Outubro de 2022
- Criado por Harrison Chase
- Abril de 2023
  - Formalizou a startup
  - Investimento inicial levantado (Series A): ~= US$ 35mi
  - Investimento (Series B): 100mi. Valor estimado da empresa 1.1bi
- Fevereiro de 2024: Lançamento do LangSmith
- Maio de 2025: LangGraph Platform

## Ecossistema de serviços

### LangSmith

- Observabilidade, debugging
- Monitoramento em produção
- Avaliação de performance de modelos
- Custos e latência

### LangServe Hosted

- Plataforma para deploy de aplicações LangChain
- Infra gerenciada
- Escalabilidade automática

### LangGraph Platform

- Infraestrutura para agentes de IA
- Agentes de longa duração
- Gerenciamento de workflows completos

### LangGraph Studio

- Interface para gerenciamento de projeto LangGraph

### LangChain Hub:

- Catálogo para publicar, versionar, testar e baixar prompts e outros artefatos
- Faz parte do LangSmith, com playground e SDK para pull/push

## Sobre o LangChain

- Simplifica a integração com LLms e serviços auxiliares:
  ex: carregamento de dados, memória, busca por documentos, agentes, roteadores.
- Fornece um conjunto de abstrações que facilitam a construção de "chains" (Fluxos de execução de chamadas e transformações)
- Evoluiu MUITO rápido entre 2023 e 2024
- Foi separada em pacotes específicos ao longo do tempo:
  - langchain-core, langchain (implementação de referência)
  - langchain-community (integrações extras)
  - pacotes de terceiros

## Principais Recursos

- Criação de chains: Fluxos de execução compostos por etapas (runnables). Cada etapa pode chamar modelos, processar dados, etc
- LangChain Expression Language (LCEL): Realiza essas operações utilizando "pipe" "|" para compor runnables
- Carregamento e divisão de documentos: Consegue carregar CSVs, JSON, HTML, Markdown, Sites, PDFs, etc.
- Modelos de Embedding e Armazenamento Vetorial: Interface para gerar embeddings com OpenAI, HuggingFace, etc e armazenar vetores em bancos como Pinecone, PGVector (postgres), Weaviate, FAISS...
- Busca por similaridade em bancos de dados vetoriais.
- Agentes e Ferramentas: Agentes usam a LLM para processar a entrada e conseguem decidir quais ações/ferramentas executar.
- Memória e histórico
- Prompt Templates e placeholders
- OutputParsing e Pydantic
- Sumarização, utilizando map-reduce
