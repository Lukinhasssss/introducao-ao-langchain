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

## Principais Tópicos Abordados no Projeto

### Fundamentos Básicos

- **Introdução ao LangChain:** Conceitos básicos e primeira integração com modelos de linguagem
- **Integração com OpenAI:** Como usar o ChatOpenAI para interações básicas
- **Integração com Google Gemini:** Configuração e uso do modelo Gemini via Google GenAI
- **Templates de Prompt:** Criação de templates simples e avançados para prompts
- **Chat Prompts:** Implementação de sistemas de conversação com templates estruturados

### Chains e Processamento Avançado

- **LCEL (LangChain Expression Language):** Sintaxe de pipe (|) para conectar componentes
- **Chains Básicas:** Criação de pipelines simples de pergunta e resposta
- **Chains Complexas:** Pipelines multi-etapas com tradução e sumarização
- **Runnables Customizados:** Decorador @chain e RunnableLambda para funções personalizadas
- **Text Splitters:** Divisão de textos longos em chunks para processamento
- **Summarization Chains:** Técnicas "stuff" e "map_reduce" para sumarização de documentos
- **Structured Output:** Extração de dados estruturados usando Pydantic models

### Agentes e Tools

- **Custom Tools:** Criação de ferramentas personalizadas com decorador @tool
- **ReAct Agents:** Implementação de agentes que raciocinam e agem
- **Agent Executors:** Orquestração de agentes com múltiplas ferramentas
- **Prompt Hub:** Uso de prompts pré-definidos da comunidade LangChain

### Gerenciamento de Memória

- **Chat History:** Implementação de histórico de conversação persistente
- **Message Memory:** Armazenamento e recuperação de mensagens em sessões
- **Memory Trimming:** Técnicas de janela deslizante para gerenciar contexto limitado
- **Session Management:** Gerenciamento de múltiplas sessões de conversa

### Document Loaders e RAG

- **Web Loaders:** Carregamento de conteúdo de páginas web
- **PDF Loaders:** Processamento e extração de texto de documentos PDF
- **Document Chunking:** Estratégias de divisão de documentos para processamento eficiente
- **RAG Foundations:** Conceitos básicos para Retrieval-Augmented Generation
