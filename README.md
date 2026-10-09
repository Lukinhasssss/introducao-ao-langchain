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

## Gerenciamento de Memória

- Memória se refere a guardar o contexto e todo o conteúdo que foi informado para o modelo
- LLMs são Stateless -> Nào guardam informação
- "Conversas" -> Não são armazenadas no modelo
- É necessário enviar 100% do histórico de mensagens

### Histórico de Curto Prazo vs Longo Prazo

- Curto Prazo: Memória que é utilizada durante uma transação/conversa/processamento
  - Pode ser armazenada temporáriamente (banco de dados de cache, memória do sevidor),
  - Pode ser armazenada em um banco de dados vetorial
- Longo Prazo: Histórico da "conversa"
  - Deve ser armazenada em um banco de dados vetorial
  - Possibilidade de restaurar o conteúdo anterior para continuar de onde parou
  - Ler o histórico para ter contexto para continuar a "conversa"
  - Sumarização (resumo da conversa) como estratégia para economizar tokens

## Document Loaders & RAG

- Document Loaders: Carregam documentos de diferentes fontes (PDF, HTML, CSV, JSON, Sites, etc)
- RAG: Retrieval-Augmented Generation
- RAG é uma técnica que combina a geração de texto com a recuperação de informações relevantes de uma base de dados ou documentos. Isso permite que o modelo de linguagem produza respostas mais precisas e informadas, utilizando dados externos como referência.
- Exemplo de RAG:
  - Usuário pergunta: "Qual é a capital da França?"
  - O sistema RAG busca em uma base de dados ou documentos e encontra a informação correta: "A capital da França é Paris."
  - O modelo de linguagem então gera uma resposta informada: "A capital da França é Paris."
  - Isso é especialmente útil em cenários onde o modelo de linguagem precisa fornecer respostas baseadas em informações atualizadas ou específicas que não estão presentes em seu treinamento inicial.
- RAG é amplamente utilizado em sistemas de perguntas e respostas, assistentes virtuais, chatbots e outras aplicações que requerem acesso a informações externas para fornecer respostas precisas e contextualmente relevantes.
- Exemplo de RAG com LangChain:
  - Carregar documentos usando Document Loaders
  - Criar embeddings dos documentos carregados
  - Armazenar os embeddings em um banco de dados vetorial
  - Quando o usuário faz uma pergunta, buscar os documentos mais relevantes no banco de dados vetorial
  - Passar os documentos relevantes para o modelo de linguagem para gerar uma resposta informada
- Exemplo de código para RAG com LangChain:

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Carregar o documento PDF
loader = PyPDFLoader("documento.pdf")
document = loader.load()

# Dividir o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

# Criar embeddings para cada chunk e armazená-los em um banco de dados vetorial
vectorstore = FAISS.from_documents(chunks, embedding_model)
```

## Chunking de Documentos

- Chunking é o processo de dividir documentos longos em partes menores (chunks) para facilitar o processamento e a análise.
- Cada chunk pode ser processado individualmente, permitindo que o modelo de linguagem lide com grandes volumes de texto sem perder contexto ou informações importantes.
- Técnicas de chunking:
  - **Fixed Size Chunking:** Dividir o documento em chunks de tamanho fixo (ex: 500 tokens)
  - **Semantic Chunking:** Dividir o documento com base em significado ou tópicos, utilizando técnicas de NLP para identificar limites naturais no texto
  - **Overlap Chunking:** Criar chunks que se sobrepõem parcialmente, garantindo que informações importantes não sejam perdidas entre os chunks
- Vantagens do chunking:
  - Facilita o processamento de grandes documentos
  - Permite a aplicação de técnicas de RAG, onde cada chunk pode ser indexado e recuperado individualmente
  - Melhora a eficiência do modelo de linguagem ao lidar com grandes volumes de texto, evitando a perda de contexto e informações importantes
  - Permite a sumarização de documentos longos, onde cada chunk pode ser resumido individualmente antes de gerar um resumo final do documento completo
  - Permite a criação de embeddings para cada chunk, facilitando a busca por similaridade em bancos de dados vetoriais
- Exemplo de chunking com LangChain:
  - Carregar um documento PDF usando um Document Loader
  - Dividir o documento em chunks utilizando um Text Splitter
  - Criar embeddings para cada chunk e armazená-los em um banco de dados vetorial
  - Quando o usuário faz uma pergunta, buscar os chunks mais relevantes no banco de dados vetorial
  - Passar os chunks relevantes para o modelo de linguagem para gerar uma resposta informada
- Exemplo de código para chunking com LangChain:

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carregar o documento PDF
loader = PyPDFLoader("documento.pdf")
document = loader.load()

# Dividir o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)
```
