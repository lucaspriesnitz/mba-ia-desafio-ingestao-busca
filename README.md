# Ingestão e Busca Semântica com LangChain e Postgres

Sistema RAG (Retrieval-Augmented Generation) que ingere um PDF, armazena os embeddings no PostgreSQL com pgvector, e responde perguntas com base no conteúdo do documento usando Google Gemini.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave de API do Google AI (Gemini)

## Como executar

### 1. Configurar variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e preencha a `GOOGLE_API_KEY`:

```bash
cp .env.example .env
```

### 2. Subir o banco de dados

```bash
docker compose up -d
```

Isso inicia o PostgreSQL com a extensão pgvector habilitada.

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Ingerir o PDF

```bash
python src/ingest.py
```

Esse comando carrega o PDF, divide em chunks, gera embeddings e armazena no banco.

### 5. Iniciar o chat

```bash
python src/chat.py
```

Digite suas perguntas sobre o conteúdo do documento. Digite `sair` para encerrar.
