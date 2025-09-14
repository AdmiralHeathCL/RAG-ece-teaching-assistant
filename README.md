## Project Overview

This project is a Retrieval-Augmented Generation (RAG)–based Q&A system focused on ECE textbooks. The system is structured as follows:

* **Vector Database**: Textbook contents are chunked and embedded into vectors, then stored in Milvus.
* **Q&A Interaction**: Convert user questions into vector queries, search the document collections, and answer based on retrieved passages.
* **OpenAI-compatible Endpoints**: FastAPI services that can connect into AI Agents and support multiple LLMs.

## System Architecture

### **Database**

#### setup.py

* **Token-based paragraph chunking**: Split documents into segments according to token limits.
* **Vector storage**: Use an embedding model to generate semantic vectors for each chunk.
* **Tag extraction**：Extract `image_url` and `chapter` from each chunk and save them into the chunk’s metadata.

### **Interaction**

#### 1. chatbot.py

* **Preprocessing**: Use regex to detect `chapter` names.
* **Vector retrieval**: Milvus `collection_tb` with HNSW index.
* **SelfQueryRetriever**: Filter documents using metadata `chapter`.
* **RetrievalQA**: Build prompts to the LLM and return any related image links.

#### FastAPI

`/v1/chat/completions`: Accepts OpenAI-format requests and routes to the corresponding model.

* **proxy_app**:

  `/v1/models`：returns single `textbook` model

* **proxy_app_multimodel**:

  `/v1/models`：returns multiple models using different LLMs

### How to Use

#### Environment Setup

1. **Python version**: 3.13
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Environment variables** (.env):

```ini
OPENROUTER_API_KEY=...      # OpenRouter API key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MILVUS_HOST=host.docker.internal
MILVUS_PORT=19530
```

#### Run

1. **[Convert textbooks into md.](https://mineru.net/)**

2. **[Start Milvus](https://milvus.io/docs/install_standalone-docker.md)**

3. **Connect to LMStudio**
```bash
# Embedding model
text-embedding-mxbai-embed-large-v1
```

4. **Create vector databases**:
 
```bash
cd hardware
python setup_hw.py
cd ../vi
python setup_vi.py
```

Ensure `collection_tb` has been created.


5. **Run chatbots** (Optional)

```bash
python chatbot.py
```

6. **Start proxy**：

```bash
uvicorn proxy_app:app --reload --port 8082
# Or
uvicorn proxy_app_multimodel:app --reload --port 8082
```

7. **API examples**：

```bash
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "combined",
    "messages": [{"role":"user","content":"What is Ohm's Law?"}]
  }'
```

8. **Connect to AI Agent (DeepChat)**

```bash
API_URL: http://localhost:8082/v1
```

9. **Deploy with Docker**

```bash
docker build -t rag-proxy:latest .
docker run -d --name rag-proxy --env-file .env --network rag-net -p 8082:8082 rag-proxy:latest
```

### Tuning

**Chunking**

1. `max_tokens`: Maximum tokens per chunk.
2. `overlap`: Overlapping tokens between adjacent chunks to improve context continuity.

**Configure database parameters**

1. `index_type`: Index algorithm (FLAT, HNSW, IVF_FLAT, …)
2. `metric_type`: Vector similarity metric (L2, IP, COSINE, …)
3. `params`: (HNSW tuning)
   - `M` Max number of neighbors per node
   - `efConstruction` Search depth during index build
   - `ef` Search depth at query time (must satisfy ef > k)

**Configure retriever parameters**

1. `k`: Number of most relevant chunks to return per query
2. `search_type`
   - `similarity` (vector similarity)
   - `bm25` (keyword search)
   - `hybrid` (vector + keyword)
3. `chain_type` (how chunks are composed into the prompt)
   - `stuff` (send all chunks)
   - `map_reduce` (summarize first, then combine)
   - `refine` (iteratively improve answer with each chunk)
   - `map_rerank` (re-rank by relevance and return top-k)

**Tags + Self-querying Retrieval**

- [Langchain doc](https://python.langchain.com/docs/how_to/self_query/)

1. Self-querying retriever:
   - Extracts specific terms from the question to filter the database by metadata, improving retrieval precision.
2. Set chunk metadata (example from setup.py):
   - `image_url` Extract image links from the passage.
   - `chapter` Classify content with their chapter.
3. Use case:
   - User asks: “Please solve question 5.11 of chapter 5.”
   - Extract "chapter 5", filter the database for chunks tagged chapter 5.
   - Return the corresponding passages.
4. Rationale:
   - Embedding models can be insensitive to small token differences. If two chunks of different chapters (e.g. example questions) have similar formats and parameters, they can be confused; metadata filtering helps avoid mixing them up.

**Switching Models**

1. embedding model

```bash
# setup.py
embeddings = OpenAIEmbeddings(
  model="text-embedding-mxbai-embed-large-v1",
  openai_api_base="http://127.0.0.1:1234/v1",
  openai_api_type="open_ai",
  openai_api_key="dummy_key", 
  check_embedding_ctx_length=False,
)
```

2. LLM

```bash
# chatbot.py
tt_llm = init_chat_model(
    "openai/gpt-4.1-mini",
    model_provider="openai",
    openai_api_base=OPENROUTER_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
)
# proxy_app_multimodel.py
# Add LLMs into LLM_CONFIGS
```