import os, glob, json, re, tiktoken
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from mrkdwn_analysis import MarkdownAnalyzer

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from chapterParser import parse_markdown_to_documents_with_chapter

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "ni-rag"
os.environ["USER_AGENT"] = "MyRAGLoader/1.0 (+https://github.com/AdmiralHeathCL/NI-Yuxuan)"

load_dotenv(override=True)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")


tt_llm = init_chat_model(
    "openai/gpt-4.1-mini",
    model_provider="openai",
    openai_api_base=OPENROUTER_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-mxbai-embed-large-v1",
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_type="open_ai",
    openai_api_key="dummy_key", 
    check_embedding_ctx_length=False,
)


connections.connect(uri="tcp://localhost:19530")

# Drop database before reinitializing
collection_dp = "collection_tb"
if utility.has_collection(collection_dp):
    utility.drop_collection(collection_dp)
    print(f"Dropped collection: {collection_dp}")
else:
    print(f"Collection not found: {collection_dp}")

fields = [
    FieldSchema(
        "pk", 
        DataType.INT64, 
        is_primary=True, 
        auto_id=True
    ),
    FieldSchema(
        "vector", 
        DataType.FLOAT_VECTOR, 
        dim=1024
    ),
    FieldSchema(
        "text", 
        DataType.VARCHAR, 
        max_length=65535
    ),
    FieldSchema(
        "source", 
        DataType.VARCHAR, 
        max_length=65535
    ),
    FieldSchema(
        "chunk_start_token", 
        DataType.INT64
    ),
    FieldSchema(
        "image_url", 
        DataType.VARCHAR, 
        max_length=65535, 
        is_nullable=True, 
        default_value=""
    ),
    FieldSchema(
        "chapter",
        DataType.VARCHAR,
        max_length=65535
    )
]

schema = CollectionSchema(fields, description="textbook")
Collection(name="collection_tb", schema=schema)


def get_dx_json(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def get_md_table_jsons(md_path: str) -> list[str]:
    analyzer = MarkdownAnalyzer(md_path)
    tables = analyzer.identify_tables().get('Table', [])
    rows = []
    for table in tables:
        hdr = table['header']
        for line in table['rows']:
            d = {hdr[i]: line[i] for i in range(len(hdr)) if hdr[i]}
            rows.append(json.dumps(d))
    return rows

def get_md_paragraph_jsons(md_path: str) -> list[str]:
    with open(md_path, 'r', encoding='utf8') as f:
        return [f.read()]
    
def get_md_image_docs(md_path: str) -> list[Document]:
    """
    Parse markdown image tags and create Document for each image URL.
    """
    docs = []
    text = Path(md_path).read_text(encoding='utf8')
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    base_dir = Path(md_path).parent
    for alt_text, rel_path in img_pattern.findall(text):
        img_path = (base_dir / rel_path).resolve()
        url = img_path.as_uri()
        docs.append(
            Document(
                page_content=alt_text or "",
                metadata={
                    'source': md_path,
                    'image_url': url,
                }
            )
        )
    return docs

def split_by_tokens(docs: list[Document], max_tokens: int = 500, overlap: int = 100):
    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    stride = max_tokens - overlap
    out = []
    for doc in docs:
        meta = doc.metadata or {}
        filename = Path(meta.get('source', '')).name

        # If this is an image doc, keep it whole but still set chunk_start_token
        if 'image_url' in meta:
            m = meta.copy()
            m['chunk_start_token'] = 0
            out.append(Document(page_content=doc.page_content, metadata=m))
            continue

        # Otherwise, split by tokens as before
        toks = encoder.encode(doc.page_content)
        for start in range(0, len(toks), stride):
            chunk_text = encoder.decode(toks[start:start+max_tokens])
            # full_text = f"File: {filename}\n{chunk_text}"
            m = meta.copy()
            m['chunk_start_token'] = start
            out.append(Document(page_content=chunk_text, metadata=m))

    return out



def build_collections():
    BATCH_SIZE = 10
    hw_docs = []
    hw_root = Path(r".\mds")

    for md_file in hw_root.rglob("*.md"):
        docs_with_chapter = parse_markdown_to_documents_with_chapter(str(md_file))
        hw_docs.extend(docs_with_chapter)

    hw_chunks = split_by_tokens(hw_docs)  # keeps metadata including `chapter`

    store_hw = Milvus(
        embedding_function=embeddings,
        collection_name="collection_tb",
        connection_args={'uri': 'tcp://localhost:19530'},
        drop_old=False,
        auto_id=True,
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 500, "ef": 500},
        },
    )

    for i in range(0, len(hw_chunks), BATCH_SIZE):
        batch = hw_chunks[i: i + BATCH_SIZE]
        store_hw.add_documents(batch)
        print(f"→ Indexed docs {i} to {i + len(batch) - 1}")

    retr_hw = store_hw.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50, "params": {"ef": 250}},
    )
    qa_ta = RetrievalQA.from_chain_type(
        llm=tt_llm,
        chain_type='stuff',
        retriever=retr_hw
    )
    return qa_ta


qa_ta = build_collections()