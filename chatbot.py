import os, re
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, utility

from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chains import LLMChain


MILVUS_HOST = os.getenv("MILVUS_HOST", "host.docker.internal")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

load_dotenv(override=True)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")

tt_llm = init_chat_model(
    "openai/gpt-4.1-mini",
    model_provider="openai",
    openai_api_base=OPENROUTER_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
)

class PrefixedEmbeddings(OpenAIEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        prompt = f"Represent this sentence for searching relevant passages: {text}"
        return super().embed_query(prompt)
    
    async def aembed_query(self, text: str) -> list[float]:
        prompt = f"Represent this sentence for searching relevant passages: {text}"
        return await super().aembed_query(prompt)

embeddings = PrefixedEmbeddings(
    model="text-embedding-mxbai-embed-large-v1",
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_type="open_ai",
    openai_api_key="dummy_key",
    check_embedding_ctx_length=False,
)

COL_TA = "collection_tb"
if not utility.has_collection(COL_TA):
    raise RuntimeError(f"Milvus collection '{COL_TA}' not found. Run setup first.")

store_hw = Milvus(
    embedding_function=embeddings,
    collection_name=COL_TA,
    connection_args={"uri": f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"},
    index_params={
        "index_type": "HNSW",
        "metric_type": "L2"
    },
    search_params={
        "ef": 250
    },
    drop_old=False,
)

metadata_field_info=[
    AttributeInfo(
        name="chapter",
        description="chapter name of each section",
        type="string",
    ),
    AttributeInfo(
        name="image_url",
        description="URL of associated image",
        type="string",
    ),
]

document_content_description = "Course textbook content"

system_text = Path("system_prompt_ta.md").read_text(encoding="utf-8")
no_think_system = SystemMessagePromptTemplate.from_template(system_text)
no_think_human = HumanMessagePromptTemplate.from_template(
    "Use the following documents to answer the question:\n\n{context}\n\nQuestion: {question}"
)
custom_prompt = ChatPromptTemplate.from_messages([no_think_system, no_think_human])

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=tt_llm,
    vectorstore=store_hw,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True, 
    search_kwargs={ "k": 50 },
    # search_kwargs={ "k": 50, "ef": 250 },
    search_type="similarity", 
)

qa_ta = RetrievalQA.from_chain_type(
    llm=tt_llm,
    chain_type="stuff",
    retriever=self_query_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt},
)

# Chat Loop
def main():
    print("Chat start (type 'exit' or 'quit' to stop)")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            print("Ending session...")
            break

        print(f"[Query]\n{query}\n")
        result = qa_ta.invoke(query)
        answer = result["result"]
        docs = result["source_documents"]
        if not docs:
            print("Cannot find relevant documents.")
            continue

        clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        
        print(f"[TA Answer]\n{clean_answer}\n")
        print("Used sources:")
        for d in docs:
            print(" -", d.metadata.get("source"))

        imgs = [d.metadata.get("image_url") for d in docs if d.metadata.get("image_url")]
        if imgs:
            print("Related image(s):")
            for url in set(imgs):
                print(" •", url)


if __name__ == "__main__":
    main()
