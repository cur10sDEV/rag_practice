import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pdfmuse import PdfmuseLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# load envs
load_dotenv()
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

# constants
PDF_PATH = (
    Path(__file__)
    .parent.joinpath(Path("raw_data/How Linux Works 3rd Edition.pdf"))
    .resolve()
)
COLLECTION_NAME = "how_linux_works"
DIMENSIONS = 2048


# setup embedding model
def setup_embedding_model():
    embedding_model = OpenAIEmbeddings(
        base_url="https://api.jina.ai/v1",
        model="jina-embeddings-v4",
        api_key=EMBEDDING_MODEL_API_KEY,
        dimensions=DIMENSIONS,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )

    return embedding_model


# setup vector db
def setup_vector_db(embedding_model: OpenAIEmbeddings):
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
        use_jsonb=True,
        embedding_length=DIMENSIONS,
        distance_strategy=DistanceStrategy.COSINE,
        connection=VECTOR_DB_URL,
    )

    print("VECTOR DB SETUP COMPLETED")
    return vector_store


# main functions
def load_documents():
    pdf_loader = PdfmuseLoader(file_path=PDF_PATH, fmt="pdf", mode="page")
    docs = pdf_loader.lazy_load()
    print("PDF LOADING COMPLETED")
    return docs


def create_chunks(docs: Iterator[Document]) -> Iterator[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
    for doc in docs:
        yield from splitter.split_documents([doc])


def create_vector_embeddings(
    vector_store: VectorStore, chunks: Iterator[Document], batch_size: int = 32
):
    batch = []
    batch_num = 0
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            vector_store.add_documents(batch)
            batch_num += 1
            print(f"EMBEDDED BATCH {batch_num}")
            batch = []
            time.sleep(15)  # to get around rate limits
    if batch:
        vector_store.add_documents(batch)
        batch_num += 1
        print(f"EMBEDDED BATCH {batch_num}")
    print("VECTOR EMBEDDING COMPLETED")


def main():
    try:
        embedding_model = setup_embedding_model()
        vector_store = setup_vector_db(embedding_model)
        docs = load_documents()
        chunks = create_chunks(docs)
        create_vector_embeddings(vector_store, chunks)
        print("INGESTION PIPELINE COMPLETED")
    except Exception as e:
        print("INGESTION PIPELINE FAILED")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
