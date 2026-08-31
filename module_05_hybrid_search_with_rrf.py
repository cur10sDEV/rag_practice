import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import models

# load envs
load_dotenv()
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

# constants
FILE_PATH = Path(__file__).parent.joinpath("./raw_data/avengers.txt").resolve()
MOVIE_TITLE = "the-avengers"
COLLECTION_NAME = "sci-fi-library"
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"
DIMENSIONS = 2048


def setup_embedding_models():
    dense_embedding_model = OpenAIEmbeddings(
        base_url="https://api.jina.ai/v1",
        model="jina-embeddings-v4",
        api_key=EMBEDDING_MODEL_API_KEY,
        dimensions=DIMENSIONS,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )

    sparse_embedding_model = FastEmbedSparse(model_name="Qdrant/bm25")

    return dense_embedding_model, sparse_embedding_model


def setup_vector_db(
    dense_embedding_model: OpenAIEmbeddings, sparse_embedding_model: FastEmbedSparse
):
    qdrant_client = QdrantClient(
        url="http://localhost:6335",
    )

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            # dense vectors for semantic search
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    distance=models.Distance.COSINE, size=DIMENSIONS
                )
            },
            # sparse vectors for bm25 search (keyword search)
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    vector_store = QdrantVectorStore(
        embedding=dense_embedding_model,  # dense model here
        sparse_embedding=sparse_embedding_model,  # sparse model here
        collection_name=COLLECTION_NAME,
        client=qdrant_client,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

    return qdrant_client, vector_store


def clean_script(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip form-feed / page-break chars (common in PDF extracts)
    text = text.replace("\x0c", "\n")

    # Strip trailing whitespace on every line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ blank lines down to exactly 2 (keep paragraph breaks, kill runaway gaps)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs within a line (careful: not leading indent if you care about it)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def add_documents(qdrant_client: QdrantClient, vector_store: QdrantVectorStore):
    if qdrant_client.get_collection(COLLECTION_NAME).points_count > 0:
        print("DOCUMENTS ALREADY IN THERE")
    else:
        f = open(FILE_PATH, "r", encoding="utf-8")
        cleaned_text = clean_script(f.read())
        f.close()

        doc = Document(page_content=cleaned_text, metadata={"title": MOVIE_TITLE})

        splitter = RecursiveCharacterTextSplitter(
            separators=[
                r"\n(?=(?:INT|EXT|INT/EXT|I/E)[./])",  # scene heading — lookahead keeps it attached
                "\n\n",
                "\n",
                " ",
                "",
            ],
            is_separator_regex=True,
            chunk_size=800,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents([doc])

        vector_store.add_documents(chunks)


def get_document_chunks(
    query: str,
    vector_store: QdrantVectorStore,
):
    retrieved_chunks = vector_store.similarity_search_with_score(
        query=query,
        k=10,
        score_threshold=0.3,
        hybrid_fusion=models.FusionQuery(fusion=models.Fusion.RRF),  # default RRF
    )

    return retrieved_chunks


def get_user_query():
    query = input("Enter a query: ")
    return f"Instruct: Given a question about The avengers movie script, retrieve the scenes that best answers it\nQuery:{query}"


def main():
    dense_embedding_model, sparse_embedding_model = setup_embedding_models()

    qdrant_client, vector_store = setup_vector_db(
        dense_embedding_model=dense_embedding_model,
        sparse_embedding_model=sparse_embedding_model,
    )

    add_documents(qdrant_client=qdrant_client, vector_store=vector_store)

    user_query = get_user_query()

    # retrieve chunks with help of retriever
    retrieved_chunks = get_document_chunks(
        query=user_query,
        vector_store=vector_store,
    )

    for chunk in retrieved_chunks:
        print(chunk[1])
        print(chunk[0], end="\n\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
