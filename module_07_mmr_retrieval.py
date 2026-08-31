import os
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import models

# load envs
load_dotenv()
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

# constants
COLLECTION_NAME = "sci-fi-library"
DENSE_VECTOR_NAME = "dense_vector"
DIMENSIONS = 2048


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


def setup_vector_db(embedding_model: OpenAIEmbeddings):
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
        )

    vector_store = QdrantVectorStore(
        embedding=embedding_model,  # dense model here
        collection_name=COLLECTION_NAME,
        client=qdrant_client,
        retrieval_mode=RetrievalMode.DENSE,
        vector_name=DENSE_VECTOR_NAME,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
    )

    return vector_store, retriever


def get_user_query():
    print("=" * 60)
    query = input("Enter a query: ")
    print("\n")
    return query


def main():
    embedding_model = setup_embedding_model()

    vector_store, retriever = setup_vector_db(
        embedding_model=embedding_model,
    )

    user_query = get_user_query()

    retrieved_chunks = retriever.invoke(user_query)

    print("=" * 60)
    print("RETRIEVED CHUNKS:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- DOCUMENT {i}/{len(retrieved_chunks)} ---")
        print(" ".join(chunk.page_content.split()), end="\n\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
