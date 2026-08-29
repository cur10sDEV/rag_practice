import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

# load envs
load_dotenv()
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

# constants
COLLECTION_NAME = "how_linux_works"
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
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
        use_jsonb=True,
        embedding_length=DIMENSIONS,
        distance_strategy=DistanceStrategy.COSINE,
        connection=VECTOR_DB_URL,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.35},
    )

    return vector_store, retriever


def get_document_chunks(query: str, retriever: VectorStoreRetriever) -> list[Document]:
    results = retriever.invoke(input=query)
    return results


def get_user_query():
    query = input("Enter a query: ")
    return f"Instruct: Given a question about Linux internals, retrieve the book passage that best answers it\nQuery:{query}"


def main():
    embedding_model = setup_embedding_model()
    vector_store, retriever = setup_vector_db(embedding_model)
    user_query = get_user_query()

    # retrieve chunks with help of retriever
    # retrieved_chunks = get_document_chunks(query=user_query, retriever=retriever)

    # or if you want to get score too - then use vector_store instead of retriever
    retrieved_chunks = vector_store.similarity_search_with_score(query=user_query, k=5)

    # sort in the decreasing order of scores
    retrieved_chunks.sort(reverse=True, key=lambda x: x[1])

    for i, chunk in enumerate(retrieved_chunks):
        print(
            f"\n\n{i + 1} CHUNK LOADED WITH SCORE: {chunk[1]} ----------------------------------"
        )
        print(chunk[0].page_content)

        # chunk[0] - actual document
        # chunk[1] - similarity score


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
