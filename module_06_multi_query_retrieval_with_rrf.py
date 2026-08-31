import os
import sys
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import models

# load envs
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

# constants
COLLECTION_NAME = "sci-fi-library"
DENSE_VECTOR_NAME = "dense_vector"
DIMENSIONS = 2048


# llm
def setup_chat_model():
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        model="qwen/qwen3.6-27b",
        temperature=0,
    )
    return llm


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

    return vector_store


def get_document_chunks(
    query: str,
    vector_store: QdrantVectorStore,
):
    retrieved_chunks = vector_store.similarity_search_with_score(
        query=query,
        k=3,
        score_threshold=0.3,
    )

    return retrieved_chunks


def get_user_query():
    print("=" * 60)
    query = input("Enter a query: ")
    print("\n")
    return query


# pydantic model for structured output
class QueryVariations(BaseModel):
    queries: Annotated[list[str], Field(min_length=3, max_length=3)]


def generate_multi_query(query: str, llm: ChatOpenAI):
    llm_with_tools = llm.with_structured_output(QueryVariations)

    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents
    
    Original Query:{query}
    
    Return 3 alternate queries that rephrase or approach the same question with different angles.
    """

    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries

    return query_variations


def reciprocal_rank_fusion(
    chunks: dict[int, dict[str, list[tuple[Document, float | int]] | str]], k=60
) -> list[tuple[Document, float]]:
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for entry in chunks.values():
        for rank, (doc, _score) in enumerate(entry["chunks"]):
            doc_id = doc.metadata.get("_id") or doc.page_content
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            doc_map[doc_id] = doc

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]


def main():
    chat_model = setup_chat_model()

    embedding_model = setup_embedding_model()

    vector_store = setup_vector_db(
        embedding_model=embedding_model,
    )

    user_query = get_user_query()

    # generate query variations
    query_variations = generate_multi_query(user_query, chat_model)

    print("=" * 60)
    print("GENERATED QUERIES:")
    print("=" * 60, end="\n\n")

    for i, query_variation in enumerate(query_variations):
        print(i + 1, query_variation, sep=" - ")
    print("\n")

    retrieved_chunks: dict[
        int, dict[str, list[tuple[Document, float | int]] | str]
    ] = {}

    for i, query_variation in enumerate(query_variations):
        retrieved_chunks[i] = {
            "chunks": get_document_chunks(
                query=query_variation,
                vector_store=vector_store,
            ),
            "query": query_variation,
        }

    print("=" * 60)
    print("RAW RESULTS:")
    print("=" * 60)
    total_retrieved_chunks = 0
    for entry in retrieved_chunks.values():
        print(f"=== Query: {entry['query']} ===")
        for i, doc in enumerate(entry["chunks"], 1):
            total_retrieved_chunks += 1
            print(f"--- {i} ---")
            print(f"SCORE: {doc[1]}")
            print(" ".join(doc[0].page_content.split()), end="\n\n")
    print(f"TOTAL RETRIEVED CHUNKS: {total_retrieved_chunks}", end="\n\n")

    fused_results = reciprocal_rank_fusion(retrieved_chunks, k=60)

    print("=" * 60)
    print(f"FUSED RESULTS - TOTAL RESULTS AFTER RRF: {len(fused_results)}")
    print("=" * 60)
    for rank, (doc, score) in enumerate(fused_results, 1):
        print(f"--- RANK: {rank} | RRF SCORE: {score:.4f} ---")
        print(" ".join(doc.page_content.split()), end="\n\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
