import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

# envs
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")

# constants
COLLECTION_NAME = "how_linux_works"
DIMENSIONS = 2048

# system prompt
SYSTEM_PROMPT = """
You are a helpful AI assistant who answers user query based on available context retrieved from a PDF file along with page_contents and a page number.

You should only ans the user based on the following context and navigate the user to open the right page number to know more.

Use the context provided to get the accurate results in conversations between the <context></context> tags.
"""


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
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
        use_jsonb=True,
        embedding_length=DIMENSIONS,
        distance_strategy=DistanceStrategy.COSINE,
        connection=VECTOR_DB_URL,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
        },
    )

    return vector_store, retriever


def get_document_chunks(query: str, retriever: VectorStoreRetriever) -> list[Document]:
    results = retriever.invoke(input=query)
    return results


def get_user_query():
    query = input("Enter a query: ")
    return f"Instruct: Given a question about Linux internals, retrieve the book passage that best answers it\nQuery:{query}"


def main():
    try:
        # setup
        chat_model = setup_chat_model()
        embedding_model = setup_embedding_model()
        vector_store, retriever = setup_vector_db(embedding_model)

        # retrieval
        user_query = get_user_query()
        retrieved_documents = get_document_chunks(user_query, retriever)

        # augment
        context = "\n\n\n".join(
            [
                f"Page Content: {result.page_content}\nPage Number: {result.metadata['page']}\nFile Location: {result.metadata['source']}"
                for result in retrieved_documents
            ]
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
                            Context retrieved from the knowledge base:
                            
                            <context>
                            {context}
                            </context>
                            
                            Question:
                            {user_query}
                            """,
            },
        ]

        # generation
        llm_response = chat_model.invoke(input=messages)

        # print results
        print("\n\n------ GENERATED RESPONSE ------")
        print(llm_response.content)
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
