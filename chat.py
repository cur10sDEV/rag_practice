import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM
chat_model = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# embeddings
embedding_model = OpenAIEmbeddings(
    model="qwen3-embedding-0.6b",
    base_url="http://localhost:8080/v1",
    api_key="not_needed",
)

# connect to vector db / store
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,  # pass embedding model here
    url="http://localhost:6335",
    collection_name="learning_rag",
)

# system prompt
SYSTEM_PROMPT = """
You are a helpful AI assistant who answers user query based on available context retrieved from a PDF file along with page_contents and a page number.

You should only ans the user based on the following context and navigate the user to open the right page number to know more.

Use the context provided to get the accurate results in conversations between the <context></context> tags.
"""

conversation_history = [
    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
]

while True:
    # take user input
    user_query = input("\n\nEnter a query: ")

    if user_query.strip().lower() == "exit":
        break

    # get relevant chunks from vector db
    search_results = vector_store.similarity_search(query=user_query)

    context = "\n\n\n".join(
        [
            f"Page Content: {result.page_content}\nPage Number: {result.metadata['page']}\nFile Location: {result.metadata['source']}"
            for result in search_results
        ]
    )

    # add context from similarity search and user query
    conversation_history.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""
                        Context retrieved from the knowledge base:
                        
                        <context>
                        {context}
                        </context>
                        
                        Question:
                        {user_query}
                        """,
                }
            ],
        }
    )

    response = chat_model.responses.create(
        model="qwen/qwen3.6-27b",
        input=conversation_history,
    )

    print(response.output_text)
