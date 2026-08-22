# embeddings
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_pdfmuse import PdfmuseLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

################################################################## CONSTANTS
PDF_FILE_PATH = Path(__file__).parent.joinpath("raw_data/build-your-own-database.pdf")
COLLECTION_NAME = "rag_pipeline"

################################################################## ENVs
load_dotenv()
GROQ_API_KEY = getenv("GROQ_API_KEY")

################################################################## embedding setup
embedding_model = OpenAIEmbeddings(
    model="qwen3-embedding-0.6b",
    base_url="http://localhost:8080/v1",
    api_key="not_needed",
)

################################################################## chunking
loader = PdfmuseLoader(PDF_FILE_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents=docs)

################################################################## qdrant setup
qdrant_client = QdrantClient(host="localhost", port=6335)

vector_store = QdrantVectorStore(
    collection_name=COLLECTION_NAME,
    client=qdrant_client,
    embedding=embedding_model,
)


################################################################## creating qdrant collection and embeddings
def create_embeddings(qdrant_client: QdrantClient, vector_store: QdrantVectorStore):
    collection = qdrant_client.get_collection(collection_name=COLLECTION_NAME)

    if collection.points_count is not None:
        print("COLLECTION ALREADY EXISTS!!! SKIPPING CREATING EMBEDDINGS!!!")

    else:
        print("CREATING EMBEDDINGS!!!")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
        )

        vector_store.add_documents(documents=chunks)

        print(qdrant_client.get_collection(collection_name=COLLECTION_NAME))


create_embeddings(qdrant_client, vector_store)

################################################################## retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

################################################################## querying
prompt = ChatPromptTemplate.from_template(
    template="""
    Answer the questions based only on the following context:
    
    {context}
    
    Question: {question}
    
    Make sure to answer in a concise manner, and if you dont know the answer, just say "I don't know"
    """
)

################################################################## llm setup
llm = init_chat_model(
    model="qwen/qwen3.6-27b",
    model_provider="groq",
    temperature=0.2,
    api_key=GROQ_API_KEY,
    # base_url="https://api.groq.com/openai/v1",
)


################################################################## utility functions
def format_docs(documents: list[Document]):
    return "\n\n\n".join(
        [
            f"Page Content: {doc.page_content}\nPage Number: {doc.metadata['page']}\nFile Location: {doc.metadata['source']}"
            for doc in documents
        ]
    )


################################################################## rag chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("\n\nAsk me anything: ")
    answer = rag_chain.invoke(query)
    print(f"Answer: {answer}")
