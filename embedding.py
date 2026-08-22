from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_pdfmuse import PdfmuseLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# file path
PDF_FILE_PATH = Path(__file__).parent / "raw_data/build-your-own-database.pdf"

# data extraction
loader = PdfmuseLoader(PDF_FILE_PATH)
docs = loader.load()

# chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

# embeddings
embedding_model = OpenAIEmbeddings(
    model="qwen3-embedding-0.6b",
    base_url="http://localhost:8080/v1",
    api_key="not_needed",
)

# connect to vector db / store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,  # pass embedding model here
    url="http://localhost:6335",
    collection_name="learning_rag",
)

print("Chunking and Embedding done!!!")
