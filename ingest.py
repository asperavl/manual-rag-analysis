import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
PDF_PATH = "manual.pdf"
DB_PATH = "chroma_db"

def create_vector_db():
    print(f"📄 Loading {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
        return

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"   Found {len(docs)} pages.")

    # Split text into chunks (approx 1 paragraph each)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks.")

    # Create Embeddings using the optimized 'nomic-embed-text' model
    print("🧠 Creating embeddings (this runs locally)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 

    # Save to disk
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH) # Clean up old DB if exists
        
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✅ Knowledge Base created successfully in 'chroma_db'!")

if __name__ == "__main__":
    create_vector_db()