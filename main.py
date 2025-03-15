import os
import asyncio
import logging
import pickle
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from worker import scrape_website
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
MODEL = os.environ['MODEL']
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def prepare_document(url):
    """Scrapes website if document doesn't exist or needs updating."""
    file_name = f"{url[8:].replace('.', '-').split('/')[0]}.txt"
    
    if not os.path.exists(f"cache/{file_name[:-4]}/{file_name}"):
        logging.info("Document not found. Scraping website...")
        content = await scrape_website(url, file_name)
        os.makedirs(f"cache/{file_name[:-4]}", exist_ok=True)  # Ensure directory exists
        with open(f"cache/{file_name[:-4]}/{file_name}", "w", encoding="utf-8") as file:
            file.write(content)
        logging.info("Scraping completed.")
    
    return file_name

def process_documents(file_path:str):
    """Processes and embeds documents using FAISS vector store."""
    try:
        logging.info("Loading and processing document...")
        doc_loader = TextLoader(f"cache/{file_path[:-4]}/{file_path}", encoding="utf-8")
        documents = doc_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        logging.info(f"Total number of chunks: {len(docs)}")

        # Generate embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embedding_model)

        # Save FAISS vectorstore
        os.makedirs(f"cache/{file_path[:-4]}", exist_ok=True)  # Ensure directory exists
        vector_db.save_local(f"cache/{file_path[:-4]}/faiss_index_store")
        with open(f"cache/{file_path[:-4]}/retriever.pkl", "wb") as f:
            pickle.dump(vector_db.as_retriever(), f)
        logging.info("FAISS index and retriever saved successfully.")
    
    except Exception as e:
        logging.error(f"Error in document processing: {e}")

def load_retriever(file_path:str):
    """Loads the FAISS retriever and index."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Check if FAISS index exists before loading
        if not os.path.exists(f"cache/{file_path[:-4]}/faiss_index_store"):
            logging.warning("FAISS index not found! Processing document...")
            process_documents(file_path)  # Process if not found
        
        vector_db = FAISS.load_local(f"cache/{file_path[:-4]}/faiss_index_store", embedding_model, allow_dangerous_deserialization=True)
        logging.info("Retriever loaded successfully.")
        return vector_db.as_retriever()
    
    except Exception as e:
        logging.warning(f"Error loading FAISS retriever: {e}")
        return None

async def async_chatbot(url: str, query: str):
    """Main chatbot function that integrates document retrieval and LLM response."""
    file_path = await prepare_document(url)
    retriever = load_retriever(file_path)

    query = f'Use the scraped content from "{file_path[:-3]}" to answer: {query}'
    
    if not retriever:
        logging.error("Failed to load retriever. Exiting.")
        return "Error: Unable to load retriever."

    # Load model locally
    hf_pipeline = pipeline(
        "text-generation", 
        model=MODEL,
        token=HUGGINGFACEHUB_API_TOKEN
    )

    # Wrap in LangChain pipeline
    llm = HuggingFacePipeline(pipeline=hf_pipeline)


    # Create Retrieval-Augmented Generation (RAG) chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Retrieve relevant documents for debugging
    retrieved_docs = retriever.invoke(query)  # Replacing deprecated get_relevant_documents()
    logging.info(f"Retrieved {len(retrieved_docs)} documents")

    # Generate response
    response = qa_chain.invoke(query)  # Replacing deprecated run()
    return response['result']

def chatbot(url: str, query: str):
    """Handles existing event loops properly"""
    try:
        loop = asyncio.get_running_loop()  # Check if an event loop is running
        future = asyncio.ensure_future(async_chatbot(url, query))  # Run in current event loop
        return loop.run_until_complete(future)
    except RuntimeError:  # If no running loop, create a new one
        return asyncio.run(async_chatbot(url, query))

if __name__ == "__main__":
    url = "https://playwright.dev"
    query = "Describe the document"
    print(asyncio.run(async_chatbot(url, query)))
