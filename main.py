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
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from langchain_community.llms import CTransformers
import numpy as np

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-2-7b-chat-hf", "TheBloke/Llama-2-7B-Chat-GGML", "sentence-transformers/all-MiniLM-L6-v2""
MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


async def prepare_document(url):
    file_name = f"{url[8:].replace('.', '-').split('/')[0]}.txt"
    cache_path = f"cache/{file_name[:-4]}"
    os.makedirs(cache_path, exist_ok=True)

    file_path = os.path.join(cache_path, file_name)

    if not os.path.exists(file_path):
        logging.info("Document not found. Scraping website...")
        content = await scrape_website(url, file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        logging.info("Scraping completed.")

    return file_path


def process_documents(file_path: str, chunk_size=500, chunk_overlap=100, embedding_model_name:str = None, api_key:str = None):
    try:
        logging.info("Loading and processing document...")

        cache_path = os.path.dirname(file_path)
        faiss_path = f"{cache_path}/faiss_index_store"
        retriever_path = f"{cache_path}/retriever.pkl"

        # Skip processing if FAISS already exists
        if os.path.exists(faiss_path):
            logging.info("FAISS index already exists. Skipping document processing.")
            return

        # Load document
        doc_loader = TextLoader(file_path, encoding="utf-8")
        documents = doc_loader.load()

        # Better chunking logic
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        logging.info(f"Total number of chunks: {len(docs)}")

        # Allow flexible embedding models
        embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
        if "openai" in embedding_model_name.lower():
            embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
        else:
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Build FAISS vector store
        vector_db = FAISS.from_documents(docs, embedding_model)
        vector_db.save_local(faiss_path)

        # Cache retriever
        with open(retriever_path, "wb") as f:
            pickle.dump(vector_db.as_retriever(), f)

        logging.info("FAISS index and retriever saved successfully.")

    except Exception as e:
        logging.error(f"Error in document processing: {e}")


def load_retriever(file_path: str, embedding_model_name:str = None, api_key:str = None):
    try:
        cache_path = os.path.dirname(file_path)

        # Select embedding model (default to Hugging Face if not specified)
        embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        if "openai" in embedding_model_name.lower():
            embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
        else:
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Check if FAISS index exists
        faiss_path = f"{cache_path}/faiss_index_store"
        retriever_path = f"{cache_path}/retriever.pkl"

        if not os.path.exists(faiss_path):
            logging.warning("FAISS index not found! Checking for retriever cache...")
            logging.warning("No retriever cache found. Processing document...")
            process_documents(file_path,embedding_model_name, api_key=api_key)

        # Load FAISS index safely
        vector_db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
        logging.info("Retriever loaded successfully.")
        return vector_db.as_retriever()

    except Exception as e:
        logging.error(f"Error loading FAISS retriever: {e}")
        return None


async def async_chatbot(url: str, query: str):
    file_path = await prepare_document(url)
    retriever = load_retriever(file_path)

    if not retriever:
        logging.error("Failed to load retriever. Exiting.")
        return "Error: Unable to load retriever."

    if "GGML" not in MODEL:
        hf_pipeline = pipeline(
            "text-generation",
            model=MODEL,
            token=HUGGINGFACEHUB_API_TOKEN
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        llm = CTransformers(model=MODEL, model_type="llama",
                            token=HUGGINGFACEHUB_API_TOKEN)
        
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    query = f"Answer the query based on the document: \nQuery: {query}"
    response = qa_chain.invoke(query)
    return response['result']


def chatbot(url: str, query: str):
    try:
        return asyncio.run(async_chatbot(url, query))
    except RuntimeError:  # Handles event loop issues in interactive environments
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(async_chatbot(url, query))


if __name__ == "__main__":
    url = "https://playwright.dev"
    query = "Describe the playwright"
    print(chatbot(url, query))
