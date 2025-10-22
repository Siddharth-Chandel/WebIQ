from dotenv import load_dotenv
load_dotenv()

import os
import time
import asyncio
import logging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from worker import scrape_website
from transformers import pipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from rich import print as rprint


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-2-7b-chat-hf", "TheBloke/Llama-2-7B-Chat-GGML", "TinyLlama/TinyLlama-1.1B-Chat-v1.0" ,"sentence-transformers/all-MiniLM-L6-v2""
MODEL = "TheBloke/Llama-2-7B-Chat-GGML"
EMBEDDING_MODEL = "BAAI/bge-small-en"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


async def prepare_document(url: str | list[str]):
    if isinstance(url, str):
        folder = f"{url[8:].replace('.', '-').split('/')[0]}"
        cache_path = os.path.join("cache", folder, "pages")
    else:
        folder = f"{url[0][8:].replace('.', '-').split('/')[0]}"
        cache_path = os.path.join("cache", f"list_{folder}", "pages")

    os.makedirs(cache_path, exist_ok=True)

    if not os.path.exists(f"{cache_path}/page_1.txt"):
        logging.info("Document not found. Scraping website...")
        await scrape_website(url, cache_path)
        logging.info("Scraping completed.")

    return cache_path


def get_embedding_model(embedding_model_name, api_key):
    embedding_model_name = embedding_model_name or EMBEDDING_MODEL
    if "openai" in embedding_model_name.lower():
        if api_key:
            return OpenAIEmbeddings(
                model="text-embedding-ada-002", api_key=api_key)
        else:
            logging.warning("Provide OpenAI API")
            exit()
    else:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def process_documents(file_path: str, embedding_model, chunk_size: int = 500, chunk_overlap: int = 100):
    try:
        logging.info("Loading and processing document...")

        cache_path = os.path.dirname(file_path)
        faiss_path = f"{cache_path}/faiss_index_store"

        # Skip processing if FAISS already exists
        if os.path.exists(faiss_path):
            logging.info(
                "FAISS index already exists. Skipping document processing.")
            return

        # Load document
        documents = []
        files = os.listdir(f"{cache_path}/pages")
        for file in files:
            doc_loader = TextLoader(
                f"{cache_path}/pages/{file}", encoding="utf-8")
            document = doc_loader.load()
            documents.extend(document)
        logging.info(f"Total number of pages: {len(documents)}")

        # Better chunking logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        logging.info(f"Total number of chunks: {len(docs)}")

        # Build FAISS vector store
        vector_db = FAISS.from_documents(docs, embedding_model)
        vector_db.save_local(faiss_path)

        logging.info("FAISS index and retriever saved successfully.")

    except Exception as e:
        logging.error(f"Error in document processing: {e}")


async def load_retriever(file_path: str, embedding_model_name: str = "", api_key: str = ""):
    try:
        cache_path = os.path.dirname(file_path)

        # Select embedding model (default to Hugging Face if not specified)
        embedding_model = get_embedding_model(embedding_model_name, api_key)

        # Check if FAISS index exists
        faiss_path = f"{cache_path}/faiss_index_store"

        if not os.path.exists(faiss_path):
            logging.warning(
                "FAISS index not found! Checking for retriever cache...")
            logging.warning("No retriever cache found. Processing document...")
            process_documents(
                file_path,
                embedding_model=embedding_model
            )

        # Load FAISS index safely
        vector_db = FAISS.load_local(
            faiss_path, embedding_model, allow_dangerous_deserialization=True)
        logging.info("Retriever loaded successfully.")
        return vector_db.as_retriever()

    except Exception as e:
        logging.error(f"Error loading FAISS retriever: {e}")
        return None


async def custom_pipeline(url: str | list, llm_model: str = "", embedding_model: str = "", api_key: str = ""):
    file_path = await prepare_document(url)
    retriever = await load_retriever(
        file_path, embedding_model_name=embedding_model, api_key=api_key)
    llm_model = llm_model.split()
    ggml = 0

    if not retriever:
        logging.error("Failed to load retriever. Exiting.")
        return "Error: Unable to load retriever."

    if 'openai' in llm_model:
        logging.info(f"LLM to be used: {llm_model[-1]}")
        llm = ChatOpenAI(model_name=os.getenv(
            "MODEL", llm_model[-1]), openai_api_key=api_key)
    else:
        try:
            logging.info(f"LLM to be used: {MODEL}")
            if MODEL.lower().endswith("-ggml"):
                ggml = 1
                llm = CTransformers(model=MODEL,
                                    token=HUGGINGFACEHUB_API_TOKEN,
                                    model_type="llama",
                                    config={"context_length": 4096}
                                    )
            else:
                hf_pipeline = pipeline(
                    "text-generation",
                    model=MODEL,
                    token=HUGGINGFACEHUB_API_TOKEN,
                    device_map="auto"
                )
                llm = HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            logging.error(f"Failed to load LLM model '{MODEL}'. Error: {e}", exc_info=True)
            # Raise an exception so the FastAPI endpoint can catch it and send an error message to the client.
            raise RuntimeError(f"Failed to load LLM model: {e}")

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an AI assistant. Answer the question based on the given context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    )

    return llm, retriever, custom_prompt

class Chatbot:
    def __init__(self,url: str | list, llm_model: str = "", embedding_model: str = "", api_key: str = ""):
        self.url = url
        self.llm_model = llm_model
        if 'ggml' not in llm_model:
            self.change = 1
        else:
            self.change = 0

        self.embedding_model = embedding_model
        self.api_key = api_key
    
    async def model_loaded(self):
        self.llm, self.retriever, self.custom_prompt = await custom_pipeline(self.url, self.llm_model, self.embedding_model, self.api_key)

    async def get_response(self, query:str):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.custom_prompt}
        )
        response = await asyncio.to_thread(qa_chain.invoke, query)
        
        # if self.change==1:
        #     return response['result'].split("Answer:")[1]
        
        return response['result']


async def chatbot(url: str | list, query: str, llm_model: str = "", embedding_model: str = "", api_key: str = ""):
    llm, retriever, custom_prompt = await custom_pipeline(url, llm_model, embedding_model, api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )
    response = await asyncio.to_thread(qa_chain.invoke, query)
    return response


async def main():
    """
    Main asynchronous function to run the chatbot.
    """

    url_input = input("Enter URL(s) separated by space: ")
    urls = url_input.split()
    query = input("Enter your query: ")
    
    if not urls or not query:
        rprint("[red]Please provide at least one URL and a query.[/red]")
        return

    rprint("\n[blue]...Loading models and processing documents...[/blue]")
    try:
        response = await chatbot(
            url=urls, 
            query=query
        )
    
        rprint(f"\n[red]{'=='*20}* Answer *{'=='*20}[/red]\n")
        rprint(f"[cyan]{response['result']}[/cyan]")
        rprint(f"\n[red]{'=='*17}* Source Documents *{'=='*17}[/red]\n")
        
        for i, source in enumerate(response['source_documents']):
            rprint(f"[red]Source {i+1}[/red]:")
            rprint(f"[yellow]file:[/yellow] [cyan]{source.metadata['source']}[/cyan]")
            rprint(f"[yellow]content:[/yellow]\n{source.page_content}\n")

    except Exception as e:
        rprint(f"[red]An error occurred: {e}[/red]")

if __name__ == "__main__":
    asyncio.run(main())