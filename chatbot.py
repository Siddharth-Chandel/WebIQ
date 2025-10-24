from dotenv import load_dotenv

load_dotenv()
import os
import asyncio
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from rich import print as rprint
from worker import scrape_website
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

DEFAULT_MODEL = "TheBloke/Llama-2-7B-Chat-GGML"
EMBEDDING_MODEL = "BAAI/bge-small-en"

# -------------------- Document Preparation --------------------
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

# -------------------- Embedding --------------------
def get_embedding_model(embedding_model_name="", api_key=""):
    # Use OpenAI if api_key provided or model name indicates OpenAI
    if api_key or "openai" in embedding_model_name.lower():
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    # Use HuggingFace otherwise
    else:
        # Ensure HF token is set in env for this thread
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

        return HuggingFaceEmbeddings(model_name=embedding_model_name or EMBEDDING_MODEL)


# -------------------- Process & Build Vector Store --------------------
def process_documents(file_path: str, embedding_model, chunk_size=500, chunk_overlap=100):
    try:
        cache_path = os.path.dirname(file_path)
        faiss_path = f"{cache_path}/faiss_index_store"

        if os.path.exists(faiss_path):
            logging.info("FAISS index exists. Skipping rebuild.")
            return

        documents = []
        for file in os.listdir(f"{cache_path}/pages"):
            doc_loader = TextLoader(os.path.join(cache_path, "pages", file), encoding="utf-8")
            documents.extend(doc_loader.load())

        logging.info(f"Loaded {len(documents)} pages")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        vector_db = FAISS.from_documents(chunks, embedding_model)
        vector_db.save_local(faiss_path)
        logging.info("FAISS store saved successfully")

    except Exception as e:
        logging.error(f"Error in document processing: {e}")

# -------------------- Load Retriever --------------------
async def load_retriever(file_path: str, embedding_model_name="", api_key=""):
    cache_path = os.path.dirname(file_path)
    embedding_model = get_embedding_model(embedding_model_name, api_key)
    faiss_path = f"{cache_path}/faiss_index_store"

    if not os.path.exists(faiss_path):
        logging.warning("FAISS index missing. Rebuilding...")
        process_documents(file_path, embedding_model)

    vector_db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------- Build Custom QA Pipeline --------------------
async def build_pipeline(url: str | list, llm_model="", embedding_model="", api_key=""):
    # Force default model if llm_model is empty or 'default'
    if not llm_model or llm_model.lower() == "default":
        llm_model = DEFAULT_MODEL
    logging.info(f"[LLM] Using model: {llm_model}")

    file_path = await prepare_document(url)
    retriever = await load_retriever(file_path, embedding_model, api_key)

    llm_model_lower = llm_model.lower()
    # OpenAI LLM
    if "openai" in llm_model_lower:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    # GGML model
    elif llm_model_lower.endswith("-ggml"):
        llm = CTransformers(model=llm_model, model_type="llama", config={"context_length": 4096})
    # Hugging Face PyTorch model
    else:
        try:
            hf_pipeline = pipeline(
                "text-generation",
                model=llm_model,
                use_auth_token=HUGGINGFACEHUB_API_TOKEN
            )
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            logging.error(f"Failed to load Hugging Face model '{llm_model}'. Error: {e}")
            raise RuntimeError(f"Cannot load Hugging Face model: {e}")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the following context to answer.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    return llm, retriever, prompt


class Chatbot:
    def __init__(self, url: str | list, llm_model="", embedding_model="", api_key=""):
        self.url = url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.api_key = api_key

    async def initialize(self):
        self.llm, self.retriever, self.prompt = await build_pipeline(
            self.url, self.llm_model, self.embedding_model, self.api_key
        )

    async def query(self, question: str):
        # Use async method if available
        if hasattr(self.retriever, "aretrieve"):
            docs = await self.retriever.aretrieve(question)
        else:
            # fallback: call the private method with run_manager=None
            docs = await asyncio.to_thread(self.retriever._get_relevant_documents, question, run_manager=None)

        context = "\n\n".join([d.page_content for d in docs])
        prompt_text = self.prompt.format(context=context, question=question)
        response = await asyncio.to_thread(self.llm.invoke, prompt_text)
        return response



# -------------------- Example Runner --------------------
async def main():
    url = input("Enter URL: ").strip()
    query = input("Enter your question: ").strip()

    bot = Chatbot([url])
    await bot.initialize()
    answer = await bot.query(query)
    rprint(f"\n[bold cyan]=== Answer ===[/bold cyan]\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())