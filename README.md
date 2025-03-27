# WebIQ – Boosts your web intelligence with AI-powered insights

## Overview
This project is a **web scraping and question-answering (QA) chatbot** that follows the **RAG (Retrieval Augmented Generation)** pipeline, retrieving relevant information from a given website and answering user queries based on the extracted data. It utilizes **FAISS for vector storage**, **LangChain for retrieval and chaining**, and **Hugging Face / OpenAI LLMs** for response generation.

## Features
- **Automated Web Scraping**: Extracts text data from webpages and caches it locally, and can be configured to scrape a set of webpages or a fully-fledged website.
- **Vector Embeddings**: Uses FAISS to store and retrieve information efficiently.
- **LLM Integration**: Supports OpenAI (GPT-4) and Hugging Face (Llama-2, Mistral, etc.).
- **Chunking for Optimization**: Documents are split into manageable chunks for better retrieval.
- **Asynchronous Processing**: Uses `asyncio` for efficient execution.
- **Caching Mechanism**: Programmed in such a way that no need to process already processed webpages again and again.
- **Batch Processing**: Curated batches to process large numbers of URLs efficiently.
- **Logging Memory Utilisation**: Shows the memory consumption before and after each batch.
- **Support for Multiple Webpages**: Added support for multi-page scraping.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/webscrape-qa-chatbot.git
    cd webscrape-qa-chatbot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables by creating a `.env` file:
    ```sh
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
    MODEL="TheBloke/Llama-2-7B-Chat-GGML"
    EMBEDDING_MODEL="BAAI/bge-small-en"
    OPENAI_API_KEY=your_openai_api_key  # If using OpenAI
    ```

## Usage

1. Run the chatbot script:
    ```sh
    python chatbot.py
    ```
2. Enter a **URL** when prompted (e.g., `https://playwright.dev`).
3. Enter your **query** (e.g., `Describe Playwright and its benefits`).
4. The chatbot will scrape the webpage, process the data, and return an AI-generated response.

## Example Output
```
====================* Answer *====================
Playwright is an end-to-end testing framework that provides...

=================* Source Documents *=================
Source 1:
file: cache/playwright-dev/pages/page_1.txt
Content: Playwright is a Node.js library that automates browsers.
```

## Technologies Used
- **RAG** (Providing better context)
- **LangChain** (Retrieval-based QA system)
- **FAISS** (Efficient similarity search)
- **Hugging Face Transformers** (LLMs & embeddings)
- **OpenAI GPT-4** (Optional for LLM-based response generation)
- **Crawl4AI** (An LLM-based web-scraper)
- **AsyncIO** (Increment the processing speed)
- **Rich** (For colorful CLI outputs)

## Future Enhancements
- Implement a web interface using Streamlit or FastAPI.
- Improve response accuracy with RAG optimization.

## License
This project is licensed under the **MIT License**.

## Author
Siddharth Chandel - Developed as part of NLP & AI research.
Let's connect on [LinkedIn](https://www.linkedin.com/in/siddharth-chandel-001097245/) !!!

---
_Contributions are welcome! Feel free to fork and enhance._ 🚀
