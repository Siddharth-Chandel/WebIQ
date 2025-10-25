# WebIQ – Boosts your web intelligence with AI-powered insights

## Overview
WebIQ is a powerful **web scraping** and **question-answering (QA)** chatbot that follows the **Retrieval-Augmented Generation (RAG)** pipeline. It extracts and retrieves key insights from any website and generates **AI-powered** responses based on the extracted data. WebIQ leverages **FAISS** for efficient similarity search, **LangChain** for retrieval orchestration, and state-of-the-art **LLMs** for response generation.

## Features
- **Automated Web Scraping**: Extracts text data from webpages, caches it locally, and supports both targeted and full-site scraping.
- **Vector Embeddings**: Uses FAISS to store and retrieve information efficiently.
- **LLM Integration**: Supports OpenAI (GPT-4) and Hugging Face (Llama-2, Mistral, etc.).
- **Chunking for Optimization**: Splits documents into meaningful chunks to enhance retrieval quality.
- **Asynchronous Processing**: Uses `asyncio` for efficient execution.
- **Caching Mechanism**: Ensures previously processed webpages are not reprocessed.
- **Batch Processing**: Processes large numbers of URLs efficiently.
- **Memory Usage Logging**: Tracks memory consumption before and after each batch for efficiency monitoring.
- **Multi-Page Scraping**: Seamlessly scrapes content from multiple webpages and aggregates insights.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Siddharth-Chandel/WebIQ.git
    cd WebIQ
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt

    playwright install
    ```

4. Set up environment variables by creating a `.env` file:
    ```sh
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
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
## Practical Use Cases
- **Research Assistance**: Quickly extract and summarize information from research papers, blogs, or documentation.
- **Competitive Analysis**: Monitor competitors' websites and extract relevant insights for business strategy.
- **Customer Support**: Enhance chatbot capabilities by integrating real-time website data retrieval.
- **Market Intelligence**: Gather structured data from news sites, product pages, or financial reports for analysis.
- **SEO Optimization**: Analyze webpage content for better keyword targeting and content strategy.

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
- Develop an interactive web UI using Streamlit or FastAPI for a seamless user experience.
- Enhance retrieval quality with advanced RAG tuning and improved embeddings.

## License
This project is licensed under the **MIT License**.

## Author
Siddharth Chandel - Developed as part of NLP & AI research.
Let's connect on [LinkedIn](https://www.linkedin.com/in/siddharth-chandel-001097245/) !!!

---
_Contributions are welcome! Feel free to fork and enhance._ 🚀
