import os
import sys
import psutil
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

batch = 32
SEP = f"\n\n{'-'*50}\n\n"


async def clean_txt(text: str, sep: str = "\n\n"):
    seen = set()
    unique_paragraphs = []

    # Iterate over paragraphs and maintain order while avoiding duplicates
    for para in text.split(sep):
        if para not in seen:
            unique_paragraphs.append(para)
            seen.add(para)

    return sep.join(unique_paragraphs)


async def crawl_parallel(urls: List[str], file_name: str, max_concurrent: int):
    text = ""
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    os.mkdir("cache") if not os.path.exists("cache") else None

    if not os.path.exists(f"cache/{file_name[:-4]}/{file_name}"):
        os.mkdir(
            f"cache/{file_name[:-4]}") if not os.path.exists(f"cache/{file_name[:-4]}") else None
    else:
        return

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,   # corrected from 'verbos=False'
        extra_args=["--disable-gpu",
                    "--disable-dev-shm-usage", "--no-sandbox"],
        text_mode=True
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6),
            options={"ignore_links": True})
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i: i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(
                    url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            # Evaluate results
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    text += f"{result.markdown_v2.fit_markdown}{SEP}"
                    success_count += 1
                else:
                    print(url)
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")
        text = await clean_txt(text)
        with open(f"cache/{file_name[:-4]}/{file_name}", "w+", encoding="utf-8") as file:
            file.write(text)
        # with open(f"cache/{file_name[:-4]}/document.txt","w+", encoding="utf-8") as file:
        #     file.write(text)
        del text


async def scrape_website(page_url: str, file_name: str):
    """Get URLs from a webpage."""
    async with aiohttp.ClientSession() as session:
        async with session.get(page_url) as response:
            if response.status != 200:
                print(
                    f"Failed to retrieve the page. Status code: {response.status}")
                return []
            html = await response.text()

    # Parse the HTML to extract all URLs
    soup = BeautifulSoup(html, "html.parser")
    raw_urls = [a['href'] for a in soup.find_all('a', href=True)]

    # Convert relative URLs to absolute URLs
    absolute_urls = [urljoin(page_url, u) for u in raw_urls]

    await crawl_parallel(absolute_urls, file_name=file_name, max_concurrent=batch)

    with open(f"cache/{file_name[:-4]}/{file_name}", "r", encoding="utf-8") as file:
        return file.read()
