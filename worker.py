import os
import sys
import psutil
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import traceback

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

batch = 32


def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))


async def get_internal_urls(myset: set, visited: set, crawler):
    internal_urls = crawler.links.get('internal', [])
    for link in internal_urls:
        href = link.get('href')
        if href and href.startswith("http"):
            normalized_href = normalize_url(href)
            if normalized_href not in visited:
                myset.add(normalized_href)  # Add only normalized URLs
    return myset


async def crawl_parallel(urls: List[str] | str, file_path: str, max_concurrent: int = 32):
    text = set()
    not_visit = set()
    visited = set()
    retry = set()
    failed = set()
    wasStr = 0
    n = 1
    if isinstance(urls, str):
        urls = [urls]
        wasStr = 1
    not_visit.update(urls)
    # We'll chunk the URLs in batches of 'max_concurrent'
    success_count = 0
    fail_count = 0
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    os.makedirs(file_path, exist_ok=True)

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
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

    while len(not_visit):
        try:
            for i in range(0, len(urls), max_concurrent):
                current_batch = urls[i: i + max_concurrent]
                tasks = []

                for j, url in enumerate(current_batch):
                    # Unique session_id per concurrent sub-task
                    session_id = f"parallel_session_{i + j}"
                    task = crawler.arun(
                        url=url, config=crawl_config, session_id=session_id)
                    tasks.append(task)

                # Check memory usage prior to launching tasks
                log_memory(prefix=f"Before batch {n}: ")

                # Gather results
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check memory usage after tasks complete
                log_memory(prefix=f"After batch {n}: ")

                # Evaluate results
                for url, result in zip(current_batch, results):
                    if isinstance(result, Exception):
                        print(f"Error crawling {url}: {result}")
                        fail_count += 1
                    elif result.success:
                        content = f"{result.markdown_v2.fit_markdown}"
                        text.add(content)
                        if wasStr:
                            not_visit = await get_internal_urls(not_visit, visited, result)
                        retry.discard(url)
                        not_visit.discard(url)
                        visited.add(url)
                        success_count += 1
                    else:
                        if url not in retry:
                            not_visit.add(url)
                            retry.add(url)
                        else:
                            print(url)
                            failed.add(url)
                            not_visit.discard(url)
                            visited.add(url)
                            fail_count += 1
                urls = list(not_visit)
                n += 1
        except Exception as e:
            traceback.print_exc()
            print(e)
            break

    print(f"\nSummary:")
    print(f"  - Successfully crawled: {success_count}")
    print(f"  - Failed: {fail_count}")
    print(f"  - Total: {success_count + fail_count}")

    print("\nClosing crawler...")
    await crawler.close()
    # Final memory log
    log_memory(prefix="Final: ")
    print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")
    total_pages = list(text)
    for i, page in enumerate(total_pages):
        if not page.strip():
            total_pages.pop(i)
        else:
            with open(f"{file_path}/page_{i+1}.txt", "w+", encoding="utf-8") as file:
                file.write(page)
    print(f"Failed urls: {failed}")
    del text, not_visit, visited, retry, failed


async def scrape_website(page_url: str | list, file_name: str):
    """Get URLs from a webpage."""
    await crawl_parallel(page_url, file_path=file_name, max_concurrent=batch)
