# worker.py
import os
import asyncio
import psutil
from urllib.parse import urlparse, urlunparse
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import traceback

# ------------------------------
# File paths & config
# ------------------------------
__location__ = os.path.dirname(os.path.abspath(__file__))
batch = 32  # max concurrent crawls
goto_timeout = 60_000  # 1 minutes

# ------------------------------
# Utility functions
# ------------------------------
def normalize_url(url: str) -> str:
    """Normalize URL to avoid duplicates."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

async def get_internal_urls(url_set: set, visited: set, crawler) -> set:
    """Collect internal links from a page."""
    internal_urls = crawler.links.get("internal", [])
    for link in internal_urls:
        href = link.get("href")
        if href and href.startswith("http"):
            normalized_href = normalize_url(href)
            if normalized_href not in visited:
                url_set.add(normalized_href)
    return url_set

# ------------------------------
# Core crawling function
# ------------------------------
async def crawl_parallel(urls: List[str] | str, file_path: str, max_concurrent: int = batch):
    """Crawl multiple URLs asynchronously with retries, save pages, and track failures."""
    text_pages = set()
    not_visited = set(urls if isinstance(urls, list) else [urls])
    visited = set()
    retry = set()
    failed = set()
    was_str = isinstance(urls, str)
    n = 1

    os.makedirs(file_path, exist_ok=True)

    process = psutil.Process()
    peak_memory = 0
    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss
        peak_memory = max(peak_memory, current_mem)
        print(f"{prefix} Memory: {current_mem // (1024*1024)} MB | Peak: {peak_memory // (1024*1024)} MB")

    # Browser & crawler config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        text_mode=True
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6),
            options={"ignore_links": True}
        ),
        page_timeout=goto_timeout
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    print("\n=== Starting robust parallel crawling ===")

    async def safe_crawl(url, session_id):
        """Crawl a URL safely, return result or None."""
        try:
            result = await crawler.arun(url=url, config=crawl_config, session_id=session_id)
            return result
        except Exception as e:
            print(f"[WARN] Failed to crawl {url}: {e}")
            return None

    try:
        while not_visited:
            urls_batch = list(not_visited)[:max_concurrent]
            tasks = [safe_crawl(url, f"session_{i}") for i, url in enumerate(urls_batch)]

            log_memory(prefix=f"Before batch {n}: ")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            log_memory(prefix=f"After batch {n}: ")

            for url, result in zip(urls_batch, results):
                if isinstance(result, Exception) or result is None or not getattr(result, "success", False):
                    if url not in retry:
                        retry.add(url)
                        print(f"[INFO] Retry scheduled for {url}")
                    else:
                        failed.add(url)
                        not_visited.discard(url)
                        visited.add(url)
                        print(f"[ERROR] Crawling failed for {url} after retry")
                else:
                    text_pages.add(result.markdown.fit_markdown)
                    if was_str:
                        internal_urls = result.links.get("internal", [])
                        for link in internal_urls:
                            href = link.get("href")
                            if href and href.startswith("http"):
                                normalized_href = normalize_url(href)
                                if normalized_href not in visited:
                                    not_visited.add(normalized_href)
                    visited.add(url)
                    retry.discard(url)
                    not_visited.discard(url)
            n += 1

    except Exception as e:
        traceback.print_exc()
        print(e)
    finally:
        await crawler.close()
        log_memory(prefix="Final: ")

    # Save pages
    pages = [p for p in text_pages if p.strip()]
    for i, page in enumerate(pages):
        with open(os.path.join(file_path, f"page_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(page)

    print(f"\nSummary:")
    print(f"  - Successfully crawled pages: {len(pages)}")
    print(f"  - Failed URLs: {len(failed)} -> {failed}")
    print(f"Peak memory usage: {peak_memory // (1024*1024)} MB")

    return {
        "success_count": len(pages),
        "failed_urls": list(failed),
        "peak_memory_MB": peak_memory // (1024*1024)
    }

# ------------------------------
# Public scrape function
# ------------------------------
async def scrape_website(urls: str | list, file_path: str):
    """Wrapper to start crawling and return summary."""
    summary = await crawl_parallel(urls, file_path, max_concurrent=batch)
    return summary
