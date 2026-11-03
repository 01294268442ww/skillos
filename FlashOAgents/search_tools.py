#!/usr/bin/env python
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional, Tuple
import os
import requests
import json
import time
import asyncio
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
from .tools import Tool
from .models import OpenAIServerModel

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

# Public page evidence for the no-skill runner; no request headers are retained.
DIRECT_PAGE_FETCHES = []


def read_page_direct(url: str) -> str:
    """Fetch a static HTTP page without a reader API or browser dependency."""
    event = {"url": url, "time_utc": datetime.now(timezone.utc).isoformat(),
             "provider": "direct", "status": "error"}
    try:
        if not url.startswith(("https://", "http://")):
            raise ValueError("unsupported_scheme")
        headers = {"User-Agent": "HyperSkill/step2"}
        # GitHub's contents API can serve the exact public file without a token.
        if urlsplit(url).hostname == "api.github.com":
            headers["Accept"] = "application/vnd.github.raw+json"
        with requests.get(url, timeout=(10, 25), stream=True,
                          headers=headers) as response:
            event["http_status"] = response.status_code
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not (content_type.startswith("text/") or any(
                    kind in content_type for kind in ("json", "xml", "yaml"))):
                raise ValueError("unsupported_content_type")
            data = bytearray()
            for chunk in response.iter_content(16384):
                data.extend(chunk)
                if len(data) > 2 * 1024 * 1024:
                    raise ValueError("page_size_limit")
            encoding = response.encoding if response.encoding and "charset=" in content_type else "utf-8"
            text = data.decode(encoding, errors="replace")
            if "html" in content_type:
                from bs4 import BeautifulSoup
                document = BeautifulSoup(text, "html.parser")
                for element in document(["script", "style", "noscript"]):
                    element.decompose()
                text = document.get_text(separator="\n", strip=True)
            if not text.strip():
                raise ValueError("empty_page")
            event.update(status="ok", final_url=response.url,
                         content_type=content_type, body_bytes=len(data),
                         body_sha256=hashlib.sha256(data).hexdigest(), text=text)
            return text
    except Exception as error:
        # Do not serialize arbitrary request/response exception strings.
        event["error_type"] = type(error).__name__
        if isinstance(error, ValueError) and str(error) in {
                "unsupported_scheme", "unsupported_content_type", "page_size_limit", "empty_page"}:
            event["error_code"] = str(error)
        return "Error reading page directly: " + type(error).__name__
    finally:
        DIRECT_PAGE_FETCHES.append(event)

def read_page_jina(url: str) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown',
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        'X-Timeout': '10',
        'X-Token-Budget': '200000',
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        if response.status_code in (401, 403):
            return f"Error reading page: Jina auth failed (status={response.status_code}). Check JINA_API_KEY."
        if response.status_code == 429:
            return f"Error reading page: Jina rate limited. Try again later."
        response.raise_for_status()
        # Some providers return 200 with quota message in body
        lower_body = (response.text or "").lower()
        if any(k in lower_body for k in ["quota", "rate limit", "payment required"]):
            return "Error reading page: Jina quota exceeded."
        return response.text
    except requests.RequestException as e:
        return f"Error reading page: {str(e)}"

def read_page_crawl4ai(url: str) -> str:
    """Read and return the content of a webpage using crawl4ai."""
    try:
        from crawl4ai import AsyncWebCrawler
        
        async def _crawl():
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                return result.markdown or ""
        
        # Run async function in sync context
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    markdown = loop.run_until_complete(_crawl())
                except ImportError:
                    # Fallback: create a new thread with new event loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _crawl())
                        markdown = future.result()
            else:
                markdown = loop.run_until_complete(_crawl())
        except RuntimeError:
            # No event loop, create a new one
            markdown = asyncio.run(_crawl())
        
        if not markdown:
            return f"Error reading page: No content extracted from {url}"
        return markdown
    except ImportError:
        return "Error: crawl4ai is not installed. Please install it with: pip install crawl4ai>=0.7.4"
    except Exception as e:
        return f"Error reading page with crawl4ai: {str(e)}"

def read_page(url: str) -> str:
    """
    Read and return the content of a webpage.
    
    Provider selection via WEB_ACCESS_PROVIDER environment variable:
    - 'jina' (default): Use Jina Reader API (requires JINA_API_KEY)
    - 'crawl4ai': Use crawl4ai library (free, open-source, no API key needed)
    - 'direct': Fetch static HTML/text over HTTP; no JavaScript rendering
    
    Note: If both JINA_API_KEY and crawl4ai are configured, the provider
    specified in WEB_ACCESS_PROVIDER will be used. No automatic fallback.
    """
    provider = os.getenv("WEB_ACCESS_PROVIDER", "jina").lower()
    
    if provider == "direct":
        return read_page_direct(url)
    elif provider == "crawl4ai":
        return read_page_crawl4ai(url)
    elif provider == "jina":
        return read_page_jina(url)
    else:
        # Invalid provider, default to jina with warning
        import warnings
        warnings.warn(
            f"Invalid WEB_ACCESS_PROVIDER='{provider}'. "
            f"Valid values are 'jina', 'crawl4ai' or 'direct'. Using 'jina' as default.",
            UserWarning
        )
        return read_page_jina(url)

def web_search_bing_rss(query: str, serp_num: int = 5) -> Tuple[List[Dict[str, Any]], str]:
    """Read Bing's public search RSS feed without an API key or hidden fallback."""
    if not query.strip():
        return [], "Query is empty."
    try:
        response = requests.get("https://www.bing.com/search",
                                params={"q": query, "format": "rss"},
                                headers={"User-Agent": "HyperSkill/step2"}, timeout=20)
        response.raise_for_status()
        document = ET.fromstring(response.content)
        results = []
        for item in document.findall("./channel/item"):
            link = (item.findtext("link") or "").strip()
            title = (item.findtext("title") or "").strip()
            if not link.startswith(("https://", "http://")) or not title:
                continue
            results.append({"idx": len(results) + 1, "title": title, "date": "",
                            "snippet": item.findtext("description") or "",
                            "source": "\nSource: Bing RSS", "link": link})
            if len(results) >= serp_num:
                break
        return (results, "") if results else ([], "No search results found")
    except Exception as error:
        return [], "Error searching Bing RSS: " + type(error).__name__


def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "United States",
        "num": serp_num
    })
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            if response.status_code in (400, 401, 402, 403, 429):
                raise RuntimeError(
                    f"TERMINATE_QUOTA: Serper API quota or rate limit exceeded (status={response.status_code})"
                )
            response.raise_for_status()
            results = response.json()

            if isinstance(results, dict):
                err = results.get("error") or results.get("statusMessage") or ""
                if isinstance(err, str) and any(k in err.lower() for k in ["quota", "rate limit", "payment required"]):
                    raise RuntimeError("TERMINATE_QUOTA: Serper API quota or rate limit exceeded")

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. Try a more general query."
            
            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"\nDate published: {page['date']}" if "date" in page else "",
                    "snippet": f"\n{page.get('snippet', 'No snippet')}",
                    "source": f"\nSource: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#')
                })
            
            return search_results, ""
        
        except RuntimeError as e:
            if "TERMINATE_QUOTA" in str(e):
                raise
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
        except (requests.RequestException, json.JSONDecodeError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (400, 401, 402, 403, 429):
                raise RuntimeError(
                    f"TERMINATE_QUOTA: Serper API quota or rate limit exceeded (status={status})"
                )
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
    
    return [], "Unexpected error in web search"
def web_search_duckduckgo(
    query: str,
    serp_num: int = 5,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using DuckDuckGo (free, no API key needed)."""
    if not query.strip():
        return [], "Query is empty."

    for attempt in range(max_retries):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=serp_num))
            if not raw:
                return [], f"No results found for '{query}'."
            results = []
            for idx, r in enumerate(raw, 1):
                results.append({
                    "idx": idx,
                    "title": r.get("title", ""),
                    "date": "",
                    "snippet": f"\n{r.get('body', '')}",
                    "source": f"\nSource: {r.get('href', '')}",
                    "link": r.get("href", ""),
                })
            return results, ""
        except Exception as e:
            if attempt == max_retries - 1:
                return [], f"DuckDuckGo search failed: {str(e)}"
            time.sleep(1)
    return [], "Unexpected error in DuckDuckGo search"
class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Retrieve relevant knowledge from Wikipedia and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "Provide a query string for the information you want to retrieve from Wikipedia."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "wiki_search"

    def forward(self, query: str) -> str:
        """Execute Wikipedia search and return formatted results."""
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url"
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                return f"Wikipedia API error: {error_info.get('code', 'unknown')} - {error_info.get('info', 'unknown')}"

            pages = data.get("query", {}).get("pages", {})
            results = []
            
            for page_id, page_info in pages.items():
                if int(page_id) < 0:  # Skip invalid pages
                    continue
                    
                title = page_info.get("title", "Unknown Title")
                extract = page_info.get("extract", "No extract available")
                page_url = page_info.get("fullurl", "No URL available")
                
                results.append(
                    f"[{title}]({page_url})\n"
                    f"Summary: {extract[:500]}{'...' if len(extract) > 500 else ''}"
                )

            return "\n\n".join(results) if results else f"No relevant information found for: {query}"
        
        except requests.Timeout:
            return "Request to Wikipedia API timed out. Please try again later."
        except requests.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

class WebSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "The web search query to perform."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "web_search"

    def forward(self, query: str) -> str:
        """Execute web search and return formatted results."""
        provider = os.getenv("WEB_SEARCH_PROVIDER", "auto").lower()
        if provider == "bing_rss":
            search_results, error_msg = web_search_bing_rss(query, serp_num=5)
        elif provider not in ("auto", "serper", "duckduckgo"):
            return "Error: unsupported WEB_SEARCH_PROVIDER"
        elif provider == "serper" or (provider == "auto" and os.getenv("SERPER_API_KEY")):
            search_results, error_msg = web_search_google_serper(query, serp_num=5)
        else:
            search_results, error_msg = web_search_duckduckgo(query, serp_num=5)
        
        if error_msg:
            return error_msg
        
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No search results found"

class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Access webpage using the provided URL and extract relevant content.  Please make full use of this tool to verify the accuracy of the searched content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        },
        "query": {
            "type": "string",
            "description": "The specific information to extract from the webpage."
        }
    }
    output_type = "string"
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__()
        self.tool_name = "crawl_page"
        self.model = model

    @staticmethod
    def truncate_text(text: str, max_length: int = 60000) -> str:
        """Truncate text to specified length."""
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    def get_summary_prompt(self, query: str, url: str, content: str) -> str:
        """Generate prompt for content summarization."""
        return (
            f"Task: Extract all content from the web page that matches the search query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content [url:{url}]:\n{content}\n\n"
            "Instructions:\n"
            "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
            "- If no relevant information exists, please straightly output 'No relevant information'\n"
            "- Keep the summary under 500 words"
        )

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        """Retry model prediction with exponential backoff."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    content = response.content
                    return content.strip() if isinstance(content, str) else str(content)
                return str(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Content extraction failed: {str(e)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        return "Content extraction failed after multiple attempts"

    def forward(self, url: str, query: str) -> str:
        """Crawl webpage and extract relevant content."""
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return "Invalid URL format. Must start with http:// or https://"
        
        page_content = read_page(url)
        if page_content.startswith("Error"):
            return page_content
        
        truncated_content = self.truncate_text(page_content)
        prompt = self.get_summary_prompt(query, url, truncated_content)
        
        return self.retry_predict(prompt)
    
__all__ = [
    "WikiSearchTool",
    "WebSearchTool",
    "CrawlPageTool",
]
