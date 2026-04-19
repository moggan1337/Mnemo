"""
Web Search Perceiver implementation.

Provides web search capabilities using DuckDuckGo and other providers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from mnemo.perceiver.base import BasePerceiver, SearchResult, Source

logger = logging.getLogger(__name__)


class WebSearchPerceiver(BasePerceiver):
    """
    Web search perceiver using DuckDuckGo.
    
    Features:
    - DuckDuckGo search (no API key required)
    - Safe search options
    - Region/language targeting
    - Result filtering by type
    - Automatic retry with exponential backoff
    
    Usage:
        perceiver = WebSearchPerceiver()
        results = await perceiver.search("machine learning transformers", max_results=10)
        for source in results.results:
            print(f"{source.title}: {source.url}")
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        safe_search: bool = True,
        region: str = "wt-wt",  # Worldwide
        max_concurrent: int = 5,
    ):
        """
        Initialize web search perceiver.
        
        Args:
            config: Configuration dictionary
            safe_search: Enable safe search filtering
            region: Search region code (e.g., "wt-wt" for worldwide)
            max_concurrent: Maximum concurrent searches
        """
        super().__init__(
            user_agent=config.get("user_agent", "Mnemo-Research-Agent/0.1") if config else "Mnemo-Research-Agent/0.1",
            timeout=config.get("timeout", 30) if config else 30,
            max_retries=config.get("max_retries", 3) if config else 3,
            cache_enabled=config.get("cache_enabled", True) if config else True,
            cache_ttl=config.get("cache_ttl", 3600) if config else 3600,
        )
        
        self.safe_search = safe_search
        self.region = region
        self.max_concurrent = max_concurrent
        
        # Rate limiting
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._last_request_time = 0
        self._min_request_interval = 1.0  # seconds between requests
        
        logger.info("WebSearchPerceiver initialized")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "all",  # all, news, videos, images
    ) -> SearchResult:
        """
        Search the web for information.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (max 30)
            search_type: Type of search - "all", "news", "videos"
            
        Returns:
            SearchResult with found sources
        """
        start_time = time.time()
        max_results = min(max_results, 30)  # DuckDuckGo limit
        
        # Check cache
        cache_key = self._get_cache_key("search", query, max_results, search_type)
        if cached := self._get_cached(cache_key):
            return cached
        
        errors = []
        sources = []
        
        try:
            # Apply rate limiting
            await self._rate_limit()
            
            # Perform search using duckduckgo-search
            sources = await self._duckduckgo_search(query, max_results, search_type)
            
            self._stats["requests_made"] += 1
            
        except Exception as e:
            logger.error(f"Web search error for '{query}': {e}")
            errors.append(str(e))
            self._stats["errors"] += 1
        
        result = SearchResult(
            query=query,
            results=sources,
            provider="duckduckgo",
            total_results=len(sources),
            execution_time=time.time() - start_time,
            errors=errors,
        )
        
        # Cache successful results
        if result.success:
            self._set_cached(cache_key, result)
        
        return result
    
    async def _duckduckgo_search(
        self,
        query: str,
        max_results: int,
        search_type: str,
    ) -> list[Source]:
        """Execute DuckDuckGo search."""
        sources = []
        
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                # Select appropriate search function
                if search_type == "news":
                    generator = ddgs.news(query, max_results=max_results)
                elif search_type == "videos":
                    generator = ddgs.videos(query, max_results=max_results)
                else:
                    generator = ddgs.text(query, max_results=max_results)
                
                for i, result in enumerate(generator):
                    if i >= max_results:
                        break
                    
                    source = self._create_source(
                        url=result.get("href", ""),
                        title=result.get("title", ""),
                        content=result.get("body", result.get("description", "")),
                        source_type="web",
                        author=result.get("author"),
                        metadata={
                            "engine": "duckduckgo",
                            "position": i + 1,
                            "score": result.get("score"),
                        }
                    )
                    
                    # Estimate reliability
                    source.reliability_score = self._estimate_reliability(source)
                    
                    sources.append(source)
                    
        except ImportError:
            logger.warning("duckduckgo-search not installed, using fallback")
            sources = await self._fallback_search(query, max_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            sources = await self._fallback_search(query, max_results)
        
        return sources
    
    async def _fallback_search(
        self,
        query: str,
        max_results: int,
    ) -> list[Source]:
        """Fallback search using requests and BeautifulSoup."""
        import requests
        from bs4 import BeautifulSoup
        
        sources = []
        
        try:
            # Simple Google search (may be blocked)
            url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
            
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Parse results
            for result in soup.select(".g")[:max_results]:
                link_elem = result.select_one("a")
                title_elem = result.select_one("h3")
                snippet_elem = result.select_one(".VwiC3b")
                
                if link_elem and title_elem:
                    url = link_elem.get("href", "")
                    if url.startswith("/url?q="):
                        url = url[7:].split("&")[0]
                    
                    if url.startswith("http"):
                        source = self._create_source(
                            url=url,
                            title=title_elem.text,
                            content=snippet_elem.text if snippet_elem else "",
                            source_type="web",
                        )
                        source.reliability_score = self._estimate_reliability(source)
                        sources.append(source)
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
        
        return sources
    
    async def fetch(self, url: str) -> Optional[Source]:
        """
        Fetch a specific webpage.
        
        Args:
            url: URL of the page to fetch
            
        Returns:
            Source object with page content
        """
        import requests
        from bs4 import BeautifulSoup
        
        # Check cache
        cache_key = self._get_cache_key("fetch", url)
        if cached := self._get_cached(cache_key):
            return cached
        
        await self._rate_limit()
        
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "lxml")
            
            # Remove scripts and styles
            for elem in soup(["script", "style", "nav", "header", "footer"]):
                elem.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else ""
            
            # Extract main content (heuristic)
            main_content = ""
            for elem in soup.find_all("p"):
                main_content += elem.get_text() + "\n"
            
            source = self._create_source(
                url=url,
                title=title,
                content=main_content[:5000],  # Limit content
                source_type="web",
            )
            
            source.reliability_score = self._estimate_reliability(source)
            self._stats["requests_made"] += 1
            self._stats["total_bytes"] += len(response.content)
            
            # Cache the result
            self._set_cached(cache_key, source)
            
            return source
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            self._stats["errors"] += 1
            return None
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with self._semaphore:
            # Enforce minimum interval
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            
            self._last_request_time = time.time()
    
    async def batch_search(
        self,
        queries: list[str],
        max_results_per_query: int = 10,
    ) -> list[SearchResult]:
        """
        Execute multiple searches concurrently.
        
        Args:
            queries: List of search queries
            max_results_per_query: Results per query
            
        Returns:
            List of SearchResults
        """
        tasks = [
            self.search(q, max_results_per_query)
            for q in queries
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def search_with_filters(
        self,
        query: str,
        max_results: int = 10,
        time_range: Optional[str] = None,  # day, week, month, year
        region: Optional[str] = None,
        language: Optional[str] = None,
    ) -> SearchResult:
        """
        Search with additional filters.
        
        Args:
            query: Search query
            max_results: Maximum results
            time_range: Time range filter
            region: Region filter
            language: Language filter
            
        Returns:
            Filtered search results
        """
        # Build query with filters
        if time_range:
            query = f"{query} past_{time_range}"
        if region:
            query = f"{query} region:{region}"
        if language:
            query = f"{query} language:{language}"
        
        return await self.search(query, max_results)
