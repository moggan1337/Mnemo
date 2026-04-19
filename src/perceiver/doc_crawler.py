"""
Document Crawler for accessing documentation and technical sites.

Crawls documentation sites, wikis, and other technical resources.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urljoin, urlparse

from mnemo.perceiver.base import BasePerceiver, SearchResult, Source

logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Configuration for document crawling."""
    max_pages: int = 100
    max_depth: int = 3
    max_concurrent: int = 5
    respect_robots: bool = True
    follow_external: bool = False
    allowed_domains: Optional[Set[str]] = None
    exclude_patterns: set[str] = field(default_factory=lambda: {
        ".pdf", ".jpg", ".png", ".gif", ".svg",
        ".css", ".js", ".ico", ".woff", ".woff2",
        "/login", "/signup", "/register",
        "logout", "signout",
    })
    delay_seconds: float = 1.0
    timeout_seconds: int = 30


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    start_url: str
    pages_crawled: int
    pages_failed: int
    sources: list[Source]
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    links_found: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.pages_crawled + self.pages_failed
        return self.pages_crawled / total if total > 0 else 0.0


class DocumentCrawler(BasePerceiver):
    """
    Documentation and website crawler.
    
    Features:
    - Configurable crawling depth and breadth
    - Respect for robots.txt
    - Link extraction and following
    - Content extraction from documentation
    - Concurrent crawling with rate limiting
    
    Usage:
        crawler = DocumentCrawler()
        result = await crawler.crawl("https://docs.example.com", max_pages=50)
        for source in result.sources:
            print(f"{source.title}: {source.url}")
    """
    
    def __init__(
        self,
        config: Optional[CrawlConfig] = None,
        download_dir: Optional[Path] = None,
    ):
        """
        Initialize document crawler.
        
        Args:
            config: Crawling configuration
            download_dir: Directory for downloaded content
        """
        super().__init__(
            cache_enabled=True,
            cache_ttl=1800,  # Documentation changes less frequently
        )
        
        self.config = config or CrawlConfig()
        self.download_dir = download_dir or Path.home() / ".mnemo" / "crawled"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._visited: Set[str] = set()
        self._robots_txt: dict[str, str] = {}
        self._crawl_semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info("DocumentCrawler initialized")
    
    async def crawl(
        self,
        start_url: str,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        query_filter: Optional[str] = None,
    ) -> CrawlResult:
        """
        Crawl a documentation site.
        
        Args:
            start_url: Starting URL for crawling
            max_pages: Maximum pages to crawl
            max_depth: Maximum link depth to follow
            query_filter: Optional text filter for pages
            
        Returns:
            CrawlResult with all discovered sources
        """
        start_time = time.time()
        
        max_pages = max_pages or self.config.max_pages
        max_depth = max_depth or self.config.max_depth
        
        # Reset state
        self._visited.clear()
        
        sources = []
        errors = []
        links_found = 0
        pages_crawled = 0
        pages_failed = 0
        
        # Initialize semaphore for concurrent crawling
        self._crawl_semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Parse start URL
        start_domain = urlparse(start_url).netloc
        
        # Queue: (url, depth)
        queue = deque([(start_url, 0)])
        
        while queue and pages_crawled < max_pages:
            url, depth = queue.popleft()
            
            # Skip if visited
            if url in self._visited:
                continue
            
            # Skip if depth exceeded
            if depth > max_depth:
                continue
            
            # Check domain restriction
            url_domain = urlparse(url).netloc
            if not self.config.follow_external and url_domain != start_domain:
                continue
            
            # Check allowed domains
            if self.config.allowed_domains and url_domain not in self.config.allowed_domains:
                continue
            
            # Apply query filter
            if query_filter and query_filter.lower() not in url.lower():
                continue
            
            # Check robots.txt
            if self.config.respect_robots:
                if not await self._can_crawl(url):
                    continue
            
            # Crawl page
            async with self._crawl_semaphore:
                try:
                    source = await self._crawl_page(url)
                    
                    if source:
                        sources.append(source)
                        pages_crawled += 1
                        
                        # Extract links
                        if depth < max_depth:
                            links = await self._extract_links(url, source.content)
                            links_found += len(links)
                            
                            for link in links[:20]:  # Limit links per page
                                if link not in self._visited:
                                    queue.append((link, depth + 1))
                    else:
                        pages_failed += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {e}")
                    errors.append(f"{url}: {str(e)}")
                    pages_failed += 1
                
                # Rate limiting
                await asyncio.sleep(self.config.delay_seconds)
            
            self._visited.add(url)
        
        return CrawlResult(
            start_url=start_url,
            pages_crawled=pages_crawled,
            pages_failed=pages_failed,
            sources=sources,
            duration_seconds=time.time() - start_time,
            errors=errors,
            links_found=links_found,
        )
    
    async def _crawl_page(self, url: str) -> Optional[Source]:
        """Crawl a single page."""
        # Check cache first
        cache_key = self._get_cache_key("crawl", url)
        if cached := self._get_cached(cache_key):
            return cached
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
            
            response = requests.get(
                url,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "lxml")
            
            # Remove unwanted elements
            for elem in soup(["script", "style", "nav", "footer", "header", 
                              "aside", "noscript", "iframe"]):
                elem.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string or ""
            else:
                h1 = soup.find("h1")
                if h1:
                    title = h1.get_text(strip=True)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Extract metadata
            meta_description = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag:
                meta_description = meta_tag.get("content", "")
            
            source = self._create_source(
                url=url,
                title=title.strip() if title else "Untitled",
                content=content,
                source_type="doc",
                metadata={
                    "description": meta_description,
                    "crawled_at": time.time(),
                    "content_length": len(content),
                }
            )
            
            source.reliability_score = self._estimate_reliability(source)
            
            # Cache
            self._set_cached(cache_key, source)
            self._stats["requests_made"] += 1
            self._stats["total_bytes"] += len(response.content)
            
            return source
            
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            self._stats["errors"] += 1
            return None
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from parsed HTML."""
        content_parts = []
        
        # Try common content containers
        content_elem = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=["content", "main-content", "documentation"]) or
            soup.find("div", id=["content", "main-content", "wiki"])
        )
        
        if content_elem:
            soup = content_elem
        
        # Extract headings and paragraphs
        for tag in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]:
            for elem in soup.find_all(tag):
                text = elem.get_text(strip=True)
                if text:
                    # Add spacing based on heading level
                    if tag.startswith("h"):
                        content_parts.append(f"\n{text}\n")
                    else:
                        content_parts.append(text)
        
        # Join and clean up
        content = " ".join(content_parts)
        
        # Clean up excessive whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', '\n', content)
        
        # Limit content length
        max_length = 10000
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        return content.strip()
    
    async def _extract_links(self, base_url: str, content: str) -> list[str]:
        """Extract links from content."""
        from bs4 import BeautifulSoup
        
        links = []
        
        try:
            # We need to re-fetch for link extraction
            # This is a simplified version
            import requests
            
            headers = {"User-Agent": self.user_agent}
            response = requests.get(base_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "lxml")
            
            base_domain = urlparse(base_url).netloc
            
            for link in soup.find_all("a", href=True):
                href = link["href"]
                
                # Skip anchors and empty links
                if not href or href.startswith("#"):
                    continue
                
                # Skip filtered patterns
                if any(pattern in href.lower() for pattern in self.config.exclude_patterns):
                    continue
                
                # Make absolute URL
                full_url = urljoin(base_url, href)
                
                # Only same-domain unless follow_external
                url_domain = urlparse(full_url).netloc
                if url_domain == base_domain or self.config.follow_external:
                    # Normalize URL
                    full_url = full_url.split("#")[0]  # Remove anchors
                    if full_url and full_url not in links:
                        links.append(full_url)
                        
        except Exception as e:
            logger.warning(f"Failed to extract links from {base_url}: {e}")
        
        return links
    
    async def _can_crawl(self, url: str) -> bool:
        """Check if URL can be crawled according to robots.txt."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Fetch robots.txt if not cached
        if robots_url not in self._robots_txt:
            try:
                import requests
                
                response = requests.get(
                    robots_url,
                    timeout=5,
                    headers={"User-Agent": self.user_agent}
                )
                
                if response.status_code == 200:
                    self._robots_txt[robots_url] = response.text
                else:
                    self._robots_txt[robots_url] = ""
                    
            except Exception:
                self._robots_txt[robots_url] = ""
        
        # Parse robots.txt
        robots_content = self._robots_txt.get(robots_url, "")
        
        if not robots_content:
            return True  # No robots.txt, allow
        
        # Simple robots.txt parsing
        allowed = True
        for line in robots_content.split("\n"):
            line = line.strip()
            
            # User-agent
            if line.lower().startswith("user-agent:"):
                # Check if we're the target
                pass
            
            # Disallow
            if line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path and path in url:
                    allowed = False
        
        return allowed
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResult:
        """
        Search within crawled documentation.
        
        Note: This searches cached/local content, not the web.
        """
        # For now, this is a placeholder
        # In full implementation, would search the knowledge graph
        return SearchResult(
            query=query,
            results=[],
            provider="local",
            total_results=0,
            execution_time=0,
        )
    
    async def fetch(self, url: str) -> Optional[Source]:
        """Fetch a specific page."""
        return await self._crawl_page(url)
    
    async def crawl_api_docs(
        self,
        api_name: str,
        base_url: str,
    ) -> list[Source]:
        """
        Crawl API documentation.
        
        Specialized crawler for common API documentation formats.
        
        Args:
            api_name: Name of the API
            base_url: Base URL of the documentation
            
        Returns:
            List of documentation pages
        """
        # For OpenAPI/Swagger docs
        openapi_urls = [
            f"{base_url}/openapi.json",
            f"{base_url}/openapi.yaml",
            f"{base_url}/swagger.json",
            f"{base_url}/api-docs",
        ]
        
        sources = []
        
        for url in openapi_urls:
            try:
                source = await self._crawl_page(url)
                if source and "openapi" in source.title.lower():
                    sources.append(source)
            except Exception:
                pass
        
        return sources
