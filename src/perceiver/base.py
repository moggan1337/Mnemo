"""
Base Perceiver class and common utilities.

Provides the foundation for all perception components including
web search, paper reading, and document crawling.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Represents a source of information."""
    url: str
    title: str
    content: str
    source_type: str  # web, paper, doc
    timestamp: float = field(default_factory=time.time)
    author: Optional[str] = None
    language: str = "en"
    reliability_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def domain(self) -> str:
        """Get the domain of the source."""
        try:
            return urlparse(self.url).netloc
        except Exception:
            return "unknown"
    
    @property
    def age_days(self) -> int:
        """Get the age of the source in days."""
        return int((time.time() - self.timestamp) / 86400)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "type": self.source_type,
            "timestamp": self.timestamp,
            "author": self.author,
            "language": self.language,
            "reliability_score": self.reliability_score,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """A search result from a perception action."""
    query: str
    results: list[Source]
    provider: str
    total_results: int
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if search was successful."""
        return len(self.results) > 0 and len(self.errors) == 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "provider": self.provider,
            "total_results": self.total_results,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }


class BasePerceiver(ABC):
    """
    Abstract base class for all perception components.
    
    Perception is the process of gathering information from the external
    world - the web, academic papers, documentation, and other sources.
    
    Subclasses must implement:
    - search(): Perform a search query
    - fetch(): Fetch a specific resource
    
    Common features:
    - Rate limiting and caching
    - Error handling and retry logic
    - Result normalization
    - Performance tracking
    """
    
    def __init__(
        self,
        user_agent: str = "Mnemo-Research-Agent/0.1",
        timeout: int = 30,
        max_retries: int = 3,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the perceiver.
        
        Args:
            user_agent: User agent string for HTTP requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_enabled: Whether to cache results
            cache_ttl: Cache time-to-live in seconds
        """
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        
        self._cache: dict[str, tuple[Any, float]] = {}
        self._stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_bytes": 0,
        }
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if valid."""
        if not self.cache_enabled:
            return None
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                self._stats["cache_hits"] += 1
                return value
            else:
                del self._cache[key]
        
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        if self.cache_enabled:
            self._cache[key] = (value, time.time())
    
    def _clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for information.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            SearchResult containing found sources
        """
        pass
    
    @abstractmethod
    async def fetch(self, url: str) -> Optional[Source]:
        """
        Fetch a specific resource.
        
        Args:
            url: URL of the resource to fetch
            
        Returns:
            Source object if successful, None otherwise
        """
        pass
    
    def _create_source(
        self,
        url: str,
        title: str,
        content: str,
        source_type: str,
        **kwargs
    ) -> Source:
        """Create a Source object with common initialization."""
        return Source(
            url=url,
            title=title,
            content=content,
            source_type=source_type,
            **kwargs
        )
    
    def _estimate_reliability(self, source: Source) -> float:
        """
        Estimate the reliability score of a source.
        
        Factors:
        - Domain reputation (academic > commercial > unknown)
        - Content length (longer generally more informative)
        - Recency
        """
        score = 0.5
        
        # Domain-based scoring
        domain = source.domain.lower()
        academic_domains = [
            "arxiv.org", "nature.com", "science.org", 
            "ieee.org", "acm.org", "springer.com",
            "plos.org", "wiley.com", "sciencedirect.com",
        ]
        trusted_domains = [
            "wikipedia.org", "github.com", "stackoverflow.com",
            "medium.com", "dev.to", "reddit.com",
        ]
        
        if any(d in domain for d in academic_domains):
            score = 0.9
        elif any(d in domain for d in trusted_domains):
            score = 0.7
        elif not domain or domain == "unknown":
            score = 0.3
        
        # Content length bonus
        if len(source.content) > 1000:
            score = min(1.0, score + 0.1)
        
        # Recency bonus
        if source.age_days < 7:
            score = min(1.0, score + 0.05)
        elif source.age_days > 365:
            score = max(0.2, score - 0.1)
        
        return score
    
    @property
    def statistics(self) -> dict:
        """Get perceiver statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["requests_made"])
            ),
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_bytes": 0,
        }
