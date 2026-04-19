"""
Paper Reader for accessing academic papers.

Supports arXiv, Semantic Scholar, and other academic databases.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mnemo.perceiver.base import BasePerceiver, SearchResult, Source

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata for an academic paper."""
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: Optional[str] = None
    updated_date: Optional[str] = None
    categories: list[str] = None
    doi: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
    
    @property
    def year(self) -> Optional[int]:
        """Extract publication year."""
        if self.published_date:
            try:
                return int(self.published_date[:4])
            except (ValueError, IndexError):
                return None
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published_date": self.published_date,
            "categories": self.categories,
            "doi": self.doi,
            "citation_count": self.citation_count,
            "year": self.year,
        }


class PaperReader(BasePerceiver):
    """
    Academic paper reader supporting multiple sources.
    
    Sources:
    - arXiv: Free access to preprints
    - Semantic Scholar: Academic paper search with citations
    
    Features:
    - Full-text search
    - Metadata extraction
    - Citation tracking
    - PDF downloading
    - Related paper discovery
    """
    
    def __init__(
        self,
        download_dir: Optional[Path] = None,
        use_arxiv: bool = True,
        use_semantic_scholar: bool = False,
        semantic_scholar_key: Optional[str] = None,
    ):
        """
        Initialize paper reader.
        
        Args:
            download_dir: Directory for downloaded PDFs
            use_arxiv: Enable arXiv access
            use_semantic_scholar: Enable Semantic Scholar
            semantic_scholar_key: API key for Semantic Scholar
        """
        super().__init__(
            cache_enabled=True,
            cache_ttl=7200,  # Papers change less frequently
        )
        
        self.download_dir = download_dir or Path.home() / ".mnemo" / "papers"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_arxiv = use_arxiv
        self.use_semantic_scholar = use_semantic_scholar
        self.semantic_scholar_key = semantic_scholar_key
        
        # Cache for paper metadata
        self._paper_cache: dict[str, PaperMetadata] = {}
        
        logger.info("PaperReader initialized")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> SearchResult:
        """
        Search for academic papers.
        
        Args:
            query: Search query
            max_results: Maximum number of papers
            categories: Filter by arXiv categories
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            
        Returns:
            SearchResult with paper sources
        """
        start_time = time.time()
        all_sources = []
        errors = []
        
        if self.use_arxiv:
            try:
                arxiv_sources = await self._search_arxiv(
                    query, max_results, categories, date_from, date_to
                )
                all_sources.extend(arxiv_sources)
            except Exception as e:
                logger.error(f"arXiv search error: {e}")
                errors.append(f"arXiv: {str(e)}")
        
        if self.use_semantic_scholar:
            try:
                ss_sources = await self._search_semantic_scholar(
                    query, max_results
                )
                all_sources.extend(ss_sources)
            except Exception as e:
                logger.error(f"Semantic Scholar search error: {e}")
                errors.append(f"Semantic Scholar: {str(e)}")
        
        # Sort by citation count (if available)
        all_sources.sort(
            key=lambda s: s.metadata.get("citation_count", 0),
            reverse=True
        )
        
        return SearchResult(
            query=query,
            results=all_sources[:max_results],
            provider="arxiv+semantic_scholar",
            total_results=len(all_sources),
            execution_time=time.time() - start_time,
            errors=errors,
        )
    
    async def _search_arxiv(
        self,
        query: str,
        max_results: int,
        categories: Optional[list[str]],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[Source]:
        """Search arXiv for papers."""
        sources = []
        
        try:
            import arxiv
            
            # Build search client
            client = arxiv.Client()
            
            # Build search query
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            
            if categories:
                category_query = " AND ".join(f'cat:{c}' for c in categories)
                search = arxiv.Search(
                    query=f"({query}) AND ({category_query})",
                    max_results=max_results,
                )
            
            # Execute search
            results = client.results(search)
            
            for result in results:
                # Extract abstract
                abstract = result.summary.replace("\n", " ")
                
                source = self._create_source(
                    url=result.entry_id,
                    title=result.title,
                    content=abstract,
                    source_type="paper",
                    author=", ".join(a.name for a in result.authors),
                    metadata={
                        "paper_id": result.entry_id.split("/")[-1],
                        "categories": result.categories,
                        "published": str(result.published.date()),
                        "updated": str(result.updated.date()),
                        "doi": result.doi,
                        "pdf_url": result.pdf_url,
                        "comment": result.comment,
                    }
                )
                
                source.reliability_score = 0.95  # arXiv is reliable
                
                # Cache metadata
                self._paper_cache[source.metadata["paper_id"]] = PaperMetadata(
                    paper_id=source.metadata["paper_id"],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=abstract,
                    published_date=str(result.published.date()),
                    categories=result.categories,
                    doi=result.doi,
                )
                
                sources.append(source)
                
        except ImportError:
            logger.warning("arxiv package not installed")
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
        
        return sources
    
    async def _search_semantic_scholar(
        self,
        query: str,
        max_results: int,
    ) -> list[Source]:
        """Search Semantic Scholar for papers."""
        sources = []
        
        try:
            import requests
            
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            headers = {}
            if self.semantic_scholar_key:
                headers["x-api-key"] = self.semantic_scholar_key
            
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,year,citationCount,openAccessPdf,externalIds",
            }
            
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            
            for paper in data.get("data", []):
                # Get PDF URL
                pdf_url = None
                if oa_pdf := paper.get("openAccessPdf"):
                    pdf_url = oa_pdf.get("url")
                
                # Get external IDs
                external_ids = paper.get("externalIds", {})
                arxiv_id = external_ids.get("ArXiv")
                
                source = self._create_source(
                    url=f"https://www.semanticscholar.org/paper/{paper['paperId']}",
                    title=paper.get("title", ""),
                    content=paper.get("abstract", ""),
                    source_type="paper",
                    author=", ".join(a.get("name", "") for a in paper.get("authors", [])),
                    metadata={
                        "paper_id": paper["paperId"],
                        "year": paper.get("year"),
                        "citation_count": paper.get("citationCount", 0),
                        "arxiv_id": arxiv_id,
                        "pdf_url": pdf_url,
                    }
                )
                
                source.reliability_score = 0.9
                sources.append(source)
                
        except ImportError:
            logger.warning("requests not available for Semantic Scholar")
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
        
        return sources
    
    async def fetch(self, url: str) -> Optional[Source]:
        """
        Fetch a paper by URL.
        
        Supports arXiv IDs and Semantic Scholar IDs.
        """
        # Parse paper ID from URL
        paper_id = self._extract_paper_id(url)
        
        if not paper_id:
            return None
        
        # Check cache
        if cached := self._paper_cache.get(paper_id):
            return self._create_source(
                url=url,
                title=cached.title,
                content=cached.abstract,
                source_type="paper",
                author=", ".join(cached.authors),
                metadata=cached.to_dict(),
            )
        
        # Fetch from arXiv
        if "arxiv.org" in url.lower():
            return await self._fetch_arxiv(paper_id)
        
        return None
    
    async def _fetch_arxiv(self, paper_id: str) -> Optional[Source]:
        """Fetch a paper from arXiv by ID."""
        try:
            import arxiv
            
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            results = list(client.results(search))
            
            if not results:
                return None
            
            result = results[0]
            abstract = result.summary.replace("\n", " ")
            
            source = self._create_source(
                url=result.entry_id,
                title=result.title,
                content=abstract,
                source_type="paper",
                author=", ".join(a.name for a in result.authors),
                metadata={
                    "paper_id": paper_id,
                    "categories": result.categories,
                    "published": str(result.published.date()),
                    "pdf_url": result.pdf_url,
                }
            )
            
            source.reliability_score = 0.95
            return source
            
        except Exception as e:
            logger.error(f"Failed to fetch arXiv paper {paper_id}: {e}")
            return None
    
    def _extract_paper_id(self, url: str) -> Optional[str]:
        """Extract paper ID from URL."""
        url_lower = url.lower()
        
        # arXiv format: https://arxiv.org/abs/2301.00001
        if "arxiv.org/abs/" in url_lower:
            return url.split("/abs/")[-1].split("/")[0].split(".")[0]
        
        # arXiv format: https://arxiv.org/pdf/2301.00001.pdf
        if "arxiv.org/pdf/" in url_lower:
            return url.split("/pdf/")[-1].split(".pdf")[0].split("/")[0]
        
        # Semantic Scholar format
        if "semanticscholar.org/paper/" in url_lower:
            parts = url.split("/paper/")
            if len(parts) > 1:
                return parts[-1].split("/")[0]
        
        return None
    
    async def get_paper_pdf(
        self,
        paper_id: str,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Download paper PDF.
        
        Args:
            paper_id: Paper ID (arXiv ID)
            output_path: Output file path
            
        Returns:
            Path to downloaded PDF
        """
        output_path = output_path or self.download_dir / f"{paper_id}.pdf"
        
        try:
            import arxiv
            
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            results = list(client.results(search))
            
            if not results:
                return None
            
            # Download PDF
            result = results[0]
            result.download_pdf(dirpath=self.download_dir, filename=f"{paper_id}.pdf")
            
            actual_path = self.download_dir / f"{paper_id}.pdf"
            
            if actual_path.exists():
                logger.info(f"Downloaded PDF: {actual_path}")
                return actual_path
            
        except Exception as e:
            logger.error(f"Failed to download PDF for {paper_id}: {e}")
        
        return None
    
    async def get_references(
        self,
        paper_id: str,
        max_references: int = 20,
    ) -> list[PaperMetadata]:
        """
        Get references/citations for a paper.
        
        Args:
            paper_id: Paper ID
            max_references: Maximum references to return
            
        Returns:
            List of referenced papers
        """
        references = []
        
        try:
            import requests
            
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
            
            headers = {}
            if self.semantic_scholar_key:
                headers["x-api-key"] = self.semantic_scholar_key
            
            params = {
                "limit": max_references,
                "fields": "title,authors,abstract,year,citationCount",
            }
            
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            
            for ref in data.get("data", []):
                citing = ref.get("citedBy", {})
                references.append(PaperMetadata(
                    paper_id=citing.get("paperId", ""),
                    title=citing.get("title", ""),
                    authors=[a.get("name", "") for a in citing.get("authors", [])],
                    abstract=citing.get("abstract", ""),
                    published_date=str(citing.get("year")) if citing.get("year") else None,
                    citation_count=citing.get("citationCount", 0),
                ))
                
        except Exception as e:
            logger.error(f"Failed to get references for {paper_id}: {e}")
        
        return references
    
    async def get_citations(
        self,
        paper_id: str,
        max_citations: int = 20,
    ) -> list[PaperMetadata]:
        """
        Get papers that cite this paper.
        
        Args:
            paper_id: Paper ID
            max_citations: Maximum citations to return
            
        Returns:
            List of citing papers
        """
        citations = []
        
        try:
            import requests
            
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            
            headers = {}
            if self.semantic_scholar_key:
                headers["x-api-key"] = self.semantic_scholar_key
            
            params = {
                "limit": max_citations,
                "fields": "title,authors,abstract,year,citationCount",
            }
            
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            
            for cit in data.get("data", []):
                citing = cit.get("citingPaper", {})
                citations.append(PaperMetadata(
                    paper_id=citing.get("paperId", ""),
                    title=citing.get("title", ""),
                    authors=[a.get("name", "") for a in citing.get("authors", [])],
                    abstract=citing.get("abstract", ""),
                    published_date=str(citing.get("year")) if citing.get("year") else None,
                    citation_count=citing.get("citationCount", 0),
                ))
                
        except Exception as e:
            logger.error(f"Failed to get citations for {paper_id}: {e}")
        
        return citations
