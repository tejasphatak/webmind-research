"""
Search Provider Protocol — MCP-like extensible search layer.

Each provider implements the same interface:
    .search(query) → List[SearchResult]
    .get_full(article_id) → str

Engine calls ALL registered providers → critic picks best result.
Adding a new source = implement the interface. No engine changes.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod


@dataclass
class SearchResult:
    """One search result from any provider."""
    title: str
    text: str                    # article content
    source: str                  # provider name
    article_id: Optional[str] = None  # for get_full()
    score: float = 0.0          # provider's relevance score


class SearchProvider(ABC):
    """Abstract search provider — implement this to add a new source."""
    name: str = 'unknown'

    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Search and return results."""
        pass

    def get_full(self, article_id: str) -> Optional[str]:
        """Get full article text by ID. Optional."""
        return None


# ============================================================
# Wikipedia Provider
# ============================================================

class WikipediaProvider(SearchProvider):
    name = 'wikipedia'

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        try:
            results = []

            # Strategy 1: opensearch (title matching — much more accurate)
            params = urllib.parse.urlencode({
                'action': 'opensearch',
                'search': query,
                'limit': max_results,
                'format': 'json',
            })
            req = urllib.request.Request(
                f'https://en.wikipedia.org/w/api.php?{params}',
                headers={'User-Agent': 'LM-RAG/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            titles = data[1] if len(data) >= 2 else []

            # Strategy 2: fallback to regular search if opensearch returns nothing
            if not titles:
                params2 = urllib.parse.urlencode({
                    'action': 'query', 'list': 'search',
                    'srsearch': query, 'srlimit': max_results, 'format': 'json',
                })
                req2 = urllib.request.Request(
                    f'https://en.wikipedia.org/w/api.php?{params2}',
                    headers={'User-Agent': 'LM-RAG/1.0'}
                )
                with urllib.request.urlopen(req2, timeout=10) as r2:
                    search_results = json.loads(r2.read()).get('query', {}).get('search', [])
                titles = [sr['title'] for sr in search_results]

            for title in titles[:max_results]:
                text = self.get_full(title)
                if text:
                    results.append(SearchResult(
                        title=title,
                        text=text,
                        source=self.name,
                        article_id=title,
                    ))
            return results
        except Exception as e:
            return []

    def get_links(self, article_id: str, max_links: int = 10) -> List[str]:
        """Get outgoing links from a Wikipedia article."""
        try:
            params = urllib.parse.urlencode({
                'action': 'query', 'titles': article_id, 'prop': 'links',
                'pllimit': max_links, 'plnamespace': 0, 'format': 'json',
            })
            req = urllib.request.Request(
                f'https://en.wikipedia.org/w/api.php?{params}',
                headers={'User-Agent': 'LM-RAG/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                pages = data.get('query', {}).get('pages', {})
                for pid, page in pages.items():
                    return [l['title'] for l in page.get('links', [])]
            return []
        except:
            return []

    def get_full(self, article_id: str, max_chars: int = 100000) -> Optional[str]:
        """Get Wikipedia article text — full content, not just summary.
        Fetches full article plaintext and truncates locally."""
        try:
            params = urllib.parse.urlencode({
                'action': 'query', 'titles': article_id, 'prop': 'extracts',
                'explaintext': 1, 'exlimit': 1, 'format': 'json',
                # No exchars — get full article, truncate locally
            })
            req = urllib.request.Request(
                f'https://en.wikipedia.org/w/api.php?{params}',
                headers={'User-Agent': 'LM-RAG/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                pages = data.get('query', {}).get('pages', {})
                for pid, page in pages.items():
                    extract = page.get('extract', '')
                    if extract:
                        return extract[:max_chars] if max_chars else extract
            return None
        except:
            return None


# ============================================================
# DuckDuckGo Provider
# ============================================================

class DuckDuckGoProvider(SearchProvider):
    name = 'duckduckgo'

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    body = r.get('body', '')
                    if body:
                        results.append(SearchResult(
                            title=r.get('title', ''),
                            text=body,
                            source=self.name,
                        ))
            return results
        except:
            return []


# ============================================================
# Brave Search Provider
# ============================================================

class BraveSearchProvider(SearchProvider):
    name = 'brave'

    def __init__(self):
        creds_path = os.path.expanduser('~/.claude/secrets/brave_search.json')
        try:
            with open(creds_path) as f:
                creds = json.load(f)
            self.api_key = creds['api_key']
        except Exception:
            self.api_key = None

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        if not self.api_key:
            return []
        try:
            params = urllib.parse.urlencode({
                'q': query, 'count': min(max_results, 20),
            })
            req = urllib.request.Request(
                f'https://api.search.brave.com/res/v1/web/search?{params}',
                headers={
                    'X-Subscription-Token': self.api_key,
                    'Accept': 'application/json',
                }
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())

            results = []
            for item in data.get('web', {}).get('results', [])[:max_results]:
                # Strip HTML tags from description
                desc = re.sub(r'<[^>]+>', '', item.get('description', ''))
                if desc:
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        text=desc,
                        source=self.name,
                        article_id=item.get('url', ''),
                    ))
            return results
        except Exception:
            return []


# ============================================================
# Google Custom Search Provider
# ============================================================

class GoogleSearchProvider(SearchProvider):
    name = 'google'

    def __init__(self):
        import json, os
        creds_path = os.path.expanduser('~/.claude/secrets/google_search.json')
        try:
            with open(creds_path) as f:
                creds = json.load(f)
            self.api_key = creds['api_key']
            self.cx = creds['cx']
        except Exception:
            self.api_key = None
            self.cx = None

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        if not self.api_key or not self.cx:
            return []
        try:
            params = urllib.parse.urlencode({
                'key': self.api_key, 'cx': self.cx,
                'q': query, 'num': min(max_results, 10),
            })
            req = urllib.request.Request(
                f'https://www.googleapis.com/customsearch/v1?{params}',
                headers={'User-Agent': 'LM-RAG/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())

            results = []
            for item in data.get('items', [])[:max_results]:
                text = item.get('snippet', '')
                # Also try to get page content from pagemap
                pagemap = item.get('pagemap', {})
                metatags = pagemap.get('metatags', [{}])
                description = metatags[0].get('og:description', '') if metatags else ''
                if description and len(description) > len(text):
                    text = description
                if text:
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        text=text,
                        source=self.name,
                        article_id=item.get('link', ''),
                    ))
            return results
        except Exception:
            return []


# ============================================================
# Synapse KB Provider (local FAISS)
# ============================================================

class SynapseKBProvider(SearchProvider):
    name = 'synapse_kb'

    def __init__(self, kb_path=None):
        self.kb_path = kb_path
        self.loaded = False
        # Lazy load — only load if called

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        # TODO: wire to Synapse's 305K KB via FAISS
        return []


# ============================================================
# Search Engine — orchestrates all providers
# ============================================================

class SearchEngine:
    """
    Orchestrates multiple search providers.
    Searches ALL providers, returns ALL results.
    The critic (in engine.py) picks the best one.
    """

    def __init__(self):
        self.providers: List[SearchProvider] = []

    def register(self, provider: SearchProvider):
        """Add a search provider."""
        self.providers.append(provider)
        return self

    def search(self, query: str, max_per_provider: int = 3) -> List[SearchResult]:
        """Search ALL providers, return combined results."""
        all_results = []
        for provider in self.providers:
            try:
                results = provider.search(query, max_results=max_per_provider)
                all_results.extend(results)
            except Exception as e:
                continue  # provider failed, skip
        return all_results

    def search_multi_query(self, queries: List[str], max_per_provider: int = 2) -> List[SearchResult]:
        """Search ALL providers with MULTIPLE queries, deduplicate."""
        all_results = []
        seen_texts = set()

        for query in queries:
            results = self.search(query, max_per_provider)
            for r in results:
                # Deduplicate by first 100 chars
                key = r.text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_results.append(r)

        return all_results


def create_default_engine() -> SearchEngine:
    """Create search engine with default providers."""
    engine = SearchEngine()
    # Brave Search — best snippets, $5/mo free credit
    brave = BraveSearchProvider()
    if brave.api_key:
        engine.register(brave)
    # Google Custom Search — DISABLED (API shut down globally)
    # Wikipedia — always available, full article text
    engine.register(WikipediaProvider())
    # DuckDuckGo — fallback (often rate-limited)
    engine.register(DuckDuckGoProvider())
    return engine
