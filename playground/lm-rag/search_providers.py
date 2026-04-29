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
            try:
                from ddgs import DDGS
            except ImportError:
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
# Dictionary Provider (Free Dictionary API — no key needed)
# ============================================================

class DictionaryProvider(SearchProvider):
    """Look up word/compound definitions from Free Dictionary API.

    Priority: check this BEFORE web search. It's free, fast, and
    gives structured definitions. If the dictionary knows the word,
    we don't need to hit the web.

    Also checks the local wordnet.db for definitions.
    """
    name = 'dictionary'

    def __init__(self, wordnet_db_path=None):
        self._db_path = wordnet_db_path
        self._conn = None

    def _get_conn(self):
        if self._conn is None and self._db_path and os.path.exists(self._db_path):
            import sqlite3
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        results = []

        # Step 1: Local wordnet.db lookup
        conn = self._get_conn()
        if conn:
            local = self._search_local(query, conn)
            if local:
                results.extend(local[:max_results])

        # Step 2: Free Dictionary API (if local didn't have it)
        if not results:
            api_results = self._search_api(query)
            results.extend(api_results[:max_results])

        return results

    def _search_local(self, query: str, conn) -> List[SearchResult]:
        """Search local wordnet.db for definitions."""
        cur = conn.cursor()
        results = []

        # Check compounds table first
        cur.execute(
            'SELECT compound, pos, definition FROM compounds '
            'WHERE compound=? AND pos != ?',
            (query.lower(), 'REJECTED')
        )
        for compound, pos, definition in cur.fetchall():
            if definition:
                results.append(SearchResult(
                    title=compound,
                    text=f'{compound} ({pos}): {definition}',
                    source='wordnet',
                    score=1.0,
                ))

        # Check senses table
        if not results:
            # For multi-word queries, try each word
            words = query.lower().split()
            for word in words:
                cur.execute(
                    'SELECT word, pos, definition FROM senses '
                    'WHERE word=? ORDER BY sense_num LIMIT 3',
                    (word,)
                )
                for w, pos, definition in cur.fetchall():
                    if definition:
                        results.append(SearchResult(
                            title=w,
                            text=f'{w} ({pos}): {definition}',
                            source='wordnet',
                            score=0.8,
                        ))

        return results

    def _search_api(self, query: str) -> List[SearchResult]:
        """Look up via Free Dictionary API (dictionaryapi.dev)."""
        results = []
        # Try the compound as-is, then individual words
        words_to_try = [query.replace(' ', '%20')]
        if ' ' in query:
            words_to_try.extend(query.split())

        for word in words_to_try:
            try:
                url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'LM-RAG/1.0'}
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read())

                if not isinstance(data, list):
                    continue

                for entry in data[:2]:
                    entry_word = entry.get('word', word)
                    meanings = entry.get('meanings', [])
                    for meaning in meanings[:2]:
                        pos = meaning.get('partOfSpeech', '')
                        defs = meaning.get('definitions', [])
                        for d in defs[:2]:
                            defn = d.get('definition', '')
                            if defn:
                                text = f'{entry_word} ({pos}): {defn}'
                                example = d.get('example', '')
                                if example:
                                    text += f' Example: {example}'
                                results.append(SearchResult(
                                    title=entry_word,
                                    text=text,
                                    source='dictionary',
                                    score=0.9,
                                ))
            except Exception:
                continue

        return results

    def define(self, word_or_compound: str) -> Optional[str]:
        """Quick definition lookup. Returns first definition or None.

        Use this from the thinker's reasoning loop for gap resolution.
        """
        results = self.search(word_or_compound, max_results=1)
        if results:
            return results[0].text
        return None


# ============================================================
# SearXNG Provider (self-hosted meta-search)
# ============================================================

class SearXNGProvider(SearchProvider):
    """Self-hosted SearXNG meta-search engine.

    Aggregates Google, Bing, DuckDuckGo, Wikipedia in one call.
    Runs locally — no API key, no rate limits, unlimited queries.
    """
    name = 'searxng'

    def __init__(self, base_url=None):
        self.base_url = base_url or 'http://127.0.0.1:8888'

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            params = urllib.parse.urlencode({
                'q': query, 'format': 'json', 'language': 'en',
            })
            req = urllib.request.Request(
                f'{self.base_url}/search?{params}',
                headers={'Accept': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())

            results = []
            for item in data.get('results', [])[:max_results]:
                text = item.get('content', '')
                if text:
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        text=text,
                        source=self.name,
                        article_id=item.get('url', ''),
                    ))
            return results
        except Exception:
            return []


# ============================================================
# Tavily Provider (AI-optimized search)
# ============================================================

class TavilyProvider(SearchProvider):
    """Tavily search — purpose-built for AI agents.

    Returns clean extracted text, not raw HTML snippets.
    Free tier: 1,000 queries/month.
    """
    name = 'tavily'

    def __init__(self):
        creds_path = os.path.expanduser('~/.claude/secrets/tavily.json')
        try:
            with open(creds_path) as f:
                creds = json.load(f)
            self.api_key = creds['api_key']
        except Exception:
            self.api_key = None

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        try:
            payload = json.dumps({
                'query': query,
                'max_results': min(max_results, 10),
                'search_depth': 'basic',
                'include_answer': True,
            }).encode()
            req = urllib.request.Request(
                'https://api.tavily.com/search',
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                }
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())

            results = []
            # Tavily returns a direct answer + individual results
            answer = data.get('answer', '')
            if answer:
                results.append(SearchResult(
                    title='Tavily Answer',
                    text=answer,
                    source=self.name,
                    score=0.95,
                ))
            for item in data.get('results', [])[:max_results]:
                text = item.get('content', '')
                if text:
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        text=text,
                        source=self.name,
                        article_id=item.get('url', ''),
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


def create_default_engine(wordnet_db_path=None) -> SearchEngine:
    """Create search engine with default providers.

    Priority order: Dictionary (free, fast) → Wikipedia → Brave → DuckDuckGo
    The dictionary provider checks wordnet.db first, then the free API.
    """
    engine = SearchEngine()
    # Dictionary — always first (free, fast, structured definitions)
    db_path = wordnet_db_path or os.path.join(
        os.path.dirname(__file__), 'data', 'vocab', 'wordnet.db'
    )
    engine.register(DictionaryProvider(wordnet_db_path=db_path))
    # Brave Search — best snippets, $5/mo free credit
    brave = BraveSearchProvider()
    if brave.api_key:
        engine.register(brave)
    # Wikipedia — always available, full article text
    engine.register(WikipediaProvider())
    # DuckDuckGo — fallback (often rate-limited)
    engine.register(DuckDuckGoProvider())
    return engine
