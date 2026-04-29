"""
ULI MCP Client — generic MCP client for extending reasoning capabilities.

The system is both an MCP SERVER (api/mcp_server.py exposes our KG outward)
and an MCP CLIENT (this module calls external MCP servers inward).

Adding a new capability = add a stanza to data/system/config.yaml.
No code changes needed.

Transports:
  builtin       — calls uli/searcher.py internally (DuckDuckGo)
  sse           — JSON-RPC 2.0 POST to an MCP SSE endpoint (DeepWiki, etc.)
  stdio         — subprocess + stdin/stdout (local MCP servers)
  rest_adapter  — direct REST API calls with per-source adapters (Wikipedia, arXiv, etc.)

Usage:
    from uli.system_prompt import SystemConfig
    from uli.mcp_client import MCPClient

    cfg = SystemConfig.load()
    client = MCPClient(cfg.settings.mcp_servers)
    result = client.call_capability('research', {'query': 'attention mechanism transformers'})
"""

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any

log = logging.getLogger('uli.mcp_client')


class MCPClient:
    """
    Generic MCP client. Routes calls by capability tag.
    Tries registered servers in config order. Returns first successful result.
    All errors are caught — never crashes the reasoner.
    """

    def __init__(self, servers):
        # Accept MCPServerConfig objects or plain dicts
        self._servers = [s for s in servers if getattr(s, 'enabled', True)]

    def servers_for(self, capability: str):
        """Return enabled servers that handle the given capability, in order."""
        return [s for s in self._servers if capability in s.capabilities]

    def call_capability(self, capability: str, args: dict) -> Optional[str]:
        """
        Try all servers for this capability in config order.
        Returns first non-empty result, or None if all fail.
        """
        for server in self.servers_for(capability):
            try:
                result = self._dispatch(server, args)
                if result and result.strip():
                    log.debug(f"MCP hit: {server.name} for capability={capability}")
                    return result
            except Exception as e:
                log.debug(f"MCP {server.name} failed: {e}")
        return None

    def _dispatch(self, server, args: dict) -> Optional[str]:
        """Route to correct transport handler."""
        t = server.transport
        if t == 'builtin':
            return self._call_builtin(server, args)
        elif t == 'sse':
            return self._call_sse(server, args)
        elif t == 'stdio':
            return self._call_stdio(server, args)
        elif t == 'rest_adapter':
            return self._call_rest_adapter(server, args)
        else:
            log.warning(f"Unknown transport: {t}")
            return None

    # ── Builtin transport (DuckDuckGo via uli/searcher.py) ─────────────────

    def _call_builtin(self, server, args: dict) -> Optional[str]:
        from uli.searcher import Searcher
        query = args.get('query', '')
        if not query:
            return None
        return Searcher().search(query)

    # ── SSE transport (MCP JSON-RPC 2.0) ────────────────────────────────────

    def _call_sse(self, server, args: dict) -> Optional[str]:
        """Call an MCP server via HTTP POST (JSON-RPC 2.0)."""
        if not server.url:
            return None

        # Check API key if required
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if server.api_key_env:
            key = os.environ.get(server.api_key_env, '')
            if not key:
                log.debug(f"Skipping {server.name}: {server.api_key_env} not set")
                return None
            headers['Authorization'] = f'Bearer {key}'

        tool = server.tools[0] if server.tools else 'search'
        payload = json.dumps({
            'jsonrpc': '2.0',
            'method': 'tools/call',
            'params': {'name': tool, 'arguments': args},
            'id': 1,
        }).encode('utf-8')

        req = urllib.request.Request(server.url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=server.timeout_seconds) as resp:
            data = json.loads(resp.read())

        # Extract text from MCP response
        result = data.get('result', {})
        content = result.get('content', [])
        if isinstance(content, list):
            parts = [c.get('text', '') for c in content if c.get('type') == 'text']
            return '\n'.join(parts) or None
        if isinstance(content, str):
            return content or None
        return None

    # ── Stdio transport (local MCP servers via subprocess) ───────────────────

    def _call_stdio(self, server, args: dict) -> Optional[str]:
        """Call a local MCP server via subprocess stdin/stdout."""
        import subprocess
        if not server.url:  # url = command for stdio
            return None
        tool = server.tools[0] if server.tools else 'search'
        payload = json.dumps({
            'jsonrpc': '2.0',
            'method': 'tools/call',
            'params': {'name': tool, 'arguments': args},
            'id': 1,
        })
        proc = subprocess.run(
            server.url.split(),
            input=payload, capture_output=True, text=True,
            timeout=server.timeout_seconds,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        content = data.get('result', {}).get('content', [])
        if isinstance(content, list):
            return '\n'.join(c.get('text', '') for c in content if c.get('type') == 'text') or None
        return str(content) or None

    # ── REST adapter (per-source implementations) ────────────────────────────

    def _call_rest_adapter(self, server, args: dict) -> Optional[str]:
        """Dispatch to per-source REST adapter."""
        name = server.name
        if name == 'wikipedia':
            return self._wikipedia(server, args)
        elif name == 'arxiv':
            return self._arxiv(server, args)
        elif name == 'semantic_scholar':
            return self._semantic_scholar(server, args)
        elif name == 'pubmed':
            return self._pubmed(server, args)
        elif name == 'openalex':
            return self._openalex(server, args)
        elif name == 'crossref':
            return self._crossref(server, args)
        elif name == 'dblp':
            return self._dblp(server, args)
        elif name == 'huggingface':
            return self._huggingface(server, args)
        elif name == 'internet_archive':
            return self._internet_archive(server, args)
        elif name == 'brave-search':
            return self._brave_search(server, args)
        elif name == 'nasa_ads':
            return self._nasa_ads(server, args)
        elif name == 'searxng':
            return self._searxng(server, args)
        elif name == 'tavily':
            return self._tavily(server, args)
        else:
            log.warning(f"No REST adapter for: {name}")
            return None

    def _get(self, url: str, timeout: int, headers: dict = None) -> dict:
        """Helper: HTTP GET → parsed JSON."""
        req = urllib.request.Request(url, headers=headers or {'User-Agent': 'ULI/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    # ── Wikipedia ──────────────────────────────────────────────────────────

    def _wikipedia(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        try:
            from search_providers import WikipediaProvider
            provider = WikipediaProvider()
            results = provider.search(query, max_results=2)
            if not results:
                return None
            parts = [f"[Wikipedia: {r.title}]\n{r.text[:800]}" for r in results]
            return '\n\n'.join(parts)
        except ImportError:
            pass
        # Fallback: direct API
        params = urllib.parse.urlencode({
            'action': 'query', 'list': 'search',
            'srsearch': query, 'srlimit': 3, 'format': 'json',
        })
        data = self._get(
            f"https://en.wikipedia.org/w/api.php?{params}",
            timeout=server.timeout_seconds,
        )
        items = data.get('query', {}).get('search', [])
        if not items:
            return None
        parts = []
        for item in items[:3]:
            snippet = re.sub(r'<[^>]+>', '', item.get('snippet', ''))
            parts.append(f"[Wikipedia: {item['title']}]\n{snippet}")
        return '\n\n'.join(parts)

    # ── arXiv ──────────────────────────────────────────────────────────────

    def _arxiv(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'search_query': f'all:{query}',
            'start': 0, 'max_results': 5,
            'sortBy': 'relevance', 'sortOrder': 'descending',
        })
        try:
            data = self._get(f"{server.url}?{params}", timeout=server.timeout_seconds)
            # arXiv returns Atom XML — parse it
            text = data if isinstance(data, str) else ''
        except Exception:
            # Fall back to raw HTTP for XML
            req = urllib.request.Request(
                f"{server.url}?{params}",
                headers={'User-Agent': 'ULI/1.0'}
            )
            with urllib.request.urlopen(req, timeout=server.timeout_seconds) as resp:
                text = resp.read().decode('utf-8')

        # Extract entries from Atom XML
        entries = re.findall(
            r'<entry>(.*?)</entry>', text, re.DOTALL
        )
        if not entries:
            return None
        parts = []
        for entry in entries[:5]:
            title   = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            authors = re.findall(r'<name>(.*?)</name>', entry)
            pub     = re.search(r'<published>(.*?)</published>', entry)
            t = (title.group(1).strip() if title else 'Unknown').replace('\n', ' ')
            s = (summary.group(1).strip() if summary else '')[:400]
            a = ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '')
            p = (pub.group(1)[:10] if pub else '')
            parts.append(f"[arXiv {p}] {t}\nAuthors: {a}\n{s}")
        return '\n\n'.join(parts)

    # ── Semantic Scholar ───────────────────────────────────────────────────

    def _semantic_scholar(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'query': query, 'limit': 5,
            'fields': 'title,abstract,year,authors,citationCount,openAccessPdf',
        })
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
            headers={'User-Agent': 'ULI/1.0'}
        )
        papers = data.get('data', [])
        if not papers:
            return None
        parts = []
        for p in papers[:5]:
            title    = p.get('title', 'Unknown')
            abstract = (p.get('abstract') or '')[:400]
            year     = p.get('year', '')
            authors  = [a.get('name', '') for a in p.get('authors', [])[:3]]
            cites    = p.get('citationCount', 0)
            a_str    = ', '.join(authors) + (' et al.' if len(p.get('authors', [])) > 3 else '')
            parts.append(
                f"[Semantic Scholar {year}] {title}\n"
                f"Authors: {a_str} | Citations: {cites}\n{abstract}"
            )
        return '\n\n'.join(parts)

    # ── PubMed ────────────────────────────────────────────────────────────

    def _pubmed(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        # Step 1: esearch → get IDs
        search_params = urllib.parse.urlencode({
            'db': 'pubmed', 'term': query, 'retmax': 5,
            'retmode': 'json', 'sort': 'relevance',
        })
        search_data = self._get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}",
            timeout=server.timeout_seconds,
        )
        ids = search_data.get('esearchresult', {}).get('idlist', [])
        if not ids:
            return None
        # Step 2: efetch → get abstracts
        fetch_params = urllib.parse.urlencode({
            'db': 'pubmed', 'id': ','.join(ids[:5]),
            'retmode': 'json', 'rettype': 'abstract',
        })
        fetch_data = self._get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{fetch_params}",
            timeout=server.timeout_seconds,
        )
        # PubMed efetch returns complex structure — extract what we can
        articles = (fetch_data.get('PubmedArticleSet', {})
                              .get('PubmedArticle', []))
        if not articles:
            return f"[PubMed] Found {len(ids)} results for: {query}\nIDs: {', '.join(ids[:5])}"
        parts = []
        for art in articles[:5]:
            citation = art.get('MedlineCitation', {})
            article  = citation.get('Article', {})
            title    = article.get('ArticleTitle', 'Unknown')
            abstract = article.get('Abstract', {}).get('AbstractText', '')
            if isinstance(abstract, list):
                abstract = ' '.join(str(a) for a in abstract)
            abstract = str(abstract)[:400]
            parts.append(f"[PubMed] {title}\n{abstract}")
        return '\n\n'.join(parts) if parts else f"[PubMed] {len(ids)} results for: {query}"

    # ── OpenAlex ──────────────────────────────────────────────────────────

    def _openalex(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'search': query, 'per-page': 5,
            'select': 'title,abstract_inverted_index,publication_year,authorships,cited_by_count',
            'mailto': 'uli-reasoner@local',   # polite pool
        })
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
        )
        works = data.get('results', [])
        if not works:
            return None
        parts = []
        for w in works[:5]:
            title = w.get('display_name') or w.get('title', 'Unknown')
            year  = w.get('publication_year', '')
            cites = w.get('cited_by_count', 0)
            authors = [a.get('author', {}).get('display_name', '') for a in w.get('authorships', [])[:3]]
            a_str = ', '.join(authors)
            parts.append(f"[OpenAlex {year}] {title}\nAuthors: {a_str} | Citations: {cites}")
        return '\n\n'.join(parts)

    # ── CrossRef ──────────────────────────────────────────────────────────

    def _crossref(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'query': query, 'rows': 5,
            'select': 'title,author,published,DOI,abstract',
            'mailto': 'uli-reasoner@local',
        })
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
        )
        items = data.get('message', {}).get('items', [])
        if not items:
            return None
        parts = []
        for item in items[:5]:
            titles  = item.get('title', ['Unknown'])
            title   = titles[0] if titles else 'Unknown'
            authors = item.get('author', [])
            a_str   = ', '.join(
                f"{a.get('family', '')}, {a.get('given', '')[:1]}."
                for a in authors[:3]
            )
            pub = item.get('published', {}).get('date-parts', [['']])[0]
            year = str(pub[0]) if pub else ''
            doi  = item.get('DOI', '')
            parts.append(f"[CrossRef {year}] {title}\nAuthors: {a_str}\nDOI: {doi}")
        return '\n\n'.join(parts)

    # ── DBLP ──────────────────────────────────────────────────────────────

    def _dblp(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'q': query, 'format': 'json', 'h': 5,
        })
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
        )
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        if not hits:
            return None
        parts = []
        for hit in hits[:5]:
            info = hit.get('info', {})
            title   = info.get('title', 'Unknown')
            year    = info.get('year', '')
            authors = info.get('authors', {}).get('author', [])
            if isinstance(authors, str):
                authors = [authors]
            elif isinstance(authors, dict):
                authors = [authors.get('text', '')]
            a_str = ', '.join(str(a) for a in authors[:3])
            venue = info.get('venue', '')
            parts.append(f"[DBLP {year}] {title}\nAuthors: {a_str} | Venue: {venue}")
        return '\n\n'.join(parts)

    # ── Hugging Face Papers ────────────────────────────────────────────────

    def _huggingface(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        # HF papers API: search
        params = urllib.parse.urlencode({'q': query, 'limit': 5})
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
            headers={'User-Agent': 'ULI/1.0'},
        )
        if not isinstance(data, list):
            data = data.get('papers', data.get('results', []))
        if not data:
            return None
        parts = []
        for p in data[:5]:
            title    = p.get('title', 'Unknown')
            abstract = (p.get('summary') or p.get('abstract') or '')[:400]
            pub      = p.get('publishedAt', '')[:10]
            parts.append(f"[HuggingFace Papers {pub}] {title}\n{abstract}")
        return '\n\n'.join(parts)

    # ── Internet Archive ───────────────────────────────────────────────────

    def _internet_archive(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'q': query, 'output': 'json', 'rows': 5,
            'fl[]': 'identifier,title,description,date,creator',
            'sort[]': 'downloads desc',
        })
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
        )
        docs = data.get('response', {}).get('docs', [])
        if not docs:
            return None
        parts = []
        for doc in docs[:5]:
            title = doc.get('title', 'Unknown')
            desc  = (doc.get('description') or '')[:300]
            date  = (doc.get('date') or '')[:10]
            creator = doc.get('creator', '')
            parts.append(f"[Internet Archive {date}] {title}\nBy: {creator}\n{desc}")
        return '\n\n'.join(parts)

    # ── Brave Search ──────────────────────────────────────────────────────

    def _brave_search(self, server, args: dict) -> Optional[str]:
        api_key = os.environ.get(server.api_key_env or 'BRAVE_API_KEY', '')
        if not api_key:
            return None
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({'q': query, 'count': 5})
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
            headers={'X-Subscription-Token': api_key, 'Accept': 'application/json'},
        )
        results = data.get('web', {}).get('results', [])
        if not results:
            return None
        parts = []
        for r in results[:5]:
            title = r.get('title', '')
            desc  = re.sub(r'<[^>]+>', '', r.get('description', ''))[:400]
            parts.append(f"[Brave] {title}\n{desc}")
        return '\n\n'.join(parts)

    # ── NASA ADS ──────────────────────────────────────────────────────────

    def _nasa_ads(self, server, args: dict) -> Optional[str]:
        token = os.environ.get(server.api_key_env or 'NASA_ADS_TOKEN', '')
        if not token:
            return None
        query = args.get('query', '')
        if not query:
            return None
        params = urllib.parse.urlencode({
            'q': query, 'fl': 'title,abstract,author,year,citation_count',
            'rows': 5, 'sort': 'citation_count desc',
        })
        data = self._get(
            f"{server.url}?{params}", timeout=server.timeout_seconds,
            headers={'Authorization': f'Bearer {token}'},
        )
        docs = data.get('response', {}).get('docs', [])
        if not docs:
            return None
        parts = []
        for doc in docs[:5]:
            title   = (doc.get('title') or ['Unknown'])[0]
            abstract = (doc.get('abstract') or '')[:400]
            authors  = doc.get('author', [])[:3]
            year     = doc.get('year', '')
            cites    = doc.get('citation_count', 0)
            a_str    = ', '.join(authors)
            parts.append(f"[NASA ADS {year}] {title}\nAuthors: {a_str} | Citations: {cites}\n{abstract}")
        return '\n\n'.join(parts)

    # ── SearXNG (self-hosted meta-search) ──────────────────────────────────

    def _searxng(self, server, args: dict) -> Optional[str]:
        query = args.get('query', '')
        if not query:
            return None
        try:
            from search_providers import SearXNGProvider
            provider = SearXNGProvider(base_url=server.url)
            results = provider.search(query, max_results=5)
            if not results:
                return None
            parts = [f"[{r.title}]\n{r.text[:500]}" for r in results]
            return '\n\n'.join(parts)
        except Exception as e:
            log.debug(f"SearXNG failed: {e}")
            return None

    # ── Tavily (AI-optimized search) ───────────────────────────────────────

    def _tavily(self, server, args: dict) -> Optional[str]:
        token = os.environ.get(server.api_key_env or 'TAVILY_API_KEY', '')
        if not token:
            # Also check secrets file
            try:
                import json as _json
                with open(os.path.expanduser('~/.claude/secrets/tavily.json')) as f:
                    token = _json.load(f).get('api_key', '')
            except Exception:
                pass
        if not token:
            return None
        query = args.get('query', '')
        if not query:
            return None
        try:
            from search_providers import TavilyProvider
            provider = TavilyProvider()
            # Override with env token if available
            if token:
                provider.api_key = token
            results = provider.search(query, max_results=5)
            if not results:
                return None
            parts = [f"[{r.title}]\n{r.text[:500]}" for r in results]
            return '\n\n'.join(parts)
        except Exception as e:
            log.debug(f"Tavily failed: {e}")
            return None
