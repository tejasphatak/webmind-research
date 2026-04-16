"""
Citation Validator — queries arXiv API to verify every citation in every paper/finding.

Per CONVENTIONS.md, no citation enters a paper without primary-source verification.
This script is the enforcement tool.

Usage:
    python tools/validate_citations.py                 # audit all papers/findings
    python tools/validate_citations.py --fix           # write fixes back to source files
    python tools/validate_citations.py --files papers/carrier-payload-v1.md

Validates:
- arXiv ID exists and is accessible
- Title (case-insensitive substring match) appears on arXiv
- First-author last name matches
- Year consistent

Writes: papers/CITATIONS_AUDIT.md with verified status for every citation.

Author: Claude Opus 4.6
Date: 2026-04-16
"""
from __future__ import annotations
import re, sys, json, time, urllib.request, urllib.parse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

ARXIV_API = "http://export.arxiv.org/api/query"


@dataclass
class Citation:
    source_file: str
    line: int
    raw_text: str
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    # Validation results
    verified: bool = False
    arxiv_title: Optional[str] = None
    arxiv_authors: Optional[list] = None
    arxiv_year: Optional[int] = None
    issues: list = field(default_factory=list)


# ── arXiv ID extraction ───────────────────────────────────────────────

ARXIV_RE = re.compile(r'arXiv[:\s]*(?P<id>\d{4}\.\d{4,5})', re.IGNORECASE)
TITLE_RE = re.compile(r'"([^"]+)"')
YEAR_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')


def extract_citations(markdown_path: Path) -> list[Citation]:
    """Pull references from a markdown file's references section."""
    text = markdown_path.read_text()
    lines = text.splitlines()
    cites = []

    # Find numbered references like [1] Author. "Title." venue. year.
    ref_pattern = re.compile(r'^\[(\d+)\]\s+(.+)$')
    for i, line in enumerate(lines):
        m = ref_pattern.match(line.strip())
        if not m:
            continue
        raw = m.group(2)
        c = Citation(source_file=str(markdown_path), line=i + 1, raw_text=raw)
        # Extract arXiv
        ax = ARXIV_RE.search(raw)
        if ax:
            c.arxiv_id = ax.group("id")
        # Extract title
        t = TITLE_RE.search(raw)
        if t:
            c.title = t.group(1)
        # Extract year
        y = YEAR_RE.search(raw)
        if y:
            c.year = int(y.group(1))
        cites.append(c)
    return cites


# ── arXiv API query ───────────────────────────────────────────────────

def query_arxiv(arxiv_id: str) -> Optional[dict]:
    """Query arXiv API for metadata by ID. Returns dict with title/authors/year or None."""
    url = f"{ARXIV_API}?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "citation-validator/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
    except Exception as e:
        return {"error": str(e)}

    # Parse the Atom XML minimally
    # Title
    title_m = re.search(r'<entry>.*?<title>(.+?)</title>', body, re.DOTALL)
    title = re.sub(r"\s+", " ", title_m.group(1).strip()) if title_m else None
    # Authors
    author_matches = re.findall(r'<author>\s*<name>(.+?)</name>', body)
    # Published year
    pub_m = re.search(r'<entry>.*?<published>(\d{4})', body, re.DOTALL)
    year = int(pub_m.group(1)) if pub_m else None
    return {"title": title, "authors": author_matches, "year": year}


# ── Validation ────────────────────────────────────────────────────────

def validate(cite: Citation) -> Citation:
    if not cite.arxiv_id:
        cite.issues.append("NO_ARXIV_ID")
        return cite
    meta = query_arxiv(cite.arxiv_id)
    if not meta or meta.get("error"):
        cite.issues.append(f"ARXIV_FETCH_FAILED: {meta.get('error', 'unknown')}")
        return cite
    cite.arxiv_title = meta["title"]
    cite.arxiv_authors = meta["authors"]
    cite.arxiv_year = meta["year"]
    # Title check (case-insensitive substring)
    if cite.title and meta["title"]:
        ct = re.sub(r"\W+", "", cite.title.lower())
        at = re.sub(r"\W+", "", meta["title"].lower())
        if ct not in at and at not in ct:
            cite.issues.append(f"TITLE_MISMATCH: claimed '{cite.title}', arxiv '{meta['title']}'")
    # Year check
    if cite.year and meta["year"] and abs(cite.year - meta["year"]) > 1:
        cite.issues.append(f"YEAR_MISMATCH: claimed {cite.year}, arxiv {meta['year']}")
    # Author last-name check (first author)
    if cite.raw_text and meta["authors"]:
        # Extract last names from raw text — look for "surname," or "surname et al" pattern
        first_arxiv_lastname = meta["authors"][0].split()[-1].lower()
        if first_arxiv_lastname not in cite.raw_text.lower():
            cite.issues.append(f"FIRST_AUTHOR_MISSING: '{meta['authors'][0]}' not in cite")
    if not cite.issues:
        cite.verified = True
    return cite


# ── Main ──────────────────────────────────────────────────────────────

def audit_files(paths: list[Path]) -> list[Citation]:
    all_cites = []
    for p in paths:
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        cites = extract_citations(p)
        print(f"  {p.name}: {len(cites)} citations")
        for c in cites:
            validated = validate(c)
            all_cites.append(validated)
            time.sleep(1.0)  # be polite to arXiv API
    return all_cites


def render_report(cites: list[Citation], out: Path) -> None:
    lines = [f"# Citation Validation Report — {time.strftime('%Y-%m-%d')}\n"]
    verified_n = sum(1 for c in cites if c.verified)
    lines.append(f"**Total: {len(cites)} citations · Verified: {verified_n} ({100*verified_n//max(len(cites),1)}%)**\n")

    by_file: dict = {}
    for c in cites:
        by_file.setdefault(c.source_file, []).append(c)

    for fp, fcites in sorted(by_file.items()):
        lines.append(f"\n## {fp}\n")
        for c in fcites:
            tag = "✅" if c.verified else "❌"
            lines.append(f"### {tag} Line {c.line}")
            lines.append(f"- **Raw:** {c.raw_text[:120]}...")
            if c.arxiv_id:
                lines.append(f"- **arXiv ID:** [{c.arxiv_id}](https://arxiv.org/abs/{c.arxiv_id})")
            if c.arxiv_title:
                lines.append(f"- **arXiv Title:** {c.arxiv_title}")
            if c.arxiv_authors:
                lines.append(f"- **arXiv Authors:** {', '.join(c.arxiv_authors[:5])}"
                            + (f" (+{len(c.arxiv_authors)-5} more)" if len(c.arxiv_authors) > 5 else ""))
            if c.arxiv_year:
                lines.append(f"- **arXiv Year:** {c.arxiv_year}")
            if c.issues:
                lines.append(f"- **Issues:**")
                for iss in c.issues:
                    lines.append(f"  - {iss}")
            lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nReport written to {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None,
                    help="Specific files to audit (default: all papers/*.md)")
    ap.add_argument("--out", default="papers/CITATIONS_AUDIT_AUTOMATED.md")
    args = ap.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = list(Path("papers").glob("*-v*.md")) + list(Path("papers").glob("*preregistration*.md"))

    print(f"Auditing {len(paths)} file(s)...")
    cites = audit_files(paths)
    render_report(cites, Path(args.out))

    n_fail = sum(1 for c in cites if not c.verified)

    # Per Gate-13 amendment A1 — record decision to append-only log.
    try:
        from gate_log import record as _gate_record
        for p in paths:
            p_cites = [c for c in cites if c.source_file == str(p)]
            p_fail = sum(1 for c in p_cites if not c.verified)
            _gate_record(
                paper=p.name,
                gate="G2",
                claim=f"all {len(p_cites)} citations arXiv-primary-source verified",
                decision="PASS" if p_fail == 0 else "FAIL",
                reason=f"{len(p_cites)-p_fail}/{len(p_cites)} verified",
                validators_run=["validate_citations.py"],
            )
    except Exception as _e:
        print(f"[gate-log] could not write: {_e}")

    if n_fail > 0:
        print(f"\n{n_fail} unverified citations — review {args.out} before submission.")
        sys.exit(1)
    print(f"\n✓ All {len(cites)} citations verified.")


if __name__ == "__main__":
    main()
