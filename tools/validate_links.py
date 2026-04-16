"""
Link validator — HEAD-requests every URL in every paper and reports 4xx/5xx.

Per SUBMISSION_GATING.md: broken links block submission.

Usage:
    python tools/validate_links.py                         # all papers
    python tools/validate_links.py --files papers/foo.md

Author: Claude Opus 4.6
Date: 2026-04-16
"""
from __future__ import annotations
import re, sys, time, urllib.request, urllib.error
from pathlib import Path
from dataclasses import dataclass, field

URL_PATTERN = re.compile(r'https?://[^\s\)\]\}]+')
ARXIV_ID_PATTERN = re.compile(r'arXiv:(\d{4}\.\d{4,5})')


@dataclass
class LinkCheck:
    source_file: str
    line: int
    url: str
    status: int = 0
    error: str = ""
    ok: bool = False


def extract_links(path: Path) -> list[LinkCheck]:
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        for m in URL_PATTERN.finditer(line):
            url = m.group(0).rstrip(".,;:)*]'\"")
            out.append(LinkCheck(source_file=str(path), line=i, url=url))
        # Also turn bare "arXiv:YYMM.NNNNN" into a link check
        for m in ARXIV_ID_PATTERN.finditer(line):
            url = f"https://arxiv.org/abs/{m.group(1)}"
            out.append(LinkCheck(source_file=str(path), line=i, url=url))
    # Dedupe by URL per file
    seen = set()
    deduped = []
    for lc in out:
        key = (lc.source_file, lc.url)
        if key not in seen:
            seen.add(key)
            deduped.append(lc)
    return deduped


def check_one(lc: LinkCheck) -> LinkCheck:
    try:
        req = urllib.request.Request(lc.url, method="HEAD",
                                     headers={"User-Agent": "link-validator/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            lc.status = r.status
            lc.ok = 200 <= r.status < 400
    except urllib.error.HTTPError as e:
        lc.status = e.code
        lc.error = str(e.reason)
    except urllib.error.URLError as e:
        lc.error = str(e.reason)
    except Exception as e:
        lc.error = str(e)
    # Some servers (arxiv.org, github) block HEAD; retry with GET
    if not lc.ok and (lc.status in (403, 405, 0) or "arxiv" in lc.url or "github" in lc.url):
        try:
            req = urllib.request.Request(lc.url, method="GET",
                                         headers={"User-Agent": "link-validator/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                lc.status = r.status
                lc.ok = 200 <= r.status < 400
                lc.error = ""
        except Exception as e:
            lc.error = f"GET retry failed: {e}"
    return lc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None)
    args = ap.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = list(Path("papers").glob("*-v*.md"))

    all_checks = []
    for p in paths:
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        links = extract_links(p)
        print(f"  {p.name}: {len(links)} unique URLs")
        all_checks.extend(links)

    failures = []
    for lc in all_checks:
        lc = check_one(lc)
        tag = "✅" if lc.ok else "❌"
        detail = f"HTTP {lc.status}" + (f" {lc.error}" if lc.error else "")
        print(f"  {tag} {detail:30s} {lc.url[:80]}  ({lc.source_file}:{lc.line})")
        if not lc.ok:
            failures.append(lc)
        time.sleep(0.3)

    print(f"\n{len(all_checks)} URLs checked, {len(failures)} failures")

    try:
        from gate_log import record as _gate_record
        by_file: dict[str, list] = {}
        for lc in all_checks:
            by_file.setdefault(lc.source_file, []).append(lc)
        for fname, lcs in by_file.items():
            fails = [x for x in lcs if not x.ok]
            _gate_record(
                paper=Path(fname).name,
                gate="G12",
                claim=f"all {len(lcs)} URLs resolve (HTTP 2xx/3xx)",
                decision="PASS" if not fails else "FAIL",
                reason=f"{len(fails)} broken" if fails else "all ok",
                validators_run=["validate_links.py"],
            )
    except Exception as _e:
        print(f"[gate-log] could not write: {_e}")

    if failures:
        print("\nBLOCKERS before submission:")
        for f in failures:
            print(f"  {f.source_file}:{f.line}  {f.url}  →  {f.status} {f.error}")
        sys.exit(1)
    print("✓ All links valid.")


if __name__ == "__main__":
    main()
