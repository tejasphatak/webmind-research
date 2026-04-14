# Security finding: Claude Code default `Co-Authored-By` email is being squatted on GitHub

**Reported by:** Webmind Research (AI-generated disclosure, human-accountable)
**Date discovered:** 2026-04-14
**Severity:** medium (attribution integrity, not code access)
**Status:** disclosed publicly; private disclosure to Anthropic pending

## Summary

Claude Code's default HEREDOC commit template emits:
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

A GitHub user (`virendrakumar456`) has registered `noreply@anthropic.com` as an email on their GitHub account. GitHub's contributor attribution logic then maps every worldwide Claude Code commit using this default template to that user's profile.

## Evidence

In the `tejasphatak/webmind-research` repo (our own), before fix:
- 14 total commits, all with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`
- GitHub contributors API reported: `virendrakumar456` with **exactly 14 contributions**
- Zenodo DOI auto-minted from a release listed `virendrakumar456` as creator (pulled from GitHub contributors data)

Numerical match confirms causal mechanism.

## Fix applied in our repo

- `git filter-branch` rewrote all commits' Co-Authored-By to an email we own
- Force-pushed to GitHub (no forks existed; safe)
- Installed a global `commit-msg` hook that auto-rewrites the default email on future commits

## Recommended fix by Anthropic (upstream)

- Change the Claude Code default Co-Authored-By email to one Anthropic owns and cannot be registered by third parties (e.g., a `@noreply.github.com` variant tied to a specific bot account they control)
- OR: warn users in Claude Code's first-run setup that they should set a custom author email
- OR: document this squat risk prominently

## Broader impact

- Any public GitHub repo with Claude Code-generated commits using the default template appears to have `virendrakumar456` as a contributor
- Academic/citation systems that pull from GitHub contributor lists (Zenodo; potentially others) will mis-attribute
- This is attribution laundering at potentially very large scale
