# Disclosure email draft — Anthropic Security

**To:** security@anthropic.com
**From:** synapse@webmind.sh *(Webmind Research; human-accountable author: Tejas Phatak)*
**Subject:** Security disclosure — Claude Code default `Co-Authored-By` email is being squatted on GitHub, enabling attribution theft at scale
**Priority:** Medium (integrity, not code access)
**Disclosure policy:** 90-day public-disclosure window from date of this email unless a shorter/longer window is negotiated

---

Hi Anthropic Security,

I'm writing to report a non-critical but systemic issue in Claude Code's default commit template that enables third-party attribution theft via GitHub's email-based contributor attribution.

## Summary

Claude Code's default HEREDOC commit template (as documented in [claude-code/src/constants/prompts.ts](https://github.com/yasasbanukaofficial/claude-code/) and embedded in every default Claude Code commit message) includes:

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

A GitHub user (`virendrakumar456`) has registered `noreply@anthropic.com` as an email on their GitHub account. GitHub's contributor attribution matches commit-body emails to registered user emails. As a result, **every public repository worldwide using Claude Code's default commit template attributes the AI co-authorship to that third party on GitHub's contributor graphs and API.**

## Evidence

I discovered this on our own research repository `tejasphatak/webmind-research` after noticing a Zenodo-minted DOI listed `virendrakumar456` as creator.

- Our repo had 14 commits, all with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.
- GitHub's contributor API (`GET /repos/tejasphatak/webmind-research/contributors`) returned `virendrakumar456` with **exactly 14 contributions**.
- Numerical match confirms the mechanism.
- Zenodo pulled creator metadata from GitHub's contributor list, mis-attributing our pre-registered scientific DOI to that user.

Full technical writeup: https://github.com/tejasphatak/webmind-research/blob/master/findings/2026-04-14-claude-code-coauthor-email-squat.md

## Impact

- **Severity:** medium (integrity + attribution, not code access or secrets)
- **Blast radius:** potentially every public GitHub repository using Claude Code's default commit template
- **Secondary impact:** downstream systems that derive authorship from GitHub (Zenodo, citation graphs, academic/DOI minting services) inherit the mis-attribution
- **Harm:** scientific-integrity harm to Claude Code users publishing research; credit-laundering for the squatter; erodes trust in AI-contributor attribution generally

## Mitigations we've applied for our own repo

1. Rewrote our commit history with `git filter-branch` to use an email we own (`ai-coauthor+claude@webmind.sh`)
2. Force-pushed (no forks existed; no downstream breakage)
3. Installed a global `commit-msg` hook that auto-rewrites the default email on any future commit
4. Will edit the Zenodo record to correct creator metadata

## Recommended upstream fixes (in increasing preference)

1. **Best:** change Claude Code's default to an email Anthropic controls and has registered on a real Anthropic-owned GitHub account (or a dedicated bot account whose email can't be squatted elsewhere — e.g., `claude-code[bot]@users.noreply.github.com` if Anthropic registers a GitHub App with that name)
2. Offer a per-install configuration prompt on first use: "Claude Code will add `Co-Authored-By: Claude <YOUR_EMAIL>` by default; would you like to customize?"
3. At minimum: add a security advisory to Claude Code's README / first-run flow warning about the squat risk, and document a recommended custom email pattern

## Related upstream consideration

GitHub itself also has some responsibility — email-based attribution without verification of identity is an old issue (e.g., Git spoofing). But the default string in a widely-used tool is where the leverage lives for fast fix.

## Disclosure plan

- **Today (2026-04-14):** this email sent. Finding documented in a public repo but not amplified.
- **+14 days:** followup if no response.
- **+90 days:** public disclosure via webmind-research repo + maybe a short blog post, unless Anthropic requests otherwise.

I'm happy to coordinate timing if you'd prefer a different window, and to privately share any additional details.

## About this disclosure

This disclosure was drafted by an AI instance (Claude Opus 4.6, your model) under human direction, as part of the Webmind research project at https://github.com/tejasphatak/webmind-research. The irony that your own model helped identify a flaw in your tooling is not lost on us — and I think it's a small demonstration of the value of AI-assisted security research done responsibly.

No bounty requested. The goal is for Claude Code to be more robust for its users, including the academic and research communities that increasingly depend on it.

Thanks for your time and for building a product that genuinely helps.

Best,
Tejas Phatak
Webmind
synapse@webmind.sh
https://github.com/tejasphatak

---

*Draft notes for Tejas before sending:*
- Confirm `security@anthropic.com` is the correct address (check `https://www.anthropic.com/.well-known/security.txt` or their published responsible-disclosure page)
- Optionally cc: `product@anthropic.com` or the Claude Code product team if they have a separate address
- Send from `synapse@webmind.sh` (now live via Cloudflare Email Routing — receives; for sending you'd reply from Gmail's "Send as" or wait until Resend is set up; for now, sending from your personal Gmail with a reply-to of synapse@webmind.sh is fine)
- This draft is ready to send. No edits strictly required unless you want to soften or sharpen the tone.
