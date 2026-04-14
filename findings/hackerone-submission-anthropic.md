# HackerOne submission — Anthropic Responsible Disclosure

**Submission URL (from Anthropic's `security.txt`):**
https://hackerone.com/297a385f-b3bd-4ecd-9466-7d9ad55371ce/embedded_submissions/new

**Policy:** https://www.anthropic.com/responsible-disclosure-policy (safe harbor, ack in 3 business days)

---

## Fields to paste into the HackerOne form

### Report title

> Claude Code's default `Co-Authored-By` email (`noreply@anthropic.com`) is squatted on GitHub, enabling large-scale attribution theft

### Severity rating

**Medium (Attribution / Integrity)** — no code execution, no data exfil, but systematic and at scale.

- CVSS-like: Low technical severity, High reach (every public repo using Claude Code's default commit template), Medium integrity impact.
- Classification (CWE): CWE-290 Authentication Bypass by Spoofing (via email) / CWE-345 Insufficient Verification of Data Authenticity.

### Vulnerability information (main body)

```markdown
## Summary

Claude Code's default HEREDOC commit template emits:

    Co-Authored-By: Claude <noreply@anthropic.com>

A GitHub user (`virendrakumar456`) has registered `noreply@anthropic.com` as
one of the emails on their GitHub account. GitHub's contributor attribution
matches commit-body emails to registered user emails. As a result, every
public GitHub repository worldwide using Claude Code's default commit
template attributes the AI co-authorship to that third party on GitHub's
contributor graphs and API, and on downstream systems that pull from those
(Zenodo, citation graphs, etc.).

This is not a code-access or secrets-disclosure vulnerability — the squatter
cannot push to or read private repos. It is an integrity / attribution
vulnerability at scale.

## Proof of concept

I discovered this on our own research repository `tejasphatak/webmind-research`
after noticing a Zenodo-minted DOI listed `virendrakumar456` as creator.

Reproduction:

1. Create any repo; commit with Claude Code default template (body contains
   `Co-Authored-By: Claude <noreply@anthropic.com>`).
2. Push to GitHub.
3. Query the contributors API:
   `GET /repos/{owner}/{repo}/contributors`
4. Observe `virendrakumar456` listed as a contributor with one entry per
   Claude-Code commit, despite no involvement with the repository.

My repo had exactly 14 such commits. GitHub reported
`virendrakumar456` as having **exactly 14 contributions**. Numerical match
confirms the mechanism.

Public writeup (with full evidence + the fix I applied to my repo):
https://github.com/tejasphatak/webmind-research/blob/master/findings/2026-04-14-claude-code-coauthor-email-squat.md

## Impact

- **Attribution theft at scale.** Potentially every public GitHub repository
  using Claude Code's default commit template shows `virendrakumar456` as a
  contributor. This is visible on every repo's contributor page and in the
  GitHub API, for an indefinite period (historic commits cannot be easily
  rewritten on repos with forks/clones).
- **Downstream scientific-integrity harm.** Services that derive authorship
  from GitHub contributors (Zenodo's auto-DOI, citation graphs, some bibliographic
  tools) inherit the mis-attribution. My pre-registered scientific paper on
  Zenodo had its creator field auto-populated with `virendrakumar456`,
  which I had to manually correct.
- **Erodes trust** in AI-contributor attribution generally at a moment when
  that trust is being built.
- **Potential profile-padding / impersonation.** `virendrakumar456`'s own
  profile may show inflated contribution counts across repos they never
  touched, which could be used for bogus credentialing, fraudulent CVs, etc.

## Mitigation I've already applied (in my own repo)

1. Rewrote commit history via `git filter-branch` to use an email we own
   (`ai-coauthor+claude@webmind.sh`).
2. Force-pushed (no forks existed; safe).
3. Installed a global `commit-msg` hook that auto-rewrites the default email
   on any future commit.

Full technical writeup:
https://github.com/tejasphatak/webmind-research/blob/master/findings/2026-04-14-claude-code-coauthor-email-squat.md

## Recommended upstream fixes (in order of preference)

1. **Best:** Change Claude Code's default `Co-Authored-By` email to one that
   Anthropic has registered on an Anthropic-owned GitHub account (or a
   dedicated bot account whose email cannot be registered elsewhere — e.g.,
   the `claude-code[bot]@users.noreply.github.com` format if Anthropic
   registers a GitHub App). This removes the squattable string entirely.
2. **Or:** On first-use, prompt the user to choose a Co-Authored-By email
   (default to their own GitHub `noreply` address if known).
3. **Or (minimal):** Add a security advisory to Claude Code's README /
   first-run output warning about the squat risk, with remediation
   instructions.

## Timeline

- 2026-04-14: Discovered this vulnerability; fixed my own repo; published
  public writeup; filing this HackerOne report.
- Willing to keep the writeup low-profile (no blog/social amplification) for
  90 days while Anthropic implements a fix, at your request.

## Thanks

For the record: this vulnerability was discovered and this disclosure was
drafted by a Claude Opus 4.6 instance running on a private GCP VM, under
human direction (Tejas Phatak / Webmind). The irony that your own model
helped find a flaw in your tooling feels like a small demonstration of the
value of AI-assisted security research done responsibly. No bounty
requested; the goal is just for Claude Code to be more robust.

Happy to provide more details or coordinate on timing.

Tejas Phatak
Webmind
https://github.com/tejasphatak/webmind-research
```

### Attachments (if HackerOne allows)

- `findings/2026-04-14-claude-code-coauthor-email-squat.md` (full finding writeup)
- Screenshot of GitHub contributors API response showing `virendrakumar456` with 14 contributions (optional; the URL above suffices)

---

## Submission checklist for Tejas

- [ ] Open: https://hackerone.com/297a385f-b3bd-4ecd-9466-7d9ad55371ce/embedded_submissions/new
- [ ] If not signed into HackerOne: sign up (free; use `tejasphatak+h1@gmail.com` or `synapse+hackerone@webmind.sh`)
- [ ] Paste "Report title" above into the title field
- [ ] Paste the "Vulnerability information" body into the main textarea
- [ ] Set severity: **Medium** (or whatever HackerOne form requires — it may have a CVSS slider; use approximately **CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N** → Medium)
- [ ] Submit.
- [ ] Anthropic will ack within 3 business days per their policy.

## After submission

- Drop the HackerOne report ID in here (`~/.claude/secrets/`?) so I can track status + follow up at 14 / 90 day marks.
- Don't amplify publicly beyond the already-committed GitHub writeup until Anthropic responds.
