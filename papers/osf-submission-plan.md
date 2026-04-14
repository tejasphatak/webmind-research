# OSF Pre-registration Submission Plan

## What's in this plan

A step-by-step to get SFCA on OSF with a real DOI + pre-reg badge. Two paths — one Tejas action, one I execute.

## Tejas one-time setup (≈ 5 min)

1. Go to https://osf.io/register
2. Email: `synapse+osf@webmind.sh` (routes to your Gmail)
3. Password: generate a strong one, save in `~/.claude/secrets/osf.json`:
   ```json
   { "email": "synapse+osf@webmind.sh", "password": "...", "token": "..." }
   ```
4. Click the email verification link (goes to your Gmail)
5. Go to https://osf.io/settings/tokens → create a Personal Access Token with scopes:
   - `osf.users.email_read`
   - `osf.full_write`
6. Paste the token into the JSON above

## What I execute (from the VM, via API) — one command

```bash
python3 ~/webmind-research/papers/osf_submit.py
```

This script (I'll write it next) will:
1. Read credentials from `~/.claude/secrets/osf.json`
2. Create a new OSF Project titled "SFCA — Shapley Faculty Credit Assignment for Multi-Perspective LLM Agent Cognition"
3. Upload `papers/sfca-preregistration-v1.md` as the main document
4. Upload the SHA256 manifest from `MANIFEST.md`
5. Link the public GitHub repo
6. Submit as a pre-registration with OSF's "OSF Standard Pre-registration" template
7. Populate the structured fields:
   - **Hypotheses**: H1–H4 from the paper
   - **Dependent variables**: ACTIVE rate, convergence velocity, Advisor-audit score, safety-floor violations
   - **Study type**: Observational, A/B, live production system
   - **Sample size**: ≥ 4 weeks paired (2 SFCA + 2 EMA)
   - **Analysis plan**: paired t-test, Holm–Bonferroni correction
   - **Stopping rule**: no early stopping; ≥ 4 complete weeks
8. Return the resulting DOI + permanent OSF URL

## What OSF gives us

- **DOI** (citable: `10.17605/OSF.IO/XXXXX`)
- **Immutable registration** — post-registration edits don't retroactively change the DOI's associated content
- **"Registered" badge** — peer reviewers recognize this as a real pre-registration
- **Public landing page** with our hypotheses, integrity hash, and GitHub link
- **Timestamp**: OSF's server-side timestamp = additional trust layer beyond git

## Backup plan if OSF doesn't work

- **Zenodo** (via GitHub Release) — also free, also DOIs, less structured but adequate
- **AsPredicted** (osf.io replacement for behavioral science, simpler form)
- **arXiv preprint** (after results are in) — standard for ML research

## Status

- [x] Paper drafted + committed + SHA256 hashed
- [x] GitHub Release v0.1.0-prereg created
- [ ] Zenodo hook enabled (Tejas 3-click)
- [ ] OSF account + token (Tejas 5-min setup)
- [ ] osf_submit.py written (I do after Tejas gives token)
- [ ] DOI minted (automatic after above)
