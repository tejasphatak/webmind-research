# WARNING: This system executes code

The SAQT knowledge base contains `<tool>` tagged entries that execute Python code in a subprocess when matched. 

**DO NOT expose serve.py to the public internet without proper sandboxing (Docker, firejail, or WebAssembly).**

The current sandbox is minimal: subprocess with restricted PATH and /tmp CWD. A determined attacker could escape it.

For safe public deployment, wrap the executor in a Docker container with:
- No network access
- Read-only filesystem
- Memory limit
- CPU timeout
- No privilege escalation
