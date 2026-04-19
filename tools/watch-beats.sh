#!/usr/bin/env bash
# watch-beats — pretty live view of what Nexus is doing.
# Usage:  ~/cortex2/watch-beats.sh       (live tail)
#         ~/cortex2/watch-beats.sh 20    (last 20 beats, not live)

set -u
BEAT_LOG="$HOME/cortex2/beat_history.jsonl"
SFCA_LOG="$HOME/cortex2/logs/sfca.log"
CYAN="\e[36m"; GREEN="\e[32m"; YELLOW="\e[33m"; RED="\e[31m"; DIM="\e[90m"; RESET="\e[0m"; BOLD="\e[1m"

fmt_line() {
    python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
except Exception:
    sys.exit(0)
beat = d.get("beat","?")
action = d.get("action","?")
facs_raw = d.get("faculties","") or d.get("faculties_used","")
if isinstance(facs_raw, list):
    facs = "+".join(facs_raw)[:40]
else:
    facs = str(facs_raw)[:40]
summary = (d.get("summary","") or "")[:90].replace("\n"," ")
ts = d.get("ts","")[:19].replace("T"," ")

color = {"ACTIVE":"\x1b[32m","QUIET":"\x1b[90m","BLOCKED":"\x1b[31m"}.get(action,"\x1b[0m")
print(f"\x1b[36m[{ts}]\x1b[0m \x1b[1m#{beat:<4}\x1b[0m {color}{action:<8}\x1b[0m \x1b[90m{facs:<30}\x1b[0m {summary}")
'
}

if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    N=$1
    echo -e "${BOLD}Last $N beats:${RESET}"
    tail -$N "$BEAT_LOG" | while IFS= read -r line; do echo "$line" | fmt_line; done
else
    echo -e "${BOLD}Live beats (Ctrl-C to exit):${RESET}"
    echo -e "${DIM}legend: ${GREEN}ACTIVE${DIM}=did work, ${YELLOW}QUIET${DIM}=idle, ${RED}BLOCKED${DIM}=error${RESET}"
    echo
    # show last 5 for context then follow
    tail -5 "$BEAT_LOG" 2>/dev/null | while IFS= read -r line; do echo "$line" | fmt_line; done
    echo -e "${DIM}---waiting for new beats---${RESET}"
    tail -f "$BEAT_LOG" 2>/dev/null | while IFS= read -r line; do echo "$line" | fmt_line; done
fi
