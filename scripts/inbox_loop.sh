#!/bin/bash
# Agent inbox polling loop. Agent calls this after completing its main task.
# Usage: inbox_loop.sh <agent_name> <timeout_seconds> <max_iterations>
#   agent_name: writer, reviewer, etc.
#   timeout: seconds to wait between checks (default 60)
#   max_iterations: max checks (default 5)

AGENT="${1:?agent name required}"
TIMEOUT="${2:-60}"
MAX_ITER="${3:-5}"
INBOX="$HOME/.claude/teams/book-factory/inboxes/${AGENT}.json"

echo "[inbox_loop] Agent=$AGENT timeout=${TIMEOUT}s max_iter=$MAX_ITER"

for i in $(seq 1 $MAX_ITER); do
  if [ -f "$INBOX" ]; then
    echo "[inbox_loop] Iteration $i/$MAX_ITER: inbox found!"
    cat "$INBOX"
    # Delete after reading so we don't reprocess
    rm -f "$INBOX"
    exit 0
  fi
  echo "[inbox_loop] Iteration $i/$MAX_ITER: no message, sleeping ${TIMEOUT}s..."
  sleep "$TIMEOUT"
done

echo "[inbox_loop] Timeout after $((TIMEOUT * MAX_ITER))s — no message received"
exit 1
