#!/bin/bash
# Agent heartbeat loop — call from agent prompt in background
# Usage: agent_heartbeat.sh <role> <check_interval_seconds> &
#   role: writer, reviewer, researcher, archivist, etc.
#   Writes heartbeat every N seconds. Checks inbox for new messages.

ROLE="${1:-unknown}"
INTERVAL="${2:-60}"

HB_DIR="/tmp/book-factory/heartbeat"
INBOX="$HOME/.claude/teams/book-factory/inboxes/${ROLE}.json"
mkdir -p "$HB_DIR"

count=0
while true; do
  count=$((count + 1))

  # Write heartbeat
  echo "{\"agent\":\"$ROLE\",\"status\":\"active\",\"iter\":$count,\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$HB_DIR/${ROLE}.json"

  # Check inbox and alert if new message
  if [ -f "$INBOX" ]; then
    msg_count=$(python3 -c "
import json
try:
    with open('$INBOX') as f: data = json.load(f)
    msgs = data if isinstance(data, list) else [data]
    print(len([m for m in msgs if isinstance(m, dict) and m.get('ack_required')]))
except: print(0)
" 2>/dev/null || echo 0)
    if [ "$msg_count" -gt 0 ]; then
      echo "[heartbeat] $ROLE: $msg_count unread messages in inbox"
      touch "/tmp/book-factory/ack/${ROLE}-has-mail"
    fi
  fi

  sleep "$INTERVAL"
done
