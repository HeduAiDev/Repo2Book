#!/usr/bin/env python3
"""
Agent heartbeat + message ack monitor.

Heartbeat: each agent touches /tmp/book-factory/heartbeat/{role}.json every 60s.
           File contains {"agent": "writer-2", "status": "writing|waiting|done", "time": "..."}

Message ACK: when A sends msg to B, A writes inbox file with msg_id.
             B acknowledges by writing /tmp/book-factory/ack/{msg_id}.json
             Monitor flags messages with no ack after timeout.

Usage:
  python3 scripts/monitor.py --check    Report status of all agents
  python3 scripts/monitor.py --watch    Continuous watch (every 60s)
  python3 scripts/monitor.py --ack {msg_id} {agent}  Write ack (for agents)
  python3 scripts/monitor.py --heartbeat {role} {status}  Write heartbeat
"""

import json, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

HEARTBEAT_DIR = Path("/tmp/book-factory/heartbeat")
ACK_DIR = Path("/tmp/book-factory/ack")
INBOX_DIR = Path.home() / ".claude" / "teams" / "book-factory" / "inboxes"

TIMEOUT_HEARTBEAT = 120  # Agent dead if no heartbeat for 2 min
TIMEOUT_ACK = 300        # Message stale if no ack for 5 min


def heartbeat(role: str, status: str):
    """Write heartbeat file."""
    HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    hb = {
        "agent": role,
        "status": status,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    with open(HEARTBEAT_DIR / f"{role}.json", "w") as f:
        json.dump(hb, f)
    print(f"[heartbeat] {role}: {status}")


def ack(msg_id: str, agent: str):
    """Acknowledge a received message."""
    ACK_DIR.mkdir(parents=True, exist_ok=True)
    ack_data = {
        "msg_id": msg_id,
        "agent": agent,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    with open(ACK_DIR / f"{msg_id}.json", "w") as f:
        json.dump(ack_data, f)
    print(f"[ack] {agent} acknowledged {msg_id}")


def send_message(to: str, content: dict):
    """Send a message with ID to an agent's inbox."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    msg_id = f"msg-{int(time.time() * 1000)}"
    msg = {
        "msg_id": msg_id,
        "from": content.get("from", "team-lead"),
        "to": to,
        "type": content.get("type", "message"),
        "content": content.get("content", ""),
        "sent": datetime.now(timezone.utc).isoformat(),
        "ack_required": True,
    }
    inbox_file = INBOX_DIR / f"{to}.json"
    # Read existing, append
    existing = []
    if inbox_file.exists():
        try:
            with open(inbox_file) as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(msg)
    with open(inbox_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[send] {msg_id} → {to}")
    return msg_id


def check():
    """Report status of all agents: heartbeats, pending acks, dead agents."""
    now = datetime.now(timezone.utc)
    issues = []

    print(f"\n{'='*50}")
    print(f"  Team Monitor — {now.strftime('%H:%M:%S')}")
    print(f"{'='*50}")

    # Check heartbeats
    print("\n[Heartbeats]")
    HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    hb_files = list(HEARTBEAT_DIR.glob("*.json"))
    if not hb_files:
        print("  ⚠ No heartbeat files — agents may not be running")
        issues.append("no_heartbeats")
    else:
        for hf in sorted(hb_files):
            with open(hf) as f:
                hb = json.load(f)
            t = datetime.fromisoformat(hb["time"])
            age = (now - t).total_seconds()
            status = "✓" if age < TIMEOUT_HEARTBEAT else "✗ TIMEOUT"
            if age >= TIMEOUT_HEARTBEAT:
                issues.append(f"agent_timeout:{hf.stem}")
            print(f"  [{status}] {hf.stem:20s} {hb['status']:15s} ({age:.0f}s ago)")

    # Check inbox messages
    print("\n[Messages]")
    ACK_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    found_msgs = False
    for inbox_file in sorted(INBOX_DIR.glob("*.json")):
        try:
            with open(inbox_file) as f:
                msgs = json.load(f)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                msg_id = m.get("msg_id", "?")
                if msg_id == "?":
                    continue
                found_msgs = True
                sent = m.get("sent", "")
                ack_file = ACK_DIR / f"{msg_id}.json"
                if ack_file.exists():
                    print(f"  ✓ {msg_id}: {m.get('type','?')} → {m.get('to','?')} [acked]")
                else:
                    if sent:
                        t = datetime.fromisoformat(sent)
                        age = (now - t).total_seconds()
                    else:
                        age = 0
                    if age > TIMEOUT_ACK:
                        print(f"  ⚠ {msg_id}: {m.get('type','?')} → {m.get('to','?')} STALE ({age:.0f}s no ack)")
                        issues.append(f"stale_msg:{msg_id}")
                    else:
                        print(f"  ○ {msg_id}: {m.get('type','?')} → {m.get('to','?')} (pending, {age:.0f}s)")
        except (json.JSONDecodeError, OSError):
            print(f"  ? {inbox_file.stem}: unreadable")
    if not found_msgs:
        print("  (no tracked messages)")

    # Summary
    print(f"\n{'='*50}")
    if issues:
        print(f"  ⚠ {len(issues)} issue(s):")
        for i in issues:
            print(f"    - {i}")
    else:
        print(f"  ✓ All agents healthy, all messages acknowledged")
    print(f"{'='*50}\n")

    return issues


def watch(interval: int = 60):
    """Continuous monitoring."""
    import signal
    running = True
    def handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handler)

    while running:
        issues = check()
        if issues:
            print(f"[monitor] ⚠ Issues detected: {issues}")
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "--heartbeat":
        role = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        status = sys.argv[3] if len(sys.argv) > 3 else "active"
        heartbeat(role, status)
    elif cmd == "--ack":
        msg_id = sys.argv[2] if len(sys.argv) > 2 else ""
        agent = sys.argv[3] if len(sys.argv) > 3 else "unknown"
        ack(msg_id, agent)
    elif cmd == "--send":
        to = sys.argv[2] if len(sys.argv) > 2 else ""
        content = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        send_message(to, content)
    elif cmd == "--check":
        check()
    elif cmd == "--watch":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        watch(interval)
    else:
        print(__doc__)
