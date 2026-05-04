#!/usr/bin/env python3
"""
Organizational decision protocol for repo2book Agent Teams.

Supports three modes:
  vote    — Formal vote with quorum, thresholds, tie-break
  discuss — Structured discussion leading to consensus or escalation
  propose — Propose a topology change for a chapter

Usage:
  python3 scripts/decide.py vote <topic> <agents> <threshold>
  python3 scripts/decide.py discuss <topic> <agents>
  python3 scripts/decide.py propose-topology <chapter_id> <mode> <reason>
  python3 scripts/decide.py tally <topic>           # Count current votes
  python3 scripts/decide.py escalate <issue> <reason> # Escalate to Lead

Protocol:
  VOTE:   Proposal → agents submit votes (APPROVE/REJECT/ABSTAIN) → tally → decision
  DISCUSS: Proposal → round 1 arguments → round 2 counter → synthesis → decision
  ESCALATE: Agent submits issue → Lead reviews → Lead decides or restructures
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

def load_config() -> dict:
    config_file = ROOT / "repo2book.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}

CONFIG = load_config()
DECISION_PROTOCOL = CONFIG.get("pipeline", {}).get("topology", {}).get("decision_protocol", {})
TEAM = CONFIG.get("team", {}).get("name", "book-factory")
DECISIONS_DIR = Path.home() / ".claude" / "teams" / TEAM / "decisions"
INBOX_DIR = Path.home() / ".claude" / "teams" / TEAM / "inboxes"


# ── Voting Protocol ──────────────────────────────────────────────

def create_vote(topic: str, agents: list, threshold: str = "majority") -> dict:
    """Create a new vote proposal."""
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    vote_id = topic.lower().replace(" ", "-")[:40]
    vote = {
        "vote_id": vote_id,
        "topic": topic,
        "threshold": threshold,
        "quorum": max(1, len(agents) // 2 + 1),
        "agents": agents,
        "created": datetime.now(timezone.utc).isoformat(),
        "votes": {},
        "status": "open",
        "result": None,
    }
    vote_file = DECISIONS_DIR / f"vote-{vote_id}.json"
    with open(vote_file, "w") as f:
        json.dump(vote, f, indent=2)

    # Notify agents via inbox
    for agent in agents:
        inbox_file = INBOX_DIR / f"{agent}.json"
        msg = {
            "type": "vote_request",
            "vote_id": vote_id,
            "topic": topic,
            "threshold": threshold,
            "options": ["APPROVE", "REJECT", "ABSTAIN"],
            "instructions": f"To vote, reply: VOTE {vote_id} APPROVE|REJECT|ABSTAIN [rationale]"
        }
        _append_inbox(inbox_file, msg)

    print(f"Vote created: {vote_id}")
    print(f"  Topic: {topic}")
    print(f"  Threshold: {threshold} ({_threshold_desc(threshold)})")
    print(f"  Quorum: {vote['quorum']}/{len(agents)}")
    print(f"  Agents: {', '.join(agents)}")
    return vote


def tally_votes(topic: str) -> Optional[dict]:
    """Count votes and determine result."""
    vote_id = topic.lower().replace(" ", "-")[:40]
    vote_file = DECISIONS_DIR / f"vote-{vote_id}.json"
    if not vote_file.exists():
        print(f"No vote found for: {topic}")
        return None

    with open(vote_file) as f:
        vote = json.load(f)

    votes = vote["votes"]
    total = len(vote["agents"])
    cast = len(votes)
    approve = sum(1 for v in votes.values() if v["vote"] == "APPROVE")
    reject = sum(1 for v in votes.values() if v["vote"] == "REJECT")
    abstain = sum(1 for v in votes.values() if v["vote"] == "ABSTAIN")

    # Check quorum
    if cast < vote["quorum"]:
        vote["status"] = "no_quorum"
        vote["result"] = f"No quorum: {cast}/{vote['quorum']} voted"
        print(f"NO QUORUM: {cast}/{vote['quorum']} agents voted (need {vote['quorum']})")
        _save_vote(vote_file, vote)
        return vote

    # Apply threshold
    deciding_votes = cast - abstain
    if deciding_votes == 0:
        vote["status"] = "no_decision"
        vote["result"] = "All abstained — no decision"
        print("ALL ABSTAINED — escalate to Lead")
        _save_vote(vote_file, vote)
        return vote

    required = _threshold_num(deciding_votes, vote["threshold"])
    if approve >= required:
        vote["status"] = "approved"
        vote["result"] = f"APPROVED: {approve}/{deciding_votes} ({vote['threshold']})"
    elif reject >= required:
        vote["status"] = "rejected"
        vote["result"] = f"REJECTED: {reject}/{deciding_votes} ({vote['threshold']})"
    else:
        vote["status"] = "split"
        vote["result"] = f"SPLIT: {approve} approve, {reject} reject — Lead decides"
        print(f"SPLIT VOTE: {approve} approve, {reject} reject → LEAD DECIDES")

    print(f"Result: {vote['result']}")
    _save_vote(vote_file, vote)
    return vote


# ── Discussion Protocol ──────────────────────────────────────────

def create_discussion(topic: str, agents: list, proposal: str) -> dict:
    """Start a structured discussion."""
    discussion_id = topic.lower().replace(" ", "-")[:40]
    disc = {
        "discussion_id": discussion_id,
        "topic": topic,
        "proposal": proposal,
        "agents": agents,
        "rounds_max": DECISION_PROTOCOL.get("discussion", {}).get("rounds_max", 3),
        "current_round": 0,
        "rounds": [],
        "created": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "conclusion": None,
    }
    disc_file = DECISIONS_DIR / f"discussion-{discussion_id}.json"
    with open(disc_file, "w") as f:
        json.dump(disc, f, indent=2)

    for agent in agents:
        inbox_file = INBOX_DIR / f"{agent}.json"
        msg = {
            "type": "discussion_request",
            "discussion_id": discussion_id,
            "topic": topic,
            "proposal": proposal,
            "round": 1,
            "instructions": f"Reply with your analysis: PROs, CONs, RISKs, and vote PREFER|OPPOSE|NEUTRAL"
        }
        _append_inbox(inbox_file, msg)

    print(f"Discussion started: {discussion_id}")
    print(f"  Proposal: {proposal}")
    print(f"  Agents: {', '.join(agents)}")
    return disc


# ── Escalation Protocol ──────────────────────────────────────────

def escalate(issue: str, reason: str, agent: str = "unknown") -> dict:
    """Escalate an issue to the Lead."""
    esc = {
        "type": "escalation",
        "issue": issue,
        "reason": reason,
        "from_agent": agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    # Write to book-editor's inbox for Lead review
    inbox_file = INBOX_DIR / "book-editor.json"
    _append_inbox(inbox_file, esc)

    # Also write to decisions log
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    esc_file = DECISIONS_DIR / f"escalation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(esc_file, "w") as f:
        json.dump(esc, f, indent=2)

    print(f"ESCALATED to Lead: {issue}")
    print(f"  From: {agent}")
    print(f"  Reason: {reason}")
    return esc


# ── Topology Proposal ────────────────────────────────────────────

def propose_topology(chapter_id: str, mode: str, reason: str, proposer: str = "unknown") -> dict:
    """Propose a topology change for a chapter."""
    valid_modes = list(CONFIG.get("pipeline", {}).get("topology", {}).get("modes", {}).keys())
    if mode not in valid_modes:
        print(f"Invalid mode: {mode}. Valid: {valid_modes}")
        sys.exit(1)

    proposal = {
        "type": "topology_proposal",
        "chapter_id": chapter_id,
        "proposed_mode": mode,
        "reason": reason,
        "proposer": proposer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "proposed",
    }
    # Requires supermajority vote from all pipeline agents
    agents = CONFIG.get("pipeline", {}).get("stages", [])
    vote = create_vote(
        f"topology-{chapter_id}-{mode}",
        agents,
        threshold="supermajority"
    )
    proposal["vote_id"] = vote["vote_id"]
    return proposal


# ── Helpers ───────────────────────────────────────────────────────

def _threshold_desc(threshold: str) -> str:
    return {
        "majority": ">50%",
        "supermajority": ">66%",
        "consensus": "100%",
    }.get(threshold, ">50%")


def _threshold_num(total: int, threshold: str) -> int:
    if threshold == "consensus":
        return total
    elif threshold == "supermajority":
        return int(total * 0.66) + 1
    else:  # majority
        return total // 2 + 1


def _append_inbox(inbox_file: Path, msg: dict):
    messages = []
    if inbox_file.exists():
        try:
            with open(inbox_file) as f:
                existing = json.load(f)
                messages = existing if isinstance(existing, list) else [existing]
        except (json.JSONDecodeError, OSError):
            pass
    messages.append(msg)
    inbox_file.parent.mkdir(parents=True, exist_ok=True)
    with open(inbox_file, "w") as f:
        json.dump(messages, f, indent=2)


def _save_vote(vote_file: Path, vote: dict):
    with open(vote_file, "w") as f:
        json.dump(vote, f, indent=2)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "vote":
        # python3 scripts/decide.py vote "Should we use pair mode for Ch14?" "implementer,tester,writer,reviewer" majority
        topic = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        agents = sys.argv[3].split(",") if len(sys.argv) > 3 else []
        threshold = sys.argv[4] if len(sys.argv) > 4 else "majority"
        create_vote(topic, agents, threshold)

    elif cmd == "tally":
        topic = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        tally_votes(topic)

    elif cmd == "discuss":
        topic = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        agents = sys.argv[3].split(",") if len(sys.argv) > 3 else []
        proposal = sys.argv[4] if len(sys.argv) > 4 else "No proposal specified"
        create_discussion(topic, agents, proposal)

    elif cmd == "escalate":
        issue = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        reason = sys.argv[3] if len(sys.argv) > 3 else "No reason given"
        agent = sys.argv[4] if len(sys.argv) > 4 else "unknown"
        escalate(issue, reason, agent)

    elif cmd == "propose-topology":
        chapter_id = sys.argv[2] if len(sys.argv) > 2 else ""
        mode = sys.argv[3] if len(sys.argv) > 3 else "linear"
        reason = sys.argv[4] if len(sys.argv) > 4 else "No reason given"
        proposer = sys.argv[5] if len(sys.argv) > 5 else "book-editor"
        propose_topology(chapter_id, mode, reason, proposer)

    elif cmd == "list-modes":
        modes = CONFIG.get("pipeline", {}).get("topology", {}).get("modes", {})
        for name, info in modes.items():
            default = " (default)" if name == "linear" else ""
            print(f"  {name}{default}: {info['description']}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
