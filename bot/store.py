"""
store.py — thread-safe in-memory context store.

Holds everything the judge pushes via POST /v1/context, keyed by
(scope, context_id), with version-based idempotency as required by
challenge-testing-brief.md §2.1.

Also holds in-flight conversation state for /v1/tick and /v1/reply.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Optional


class ContextStore:
    def __init__(self):
        self._lock = threading.Lock()
        # (scope, context_id) -> {"version": int, "payload": dict}
        self._contexts: dict[tuple[str, str], dict[str, Any]] = {}
        # conversation_id -> list of turn dicts (both directions)
        self._conversations: dict[str, list[dict]] = {}
        # conversation_id -> {"merchant_id", "customer_id", "trigger_id"}
        self._conversation_meta: dict[str, dict] = {}
        # suppression_key -> True, once we've sent for it (dedup across ticks)
        self._sent_suppression_keys: set[str] = set()
        # (merchant_id, exact_incoming_text) -> occurrence count, tracked
        # ACROSS conversations — a canned auto-reply can arrive on a fresh
        # conversation_id each time, so per-conversation history alone can't
        # detect the repeat.
        self._incoming_text_counts: dict[tuple[str, str], int] = {}
        self._start_time = datetime.now(timezone.utc)

    # ---------------- context ----------------

    def push(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        """Returns the ack/conflict dict to send back to the judge."""
        with self._lock:
            key = (scope, context_id)
            current = self._contexts.get(key)
            if current and current["version"] >= version:
                return {
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current["version"],
                }
            self._contexts[key] = {"version": version, "payload": payload}
            return {
                "accepted": True,
                "ack_id": f"ack_{context_id}_v{version}",
                "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._contexts.get((scope, context_id))
            return entry["payload"] if entry else None

    def counts(self) -> dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _key) in self._contexts.keys():
                counts[scope] = counts.get(scope, 0) + 1
        return counts

    def uptime_seconds(self) -> int:
        return int((datetime.now(timezone.utc) - self._start_time).total_seconds())

    def wipe(self):
        """Called on POST /v1/teardown."""
        with self._lock:
            self._contexts.clear()
            self._conversations.clear()
            self._conversation_meta.clear()
            self._sent_suppression_keys.clear()
            self._incoming_text_counts.clear()

    # ---------------- conversations ----------------

    def get_conversation(self, conversation_id: str) -> list[dict]:
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def append_turn(self, conversation_id: str, turn: dict):
        with self._lock:
            self._conversations.setdefault(conversation_id, []).append(turn)

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._lock:
            return conversation_id in self._conversations

    def start_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        trigger_id: Optional[str],
    ):
        with self._lock:
            self._conversations.setdefault(conversation_id, [])
            self._conversation_meta[conversation_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id": trigger_id,
            }

    def get_conversation_meta(self, conversation_id: str) -> Optional[dict]:
        with self._lock:
            meta = self._conversation_meta.get(conversation_id)
            return dict(meta) if meta else None

    # ---------------- suppression dedup ----------------

    def already_sent(self, suppression_key: str) -> bool:
        if not suppression_key:
            return False
        with self._lock:
            return suppression_key in self._sent_suppression_keys

    def mark_sent(self, suppression_key: str):
        if not suppression_key:
            return
        with self._lock:
            self._sent_suppression_keys.add(suppression_key)

    # ---------------- cross-conversation repeat tracking ----------------

    def record_incoming_and_count_prior(self, merchant_id: str, text: str) -> int:
        """Record this incoming text for this merchant and return how many
        times it was already seen BEFORE this call (0 = first time), counted
        across ALL conversations with this merchant — not just the current
        conversation_id — since a canned auto-reply can land on a fresh
        conversation_id each time."""
        if not merchant_id or not text:
            return 0
        key = (merchant_id, text)
        with self._lock:
            prior = self._incoming_text_counts.get(key, 0)
            self._incoming_text_counts[key] = prior + 1
            return prior


# Single process-wide store instance, imported by bot.py
store = ContextStore()