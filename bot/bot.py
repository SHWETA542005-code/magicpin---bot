"""
bot.py — magicpin Vera Challenge bot.

Step 1 skeleton: app boots, /v1/healthz and /v1/metadata work.
/v1/context, /v1/tick, /v1/reply are stubbed — filled in next steps.
"""

import os
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional

import composer
from store import store

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
MAX_ACTIONS_PER_TICK = 20
TICK_TIME_BUDGET_SECONDS = 25  # stay under the judge's 30s timeout with margin
TICK_MAX_WORKERS = 3  # compose multiple triggers' messages concurrently — a
                       # tick can carry several triggers at once (e.g. 5 per
                       # batch), and composing them one-by-one can blow the
                       # time budget since each is a separate LLM call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

load_dotenv()

app = FastAPI(title="Vera Challenge Bot")

START_TIME = time.time()

# ---------------- metadata: fill these in for your team ----------------
TEAM_NAME = "Shweta"
TEAM_MEMBERS = ["Shweta"]
CONTACT_EMAIL = "shweta.ug23@nsut.ac.in"
VERSION = "0.1.0"


@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": store.uptime_seconds(),
        "contexts_loaded": store.counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "approach": "single-prompt composer over Groq, with per-category prompt templates",
        "contact_email": CONTACT_EMAIL,
        "version": VERSION,
        "submitted_at": "2026-09-26T19:30:00+05:30",  # fill at submission time
    }


class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextPush):
    if body.scope not in VALID_SCOPES:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(VALID_SCOPES)}, got '{body.scope}'",
        }
    result = store.push(body.scope, body.context_id, body.version, body.payload)
    return result


@app.post("/v1/teardown")
async def teardown():
    """Optional — judge may call this at end of test to ask us to wipe state
    (challenge-testing-brief.md §11: bots must not persist context after test ends)."""
    store.wipe()
    return {"status": "wiped"}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    started = time.monotonic()
    actions: list[dict] = []

    # Resolve triggers up front and process the most urgent first — if we run
    # out of time budget, we want the important ones sent, not the first ones
    # in an arbitrary list.
    candidates = []
    for trigger_id in body.available_triggers:
        trigger = store.get("trigger", trigger_id)
        if trigger is None:
            continue
        candidates.append((trigger.get("urgency", 0), trigger_id, trigger))
    candidates.sort(key=lambda t: -t[0])

    # Cheap, no-I/O filtering first (dedup + missing context) so we don't burn
    # concurrency slots and LLM quota on triggers we'd skip anyway.
    work_items = []
    for _urgency, trigger_id, trigger in candidates:
        if len(work_items) >= MAX_ACTIONS_PER_TICK:
            break

        suppression_key = trigger.get("suppression_key", "")
        if store.already_sent(suppression_key):
            continue  # already messaged for this exact trigger occasion

        merchant_id = trigger.get("merchant_id") or (trigger.get("payload") or {}).get("merchant_id")
        if not merchant_id:
            continue

        merchant = store.get("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug")
        category = store.get("category", category_slug) if category_slug else None
        if not category:
            continue

        customer_id = trigger.get("customer_id")
        customer = store.get("customer", customer_id) if customer_id else None

        work_items.append({
            "trigger_id": trigger_id, "trigger": trigger,
            "merchant_id": merchant_id, "merchant": merchant,
            "category": category, "customer_id": customer_id, "customer": customer,
            "suppression_key": suppression_key,
        })

    def _compose_one(item: dict) -> dict:
        return composer.compose_message(item["category"], item["merchant"], item["trigger"], item["customer"])

    if work_items:
        pool = ThreadPoolExecutor(max_workers=min(TICK_MAX_WORKERS, len(work_items)))
        futures = {pool.submit(_compose_one, item): item for item in work_items}
        try:
            remaining = max(TICK_TIME_BUDGET_SECONDS - (time.monotonic() - started), 0.1)
            for future in as_completed(futures, timeout=remaining):
                item = futures[future]
                try:
                    composed = future.result()
                except Exception:
                    logger.exception("compose_message failed for trigger %s", item["trigger_id"])
                    continue

                if not composed.get("body"):
                    continue  # never send a malformed/empty action

                trigger_id = item["trigger_id"]
                merchant_id = item["merchant_id"]
                merchant = item["merchant"]
                trigger = item["trigger"]
                customer_id = item["customer_id"]
                suppression_key = item["suppression_key"]

                conversation_id = f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:8]}"
                store.start_conversation(conversation_id, merchant_id, customer_id, trigger_id)
                store.append_turn(conversation_id, {
                    "from": "vera",
                    "message": composed["body"],
                    "ts": body.now,
                })
                if suppression_key:
                    store.mark_sent(suppression_key)

                actions.append({
                    "conversation_id": conversation_id,
                    "merchant_id": merchant_id,
                    "customer_id": customer_id,
                    "send_as": composed["send_as"],
                    "trigger_id": trigger_id,
                    "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
                    "template_params": [(merchant.get("identity") or {}).get("name", "")],
                    "body": composed["body"],
                    "cta": composed["cta"],
                    "suppression_key": composed["suppression_key"],
                    "rationale": composed["rationale"],
                })

                if len(actions) >= MAX_ACTIONS_PER_TICK:
                    break
        except FutureTimeoutError:
            logger.warning("Tick time budget exceeded; returning %d actions early", len(actions))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


HOSTILE_HINTS = ("stop", "spam", "useless", "harassment", "block", "report", "fuck", "bakwas")
POSITIVE_HINTS = ("yes", "ok", "okay", "sure", "lets do", "let's do", "haan", "go ahead", "proceed", "sounds good")


def _looks_hostile(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in HOSTILE_HINTS)


def _looks_positive(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in POSITIVE_HINTS)


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    meta = store.get_conversation_meta(body.conversation_id)
    if meta is None:
        # Judge is replying to a conversation we don't recognize (shouldn't
        # normally happen) — bootstrap it from what was given so we can still
        # try to help rather than failing the call.
        store.start_conversation(body.conversation_id, body.merchant_id or "", body.customer_id, None)
        meta = {"merchant_id": body.merchant_id, "customer_id": body.customer_id, "trigger_id": None}

    merchant_id = meta.get("merchant_id") or body.merchant_id
    customer_id = meta.get("customer_id") or body.customer_id
    trigger_id = meta.get("trigger_id")

    merchant = store.get("merchant", merchant_id) if merchant_id else None
    category = None
    if merchant:
        category_slug = merchant.get("category_slug")
        category = store.get("category", category_slug) if category_slug else None
    trigger = store.get("trigger", trigger_id) if trigger_id else None
    customer = store.get("customer", customer_id) if customer_id else None

    history = store.get_conversation(body.conversation_id)

    # Auto-reply signal: how many times has this exact text already come
    # from the same sender in this conversation, OR from this merchant
    # across any conversation (canned auto-replies can land on a fresh
    # conversation_id each time)?
    conv_repeat_count = sum(
        1 for t in history
        if t.get("from") == body.from_role and t.get("message") == body.message
    )
    merchant_repeat_count = store.record_incoming_and_count_prior(merchant_id or "", body.message)
    repeat_count = max(conv_repeat_count, merchant_repeat_count)

    store.append_turn(body.conversation_id, {
        "from": body.from_role,
        "message": body.message,
        "ts": body.received_at,
    })

    # Deterministic auto-reply cutoff — don't leave this to LLM judgment once
    # we have hard evidence (3+ prior identical occurrences from this sender).
    # Skip this cutoff for messages that clearly show human agreement/intent —
    # a canned auto-reply is boilerplate, never a commitment sentence.
    if repeat_count >= 3 and not _looks_positive(body.message):
        logger.info("Auto-reply confirmed for %s after %d prior occurrences; ending", merchant_id, repeat_count)
        return {
            "action": "end",
            "rationale": "Same message seen 3+ times from this sender — treating as a canned auto-reply.",
        }

    if merchant is None or category is None:
        logger.warning("Missing context for conversation %s (merchant_id=%s)", body.conversation_id, merchant_id)
        return {"action": "end", "rationale": "Missing merchant/category context to continue safely"}

    try:
        decision = composer.compose_reply(
            category, merchant, trigger, customer,
            conversation_history=history,
            incoming_message=body.message,
            repeat_count=repeat_count,
        )
    except Exception:
        logger.exception("compose_reply failed for conversation %s", body.conversation_id)
        return {"action": "end", "rationale": "Internal composition error"}

    action = decision.get("action", "end")

    # Safety net: an "end" on a clearly positive/agreeable, non-hostile message
    # is always a bug (see composer.py rule 10) — force it into a helpful send
    # rather than silently dropping the merchant mid-commitment.
    if action == "end" and _looks_positive(body.message) and not _looks_hostile(body.message):
        logger.warning(
            "compose_reply returned action=end on a positive message in %s; overriding to send",
            body.conversation_id,
        )
        action = "send"
        decision = {
            "action": "send",
            "body": "Great — I'll get that sorted and follow up shortly with the next step.",
            "cta": "none",
            "rationale": "safety-net override: positive/agreeable message must not end the conversation",
        }

    if action == "send":
        if not decision.get("body"):
            return {"action": "end", "rationale": "Composer returned empty body; ending gracefully"}
        store.append_turn(body.conversation_id, {
            "from": "vera", "message": decision["body"], "ts": body.received_at,
        })
        return {
            "action": "send",
            "body": decision["body"],
            "cta": decision.get("cta", "none"),
            "rationale": decision.get("rationale", ""),
        }

    if action == "wait":
        return {
            "action": "wait",
            "wait_seconds": decision.get("wait_seconds", 1800),
            "rationale": decision.get("rationale", ""),
        }

    # action == "end"
    result = {"action": "end", "rationale": decision.get("rationale", "")}
    if decision.get("body"):
        store.append_turn(body.conversation_id, {
            "from": "vera", "message": decision["body"], "ts": body.received_at,
        })
    return result