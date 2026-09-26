"""
composer.py — turns (category, merchant, trigger, customer) contexts into
a composed WhatsApp message using Groq, following the rules in
challenge-brief.md (§4 framework, §5 constraints, §9 patterns, §10 levers,
§11 anti-patterns).

Two entry points:
  - compose_message(...)  -> used by POST /v1/tick to start a new outbound
  - compose_reply(...)    -> used by POST /v1/reply to continue a conversation
"""

import os
import json
import logging
from typing import Optional, Any

from groq import Groq

logger = logging.getLogger("composer")

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (check your .env file)")
        _client = Groq(api_key=api_key)
    return _client


MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — encodes the whole spec so the LLM judges itself the way the
# real judge will. Keep this in sync with challenge-brief.md if it changes.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp AI assistant for local merchants \
in India (dentists, salons, restaurants, gyms, pharmacies, etc). You compose short, \
high-engagement WhatsApp messages, either to the merchant directly, or on the \
merchant's behalf to one of their customers.

You are given four context layers as JSON:
- category: slow-changing knowledge about the business vertical (voice, offer_catalog,
  peer_stats, digest research items, seasonal_beats, trend_signals)
- merchant: this specific business's current state (identity, subscription,
  performance, offers, conversation_history, customer_aggregate, signals)
- trigger: the event that justifies messaging right now (kind, urgency, payload)
- customer (optional): populated only for customer-facing messages

RULES YOU MUST FOLLOW:
1. Anchor the message on a concrete, verifiable fact from the given contexts
   (a number, a date, a headline, a peer stat, a signal). Never invent facts,
   citations, competitor names, or offers that are not present in the contexts.
2. Match the category's voice exactly. Clinical/peer categories (dentists, doctors,
   lawyers) must NOT sound promotional ("AMAZING DEAL!"); use technical vocabulary
   if the category allows it. Retail/lifestyle categories can be warmer.
3. Prefer service+price framing ("Haircut @ ₹99") over generic discount framing
   ("10% off") whenever the category's offer_catalog has service+price options.
4. Personalize to THIS merchant: reference their actual numbers, their actual
   offers, their actual signals (e.g. "ctr_below_peer_median", "stale_posts").
5. Make the "why now" (the trigger) explicit and specific — never a generic
   "you should improve your profile" framing.
6. Use at least one compulsion lever: specificity, loss aversion, social proof,
   effort externalization, curiosity, reciprocity, asking the merchant a question,
   or a single binary commitment (YES/STOP). Prefer social proof and "asking the
   merchant" — these are underused and score well.
7. Exactly ONE primary call-to-action. Never stack multiple CTAs
   ("Reply YES for X, NO for Y"). For pure-information triggers, cta can be "none".
8. Put the call-to-action in the LAST sentence, not buried in the middle.
9. No long preambles ("I hope you're doing well..."). Get to the point in the
   first sentence. Do not re-introduce yourself if conversation_history is non-empty.
10. Match the merchant's / customer's language preference. Hindi-English code-mix
    (Hinglish) is preferred for Indian merchants unless languages says pure "en".
    ALWAYS write Hinglish in ROMAN/Latin script (e.g. "kal tak bhej dunga"), NEVER in
    Devanagari script (e.g. never "कल तक भेज दूंगा") — WhatsApp business messages in
    India are written in Roman script even when code-mixing Hindi words.
11. Never send the same body verbatim that appears in conversation_history —
    if you must revisit a topic, phrase it differently and reference what changed.
12. Keep it concise — no hard length cap, but every sentence must earn its place.
13. send_as is "vera" when messaging the merchant directly, or "merchant_on_behalf"
    when customer is populated (message goes out under the merchant's own WhatsApp,
    drafted by you).

You must respond with ONLY a JSON object (no markdown fences, no commentary), with
exactly these keys: "body", "cta", "send_as", "suppression_key", "rationale".
- cta must be one of: "binary_yes_stop", "open_ended", "none"
- rationale is one short sentence: why this message, what it should achieve.
"""

REPLY_SYSTEM_PROMPT = """You are Vera, continuing an in-progress WhatsApp conversation \
with a merchant (or a merchant's customer, if customer context is populated). You are \
given the original contexts (category, merchant, trigger, optional customer), the \
conversation so far, and the latest incoming message. Decide your next move.

RULES:
1. Detect merchant auto-replies: if the SAME incoming message text has appeared
   3 or more times already in this conversation from the same role, treat it as a
   WhatsApp Business canned auto-reply. Try at most once more to re-engage a real
   human, then action="end" gracefully and politely if it repeats again.
2. Detect explicit intent / commitment ("yes let's do it", "go ahead", "haan kar do",
   "ok proceed", "whats next"): when detected, ALWAYS action="send" — never "end",
   even if conversation_history is empty or trigger is null (context may be missing
   because this is the first turn of a reply-only test). Do NOT ask another
   qualifying question ("would you like...", "do you want..."). Instead confirm
   what you're doing next in concrete terms — if you don't know the exact specific
   next step (no trigger/history to anchor on), still respond helpfully and
   concretely, e.g. "Great — I'll get that sorted and follow up shortly", never a
   bare acknowledgement and never an "end".
3. If the incoming message is hostile, abusive, or off-topic: if hostile, apologize
   briefly and action="end" (do not escalate or argue). If off-topic but not hostile,
   politely redirect to the mission in one line, no lecture.
4. If the merchant asks for time / says "not now" / "later": action="wait" with a
   reasonable wait_seconds (900-3600), rationale explaining the back-off.
5. If the merchant clearly declines or says not interested: action="end", polite
   sign-off, no further pitching.
6. Otherwise: action="send" with the next best message — advance the conversation,
   honor what was just said, add at most one new low-friction next step.
7. Never repeat, verbatim, any body already sent by you (vera) in this conversation.
8. Same voice/category/CTA rules as composing a fresh message (concise, one CTA,
   category-appropriate voice, specific, no fabrication).
9. Match the language of the incoming message where reasonable (Hindi-English
   code-mix is fine and often expected). Always write in ROMAN/Latin script,
   NEVER in Devanagari script, even when code-mixing Hindi words.
10. action="end" is ONLY for: repeated auto-replies (rule 1), hostility (rule 3),
    or explicit decline (rule 5). A positive/agreeable/commitment message must
    NEVER get action="end" — that is always a bug, not a valid response.

Respond with ONLY a JSON object (no markdown fences, no commentary):
- If action == "send": keys are "action", "body", "cta", "rationale"
  (cta one of "binary_yes_stop", "open_ended", "none")
- If action == "wait": keys are "action", "wait_seconds", "rationale"
- If action == "end": keys are "action", "rationale" (and optionally "body" for a
  polite sign-off message; omit body to end silently)
"""


def _chat_json(system: str, user: str, max_tokens: int = 700) -> dict[str, Any]:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON: %r", raw)
        raise


def _trim(context: Optional[dict], max_chars: int = 4000) -> str:
    """Serialize a context dict to JSON, truncated defensively so a huge
    conversation_history or digest list can't blow the prompt budget."""
    if context is None:
        return "null"
    s = json.dumps(context, ensure_ascii=False)
    if len(s) > max_chars:
        s = s[:max_chars] + '..."<truncated>"'
    return s


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """Compose a brand-new proactive message. Returns dict with keys:
    body, cta, send_as, suppression_key, rationale."""
    user_prompt = f"""CATEGORY CONTEXT:
{_trim(category)}

MERCHANT CONTEXT:
{_trim(merchant)}

TRIGGER CONTEXT:
{_trim(trigger)}

CUSTOMER CONTEXT (null if this is a merchant-facing message):
{_trim(customer)}

Compose the message now. Respond with the JSON object only."""

    data = _chat_json(SYSTEM_PROMPT, user_prompt)

    return {
        "body": str(data.get("body", "")).strip(),
        "cta": data.get("cta", "none"),
        "send_as": data.get("send_as") or ("merchant_on_behalf" if customer else "vera"),
        "suppression_key": data.get("suppression_key") or trigger.get("suppression_key", ""),
        "rationale": data.get("rationale", ""),
    }


def compose_reply(
    category: dict,
    merchant: dict,
    trigger: Optional[dict],
    customer: Optional[dict],
    conversation_history: list[dict],
    incoming_message: str,
    repeat_count: int = 0,
) -> dict:
    """Decide the next move in an ongoing conversation. Returns a dict with
    key "action" in {"send", "wait", "end"} plus the fields relevant to it.

    repeat_count: how many times this exact incoming text has already
    appeared before in this conversation from the same sender (computed by
    the caller from conversation history) — a strong auto-reply signal per
    challenge-testing-brief.md's hint ("same message verbatim 3+ times").
    """
    user_prompt = f"""CATEGORY CONTEXT:
{_trim(category)}

MERCHANT CONTEXT:
{_trim(merchant)}

TRIGGER CONTEXT (the reason this conversation started, may be null):
{_trim(trigger)}

CUSTOMER CONTEXT (null if this is a merchant-facing conversation):
{_trim(customer)}

CONVERSATION SO FAR (oldest first, "from" is "vera" or "merchant"/"customer"):
{_trim(conversation_history)}

LATEST INCOMING MESSAGE:
{incoming_message!r}

AUTO-REPLY SIGNAL: this exact incoming text has already appeared {repeat_count} \
time(s) before in this conversation from the same sender. 0 = first time (treat \
normally). 1-2 = possible auto-reply, you may try once more to reach a human. \
3+ = treat as a confirmed WhatsApp Business canned auto-reply; end gracefully.

Decide your next move now. Respond with the JSON object only."""

    data = _chat_json(REPLY_SYSTEM_PROMPT, user_prompt, max_tokens=450)

    action = data.get("action", "send")
    if action == "send":
        return {
            "action": "send",
            "body": str(data.get("body", "")).strip(),
            "cta": data.get("cta", "none"),
            "rationale": data.get("rationale", ""),
        }
    elif action == "wait":
        return {
            "action": "wait",
            "wait_seconds": int(data.get("wait_seconds", 1800)),
            "rationale": data.get("rationale", ""),
        }
    else:  # "end"
        result = {"action": "end", "rationale": data.get("rationale", "")}
        if data.get("body"):
            result["body"] = str(data["body"]).strip()
        return result