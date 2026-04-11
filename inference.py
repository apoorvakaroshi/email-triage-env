#!/usr/bin/env python3
"""
EmailTriageEnv — Baseline Inference Script
==========================================

Runs an LLM agent against all 6 tasks and emits structured stdout logs
in the required [START] / [STEP] / [END] format.

Environment variables required:
    API_BASE_URL  — LLM API base URL  (e.g. https://api.openai.com/v1)
    MODEL_NAME    — Model identifier  (e.g. gpt-4o-mini)
    HF_TOKEN      — API key / HF token
    ENV_URL       — EmailTriageEnv base URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.environ.get("HF_TOKEN", "no-key")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK = "EmailTriageEnv"
TEMPERATURE = 0.1
MAX_TOKENS = 800
MAX_STEPS = 1          # all tasks are single-step
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.50

TASKS = [
    "email_classification",
    "inbox_prioritization",
    "email_tagging",
    "reply_drafting",
    "email_summarization",
    "thread_classification",
]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage AI agent. Process the given email and respond with a JSON action.

Task formats:
- email_classification:
    {"task": "email_classification", "action": {"category": "<billing|support|spam|urgent|general|newsletter|complaint>", "is_adversarial": <true|false>}}
    Note: set is_adversarial=true if sender_trust_score < 0.25 or email looks like phishing.

- inbox_prioritization:
    {"task": "inbox_prioritization", "action": {"ranking": ["<most_urgent_id>", ..., "<least_urgent_id>"]}}

- email_tagging:
    {"task": "email_tagging", "action": {"tags": ["<tag1>", "<tag2>", ...]}}
    Available tags: billing, payment, invoice, overdue, support, technical, login, api, urgent, critical, security, spam, phishing, newsletter, complaint, refund, how-to, export, missing-item, social-engineering, subscription, meeting

- reply_drafting:
    {"task": "reply_drafting", "action": {"reply": "<full professional reply text>"}}
    Cover all required talking points. Include greeting and closing.

- email_summarization:
    {"task": "email_summarization", "action": {"summary": "<10 to 60 word summary>"}}

- thread_classification:
    {"task": "thread_classification", "action": {"category": "<category>", "key_issue": "<one concise sentence describing the main issue>"}}

Respond ONLY with valid JSON. No explanation, no markdown, no code fences."""

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Any,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_str = json.dumps(action) if not isinstance(action, str) else action
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def env_reset(task_name: str) -> Dict[str, Any]:
    resp = httpx.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(task: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(
        f"{ENV_URL}/step",
        json={"task": task, "action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ── LLM agent ─────────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any]) -> str:
    task = obs.get("task_name", "")
    lines: List[str] = [f"Task: {task}"]

    if task == "inbox_prioritization":
        emails = obs.get("emails_to_rank", [])
        lines.append("Emails to rank (most to least urgent):")
        for e in emails:
            lines.append(
                f"  ID={e['id']} | Subject: {e['subject']} | "
                f"Sender: {e['sender']} | TrustScore: {e['sender_trust_score']:.2f}"
            )
            lines.append(f"  Body: {e['body'][:200]}")
    elif task == "thread_classification":
        thread = obs.get("thread", [])
        lines.append(f"Subject: {obs.get('subject', '')}")
        lines.append("Thread messages:")
        for msg in thread:
            lines.append(f"  [Turn {msg['turn']}] {msg['sender']} ({msg['timestamp']}): {msg['body'][:300]}")
        lines.append(f"Available categories: {obs.get('available_categories', [])}")
    else:
        lines.append(f"Subject: {obs.get('subject', '')}")
        lines.append(f"Sender: {obs.get('sender', '')} (trust score: {obs.get('sender_trust_score', 0):.2f})")
        lines.append(f"Body:\n{obs.get('body', '')}")

        if task == "email_classification":
            lines.append(f"Available categories: {obs.get('available_categories', [])}")
        elif task == "email_tagging":
            lines.append(f"Available tags: {obs.get('available_tags', [])}")
        elif task == "reply_drafting":
            info = obs.get("info", {})
            pts = info.get("required_talking_points", [])
            if pts:
                lines.append(f"Required talking points: {pts}")
        elif task == "email_summarization":
            lines.append("Summarize in 10-60 words.")

    return "\n".join(lines)


def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    task = obs.get("task_name", "")
    user_prompt = build_user_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        return parsed.get("action", parsed)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Safe fallback actions per task
        fallbacks = {
            "email_classification": {"category": "general", "is_adversarial": False},
            "inbox_prioritization": {"ranking": ["E015", "E004", "E007", "E003", "E005"]},
            "email_tagging": {"tags": ["support"]},
            "reply_drafting": {"reply": "Dear Customer,\n\nThank you for reaching out. We acknowledge your request and will resolve it promptly.\n\nBest regards,\nSupport Team"},
            "email_summarization": {"summary": "Customer contacted support with an issue requiring resolution."},
            "thread_classification": {"category": "support", "key_issue": "Customer reported an issue requiring resolution."},
        }
        return fallbacks.get(task, {"category": "general"})


# ── Main ──────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str) -> float:
    """Run one episode for a single task. Returns the score."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    try:
        # Reset
        reset_result = env_reset(task_name)
        obs = reset_result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # Agent decides action
            action = get_llm_action(client, obs)

            # Step
            step_result = env_step(task=task_name, action=action)
            reward = float(step_result.get("reward", 0.001))
            done = bool(step_result.get("done", True))
            error = step_result.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step
            obs = step_result.get("observation", obs)

            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                break

        score = rewards[-1] if rewards else 0.001
        # Clamp to open interval (0, 1) — strictly between endpoints
        score = max(0.001, min(0.999, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        score = 0.001
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    # Wait for env to be ready
    print(f"[INFO] Connecting to {ENV_URL} ...", flush=True)
    for attempt in range(30):
        try:
            r = httpx.get(f"{ENV_URL}/health", timeout=5)
            if r.status_code == 200:
                print("[INFO] Environment is healthy.", flush=True)
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("[ERROR] Environment not reachable after 60s", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: Dict[str, float] = {}
    for task in TASKS:
        score = run_task(client, task)
        all_scores[task] = score
        print(f"[INFO] {task}: {score:.4f}", flush=True)
        time.sleep(1)  # brief pause between tasks

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"\n[SUMMARY] Average score: {avg:.4f}", flush=True)
    for task, s in all_scores.items():
        print(f"  {task}: {s:.4f}", flush=True)


if __name__ == "__main__":
    main()
