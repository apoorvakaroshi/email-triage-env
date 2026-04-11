"""
EmailTriageEnvironment — core OpenEnv implementation.

Implements:  reset()  step()  state()
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from app.data import (
    ALL_CATEGORIES,
    ALL_TAGS,
    EMAILS,
    PRIORITIZATION_EMAIL_IDS,
    PRIORITIZATION_GROUND_TRUTH_ORDER,
    THREADS,
    EmailRecord,
    ThreadRecord,
)
from app.graders import (
    grade_classification,
    grade_prioritization,
    grade_reply,
    grade_summarization,
    grade_tagging,
    grade_thread_classification,
)
from app.models import (
    Action,
    Observation,
    ResetResult,
    StateResponse,
    StepResult,
    TaskInfo,
)

# ── Task registry ─────────────────────────────────────────────────────────────

TASK_INFO: List[TaskInfo] = [
    TaskInfo(
        name="email_classification",
        difficulty="easy",
        description="Classify an email into one of 7 categories: billing, support, spam, urgent, general, newsletter, complaint",
        baseline_score=0.90,
    ),
    TaskInfo(
        name="inbox_prioritization",
        difficulty="medium",
        description="Rank 5 emails from most to least urgent. Scored by Kendall tau correlation.",
        baseline_score=0.78,
    ),
    TaskInfo(
        name="email_tagging",
        difficulty="medium-hard",
        description="Apply all relevant tags from a 12-tag vocabulary. Scored by F1.",
        baseline_score=0.74,
    ),
    TaskInfo(
        name="reply_drafting",
        difficulty="hard",
        description="Draft a professional reply covering all required talking points.",
        baseline_score=0.71,
    ),
    TaskInfo(
        name="email_summarization",
        difficulty="medium",
        description="Summarize an email in 10-60 words covering all key points.",
        baseline_score=0.80,
    ),
    TaskInfo(
        name="thread_classification",
        difficulty="hard",
        description="Classify a multi-turn email thread and extract the key issue in one sentence.",
        baseline_score=0.68,
    ),
]

TASK_NAMES = [t.name for t in TASK_INFO]


def _email_to_obs_dict(email: EmailRecord) -> Dict[str, Any]:
    return {
        "id": email.id,
        "subject": email.subject,
        "sender": email.sender,
        "sender_trust_score": email.sender_trust_score,
        "body": email.body[:300] + ("…" if len(email.body) > 300 else ""),
    }


class EmailTriageEnvironment:
    """Stateful OpenEnv-compliant email triage environment."""

    def __init__(self) -> None:
        self._task_name: str = TASK_NAMES[0]
        self._current_email: Optional[EmailRecord] = None
        self._current_thread: Optional[ThreadRecord] = None
        self._step: int = 0
        self._done: bool = False
        self._episode_rewards: List[float] = []
        self._total_reward: float = 0.0
        self._current_obs: Optional[Observation] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> ResetResult:
        """Reset environment to a new episode."""
        if task_name and task_name in TASK_NAMES:
            self._task_name = task_name
        else:
            self._task_name = random.choice(TASK_NAMES)

        self._step = 0
        self._done = False
        self._episode_rewards = []
        self._total_reward = 0.0

        obs = self._build_observation()
        self._current_obs = obs
        return ResetResult(observation=obs, info={"task": self._task_name})

    def step(self, action: Action) -> StepResult:
        """Take one action and return reward + next observation."""
        if self._done:
            obs = self._current_obs or self._build_observation()
            return StepResult(
                observation=obs,
                reward=0.001,
                done=True,
                info={"warning": "Episode already done. Call /reset first."},
            )

        self._step += 1
        reward, info = self._grade_action(action)
        self._done = True  # all tasks are single-step
        self._episode_rewards.append(reward)
        self._total_reward += reward

        # Build terminal observation
        obs = self._build_observation(done=True)
        self._current_obs = obs

        return StepResult(observation=obs, reward=reward, done=True, info=info)

    def state(self) -> StateResponse:
        """Return current environment state."""
        return StateResponse(
            task_name=self._task_name,
            email_id=self._current_email.id if self._current_email else (
                self._current_thread.id if self._current_thread else ""
            ),
            step=self._step,
            done=self._done,
            total_reward=self._total_reward,
            episode_rewards=self._episode_rewards,
            current_observation=self._current_obs,
        )

    def get_tasks(self) -> Dict[str, Any]:
        return {
            "tasks": [t.model_dump() for t in TASK_INFO],
            "count": len(TASK_INFO),
            "categories": ALL_CATEGORIES,
            "tags": ALL_TAGS,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(self, done: bool = False) -> Observation:
        task = self._task_name

        if task == "email_classification":
            email = self._pick_email()
            return Observation(
                task_name=task,
                email_id=email.id,
                subject=email.subject,
                sender=email.sender,
                sender_trust_score=email.sender_trust_score,
                body=email.body,
                available_categories=ALL_CATEGORIES,
                step=self._step,
                done=done,
                info={
                    "instruction": (
                        "Classify this email. If the sender_trust_score < 0.25, "
                        "also set is_adversarial=true in your action."
                    ),
                    "sender_trust_warning": email.sender_trust_score < 0.30,
                },
            )

        if task == "inbox_prioritization":
            prio_emails = [e for e in EMAILS if e.id in PRIORITIZATION_EMAIL_IDS]
            random.shuffle(prio_emails)
            self._current_email = None
            return Observation(
                task_name=task,
                email_id="BATCH",
                subject="[Inbox Prioritization]",
                sender="system",
                sender_trust_score=0.99,
                body="Rank the provided emails from most to least urgent.",
                emails_to_rank=[_email_to_obs_dict(e) for e in prio_emails],
                step=self._step,
                done=done,
                info={
                    "instruction": (
                        "Provide a 'ranking' list of email IDs ordered from "
                        "most urgent (index 0) to least urgent."
                    )
                },
            )

        if task == "email_tagging":
            email = self._pick_email()
            return Observation(
                task_name=task,
                email_id=email.id,
                subject=email.subject,
                sender=email.sender,
                sender_trust_score=email.sender_trust_score,
                body=email.body,
                available_tags=ALL_TAGS,
                step=self._step,
                done=done,
                info={"instruction": "Apply all relevant tags from available_tags."},
            )

        if task == "reply_drafting":
            # Avoid adversarial emails for reply drafting
            pool = [e for e in EMAILS if not e.is_adversarial and e.required_reply_points]
            email = random.choice(pool)
            self._current_email = email
            return Observation(
                task_name=task,
                email_id=email.id,
                subject=email.subject,
                sender=email.sender,
                sender_trust_score=email.sender_trust_score,
                body=email.body,
                step=self._step,
                done=done,
                info={
                    "instruction": (
                        "Draft a professional reply. Cover all talking points. "
                        "Action format: {'reply': '<your reply text>'}"
                    ),
                    "required_talking_points": email.required_reply_points,
                    "sender_name": email.sender.split("@")[0].replace(".", " ").title(),
                },
            )

        if task == "email_summarization":
            pool = [e for e in EMAILS if not e.is_adversarial and e.summary_keywords]
            email = random.choice(pool)
            self._current_email = email
            return Observation(
                task_name=task,
                email_id=email.id,
                subject=email.subject,
                sender=email.sender,
                sender_trust_score=email.sender_trust_score,
                body=email.body,
                step=self._step,
                done=done,
                info={
                    "instruction": "Summarize this email in 10-60 words covering all key points.",
                    "word_limit": {"min": 10, "max": 60},
                },
            )

        if task == "thread_classification":
            thread = random.choice(THREADS)
            self._current_thread = thread
            self._current_email = None
            return Observation(
                task_name=task,
                email_id=thread.id,
                subject=thread.subject,
                sender="[thread]",
                sender_trust_score=0.75,
                body=f"Multi-turn thread with {len(thread.messages)} messages. See 'thread' field.",
                thread=[m for m in thread.messages],
                available_categories=ALL_CATEGORIES,
                step=self._step,
                done=done,
                info={
                    "instruction": (
                        "Action format: {'category': '<category>', 'key_issue': '<one sentence>'}"
                    )
                },
            )

        # Fallback
        return Observation(
            task_name=task, email_id="", subject="", sender="",
            sender_trust_score=0.50, body="", step=self._step, done=done,
        )

    def _pick_email(self) -> EmailRecord:
        email = random.choice(EMAILS)
        self._current_email = email
        return email

    def _grade_action(self, action: Action) -> tuple[float, Dict[str, Any]]:
        task = self._task_name
        a = action.action

        # ── Classification ──────────────────────────────────────────────────
        if task == "email_classification":
            email = self._current_email
            if not email:
                return 0.001, {"error": "No email loaded"}
            predicted = str(a.get("category", "")).lower()
            flagged = bool(a.get("is_adversarial", False))
            reward = grade_classification(
                predicted=predicted,
                ground_truth=email.category,
                is_adversarial=email.is_adversarial,
                flagged_adversarial=flagged,
            )
            return reward, {
                "ground_truth_category": email.category,
                "predicted_category": predicted,
                "is_adversarial": email.is_adversarial,
                "flagged_adversarial": flagged,
            }

        # ── Prioritization ──────────────────────────────────────────────────
        if task == "inbox_prioritization":
            predicted_order: List[str] = a.get("ranking", [])
            reward = grade_prioritization(
                predicted_order=predicted_order,
                ground_truth_order=PRIORITIZATION_GROUND_TRUTH_ORDER,
            )
            return reward, {
                "ground_truth_order": PRIORITIZATION_GROUND_TRUTH_ORDER,
                "predicted_order": predicted_order,
            }

        # ── Tagging ─────────────────────────────────────────────────────────
        if task == "email_tagging":
            email = self._current_email
            if not email:
                return 0.001, {"error": "No email loaded"}
            predicted_tags: List[str] = a.get("tags", [])
            reward = grade_tagging(
                predicted_tags=predicted_tags,
                ground_truth_tags=email.tags,
            )
            return reward, {
                "ground_truth_tags": email.tags,
                "predicted_tags": predicted_tags,
            }

        # ── Reply Drafting ──────────────────────────────────────────────────
        if task == "reply_drafting":
            email = self._current_email
            if not email:
                return 0.001, {"error": "No email loaded"}
            reply = str(a.get("reply", ""))
            sender_name = email.sender.split("@")[0].replace(".", " ").title()
            reward = grade_reply(
                reply=reply,
                required_points=email.required_reply_points,
                sender_name=sender_name,
            )
            return reward, {
                "required_talking_points": email.required_reply_points,
                "word_count": len(reply.split()),
            }

        # ── Summarization ────────────────────────────────────────────────────
        if task == "email_summarization":
            email = self._current_email
            if not email:
                return 0.001, {"error": "No email loaded"}
            summary = str(a.get("summary", ""))
            reward = grade_summarization(
                summary=summary,
                key_terms=email.summary_keywords,
            )
            return reward, {
                "key_terms": email.summary_keywords,
                "word_count": len(summary.split()),
            }

        # ── Thread Classification ────────────────────────────────────────────
        if task == "thread_classification":
            thread = self._current_thread
            if not thread:
                return 0.001, {"error": "No thread loaded"}
            predicted_category = str(a.get("category", ""))
            key_issue = str(a.get("key_issue", ""))
            reward = grade_thread_classification(
                predicted_category=predicted_category,
                key_issue=key_issue,
                ground_truth_category=thread.category,
                ground_truth_keywords=thread.key_keywords,
            )
            return reward, {
                "ground_truth_category": thread.category,
                "ground_truth_key_issue": thread.key_issue,
                "predicted_category": predicted_category,
            }

        return 0.001, {"error": f"Unknown task: {task}"}
