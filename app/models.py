"""
Typed Pydantic models for EmailTriageEnv — OpenEnv spec compliant.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Email data types ──────────────────────────────────────────────────────────

class ThreadMessage(BaseModel):
    turn: int
    sender: str
    timestamp: str
    body: str


class EmailRecord(BaseModel):
    id: str
    subject: str
    sender: str
    sender_trust_score: float = Field(ge=0.0, le=1.0)
    body: str
    category: str
    tags: List[str]
    priority: int  # 1 = most urgent
    is_adversarial: bool = False
    adversarial_type: Optional[str] = None
    required_reply_points: List[str] = []
    summary_keywords: List[str] = []


class ThreadRecord(BaseModel):
    id: str
    subject: str
    category: str
    key_issue: str
    key_keywords: List[str]
    messages: List[ThreadMessage]


# ── OpenEnv core types ────────────────────────────────────────────────────────

class Observation(BaseModel):
    """Returned by reset() and step()."""
    task_name: str
    email_id: str
    subject: str
    sender: str
    sender_trust_score: float
    body: str
    thread: List[ThreadMessage] = []
    available_categories: List[str] = []
    available_tags: List[str] = []
    emails_to_rank: List[Dict[str, Any]] = []
    step: int = 0
    done: bool = False
    info: Dict[str, Any] = {}


class Action(BaseModel):
    """Agent action payload."""
    task: str
    action: Dict[str, Any]


class StepResult(BaseModel):
    """Returned by POST /step."""
    observation: Observation
    reward: float = Field(gt=0.0, lt=1.0, description="Strictly between 0 and 1")
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    """Returned by POST /reset."""
    observation: Observation
    info: Dict[str, Any] = {}


class StateResponse(BaseModel):
    """Returned by GET /state."""
    task_name: str
    email_id: str
    step: int
    done: bool
    total_reward: float
    episode_rewards: List[float]
    current_observation: Optional[Observation] = None


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    email_id: Optional[str] = None


class TaskInfo(BaseModel):
    name: str
    difficulty: str
    description: str
    baseline_score: float
