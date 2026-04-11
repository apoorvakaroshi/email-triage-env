"""
EmailTriageEnv — FastAPI application.
OpenEnv-compliant endpoints + bonus: /history /leaderboard /explain
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.environment import EmailTriageEnvironment
from app.models import Action, ResetRequest, StepResult, ResetResult, StateResponse

app = FastAPI(
    title="EmailTriageEnv",
    description=(
        "A real-world email triage environment for training and evaluating AI agents. "
        "6 tasks: classification, prioritization, tagging, reply drafting, "
        "summarization, and multi-turn thread classification. "
        "All rewards strictly in open interval (0, 1)."
    ),
    version="1.0.0",
)

env = EmailTriageEnvironment()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── In-memory stores ──────────────────────────────────────────────────────────
_history: List[Dict[str, Any]] = []
_leaderboard: Dict[str, List[float]] = defaultdict(list)


# ── Core OpenEnv routes ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage() -> str:
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "env": "EmailTriageEnv",
        "tasks": 6,
        "reward_range": "(0, 1) strictly",
    }


@app.get("/tasks", summary="List all tasks")
async def list_tasks() -> dict:
    return env.get_tasks()


@app.post("/reset", response_model=ResetResult, summary="Reset environment, get initial observation")
async def reset(request: ResetRequest | None = None) -> ResetResult:
    task_name = request.task_name if request else None
    return env.reset(task_name)


@app.post("/step", response_model=StepResult, summary="Take action, get reward and next observation")
async def step(action: Action) -> StepResult:
    try:
        result = env.step(action)
        # Record to history and leaderboard
        entry = {
            "timestamp": time.time(),
            "task": action.task,
            "action": action.action,
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
        _history.append(entry)
        _leaderboard[action.task].append(result.reward)
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse, summary="Get current environment state")
async def state() -> StateResponse:
    return env.state()


# ── Bonus endpoints ───────────────────────────────────────────────────────────

@app.get("/history", summary="Full episode history — all actions and rewards this session")
async def history(limit: int = 50) -> dict:
    recent = _history[-limit:]
    return {
        "total_steps": len(_history),
        "showing": len(recent),
        "history": recent,
    }


@app.get("/leaderboard", summary="Best scores per task across all episodes")
async def leaderboard() -> dict:
    board = {}
    for task, scores in _leaderboard.items():
        board[task] = {
            "best": round(max(scores), 4),
            "average": round(sum(scores) / len(scores), 4),
            "attempts": len(scores),
            "all_scores": [round(s, 4) for s in scores[-10:]],
        }
    # Fill in tasks with no attempts
    all_tasks = ["email_classification","inbox_prioritization","email_tagging",
                 "reply_drafting","email_summarization","thread_classification"]
    for t in all_tasks:
        if t not in board:
            board[t] = {"best": None, "average": None, "attempts": 0, "all_scores": []}
    return {"leaderboard": board, "total_attempts": len(_history)}


@app.post("/explain", summary="Explain why a reward score was given for an action")
async def explain(action: Action) -> dict:
    """
    Re-grades the action and returns a detailed breakdown of the score,
    explaining each component and what contributed to the final reward.
    """
    task = action.task
    a = action.action

    explanations = {
        "email_classification": _explain_classification,
        "inbox_prioritization": _explain_prioritization,
        "email_tagging": _explain_tagging,
        "reply_drafting": _explain_reply,
        "email_summarization": _explain_summarization,
        "thread_classification": _explain_thread,
    }

    if task not in explanations:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    # Step to get actual reward
    try:
        result = env.step(action)
        explanation = explanations[task](a, result)
        return {
            "task": task,
            "reward": result.reward,
            "explanation": explanation,
            "info": result.info,
            "tip": _get_tip(task, result.reward),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _explain_classification(action: dict, result: StepResult) -> dict:
    reward = result.reward
    info = result.info
    return {
        "scoring_method": "Exact category match with semantic group partial credit",
        "predicted": info.get("predicted_category", "—"),
        "ground_truth": info.get("ground_truth_category", "—"),
        "is_adversarial_email": info.get("is_adversarial", False),
        "flagged_as_adversarial": info.get("flagged_adversarial", False),
        "score_breakdown": {
            "exact_match_score": 0.90,
            "semantic_group_score": 0.45,
            "wrong_score": 0.05,
            "adversarial_correct": 0.92,
            "your_score": round(reward, 4),
        },
    }


def _explain_prioritization(action: dict, result: StepResult) -> dict:
    info = result.info
    return {
        "scoring_method": "Kendall tau correlation between predicted and ground-truth ranking",
        "ground_truth_order": info.get("ground_truth_order", []),
        "your_order": info.get("predicted_order", []),
        "score_breakdown": {
            "kendall_tau_range": "[-1, 1] mapped to (0, 1)",
            "perfect_ranking": 0.999,
            "random_ranking": 0.50,
            "reversed_ranking": 0.001,
            "your_score": round(result.reward, 4),
        },
    }


def _explain_tagging(action: dict, result: StepResult) -> dict:
    info = result.info
    pred = set(info.get("predicted_tags", []))
    truth = set(info.get("ground_truth_tags", []))
    overlap = pred & truth
    precision = len(overlap) / len(pred) if pred else 0
    recall = len(overlap) / len(truth) if truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "scoring_method": "F1 score between predicted and ground-truth tags",
        "ground_truth_tags": list(truth),
        "your_tags": list(pred),
        "correct_tags": list(overlap),
        "missed_tags": list(truth - pred),
        "extra_tags": list(pred - truth),
        "score_breakdown": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "your_score": round(result.reward, 4),
        },
    }


def _explain_reply(action: dict, result: StepResult) -> dict:
    info = result.info
    reply = action.get("reply", "")
    wc = len(reply.split())
    return {
        "scoring_method": "Talking-point coverage (65%) + professionalism (20%) + length (10%) + personalisation (5%)",
        "word_count": wc,
        "required_talking_points": info.get("required_talking_points", []),
        "score_breakdown": {
            "talking_point_coverage": "65% weight — keywords matched in reply",
            "professionalism": "20% weight — greeting, closing, structure",
            "length": "10% weight — ideal 30-350 words",
            "personalisation": "5% weight — sender name in reply",
            "your_score": round(result.reward, 4),
        },
        "tip": "Include greeting (Dear/Hello), all talking points, and a closing (Best regards)",
    }


def _explain_summarization(action: dict, result: StepResult) -> dict:
    info = result.info
    wc = info.get("word_count", 0)
    return {
        "scoring_method": "Length compliance (40%) + key-term coverage (60%)",
        "word_count": wc,
        "ideal_range": "10-60 words",
        "key_terms": info.get("key_terms", []),
        "score_breakdown": {
            "length_score": "40% weight — must be 10-60 words",
            "key_term_coverage": "60% weight — how many key terms appear",
            "your_score": round(result.reward, 4),
        },
    }


def _explain_thread(action: dict, result: StepResult) -> dict:
    info = result.info
    return {
        "scoring_method": "Category match (50%) + key issue keyword coverage (50%)",
        "predicted_category": info.get("predicted_category", "—"),
        "ground_truth_category": info.get("ground_truth_category", "—"),
        "ground_truth_key_issue": info.get("ground_truth_key_issue", "—"),
        "score_breakdown": {
            "category_match": "50% weight — exact=0.92, related=0.45, wrong=0.05",
            "keyword_coverage": "50% weight — key issue keywords found",
            "your_score": round(result.reward, 4),
        },
    }


def _get_tip(task: str, reward: float) -> str:
    if reward >= 0.85:
        return "🎉 Excellent! Near-perfect score."
    if reward >= 0.65:
        tips = {
            "email_classification": "Check for adversarial signals — low trust score emails need is_adversarial=true",
            "inbox_prioritization": "Security/critical emails should always rank #1 and #2",
            "email_tagging": "Add more tags — recall matters as much as precision",
            "reply_drafting": "Cover all talking points and include Dear/Best regards",
            "email_summarization": "Keep between 10-60 words and mention key facts",
            "thread_classification": "Include more keywords from the thread in your key_issue",
        }
        return "⚠️ Good but improvable. Tip: " + tips.get(task, "Review the scoring criteria")
    return "❌ Low score. Re-read the email carefully and try again."