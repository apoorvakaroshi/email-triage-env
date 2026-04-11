"""
EmailTriageEnv — FastAPI application.
OpenEnv-compliant endpoints: /health /tasks /reset /step /state
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.environment import EmailTriageEnvironment
from app.models import Action, ResetRequest, StepResult, ResetResult, StateResponse

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EmailTriageEnv",
    description=(
        "A real-world email triage environment for training and evaluating AI agents. "
        "Supports 6 tasks: classification, prioritization, tagging, reply drafting, "
        "summarization, and multi-turn thread classification."
    ),
    version="1.0.0",
)

env = EmailTriageEnvironment()

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage() -> str:
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "healthy", "version": "1.0.0", "env": "EmailTriageEnv"}


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
        return env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse, summary="Get current environment state")
async def state() -> StateResponse:
    return env.state()
