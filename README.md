---
title: EmailTriageEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📧 EmailTriageEnv

> 🌍 A real-world email triage environment for training and evaluating AI agents.

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://openenv.dev)
[![HF Spaces](https://img.shields.io/badge/HF-Spaces-yellow)](https://huggingface.co/spaces/appuk10/email-triage-env)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://hub.docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com)

---

## 🧠 Overview

EmailTriageEnv simulates a real workplace email inbox. AI agents must classify, prioritize, tag, summarize, and reply to **15 realistic emails** — including adversarial phishing attempts and multi-turn conversation threads.

> ⚠️ All reward scores are **strictly between 0 and 1** — the open interval `(0, 1)`. Endpoints are excluded by design.

---

## 📋 Tasks

| # | 🏷️ Name | 💪 Difficulty | 📝 Description |
|---|---------|--------------|----------------|
| 1 | `email_classification` | 🟢 Easy | Classify into 7 categories + detect adversarial emails |
| 2 | `inbox_prioritization` | 🟡 Medium | Rank 5 emails by urgency (Kendall tau) |
| 3 | `email_tagging` | 🟠 Medium-Hard | Apply tags from 12-tag vocab (F1 scored) |
| 4 | `reply_drafting` | 🔴 Hard | Draft professional reply covering required talking points |
| 5 | `email_summarization` | 🟡 Medium | Summarize email in 10-60 words |
| 6 | `thread_classification` | 🔴 Hard | Classify multi-turn thread + extract key issue |

---

## ✨ Unique Features

- 🎭 **Adversarial emails** — Phishing and social-engineering emails that agents must detect
- 🔗 **Multi-turn threads** — 3 realistic back-and-forth email chains for thread tasks
- 🔒 **Sender trust scores** — Every email includes a pre-computed domain trust score (0–1)
- 📐 **Strict score range** — All graders clamp to (0.001, 0.999) — never touching 0 or 1
- 🧮 **Kendall tau ranking** — Graded ordering metric for prioritization task
- 🎯 **F1 tag scoring** — Partial credit for tag overlap

---

## 👁️ Observation Space

```json
{
  "task_name": "email_classification",
  "email_id": "E004",
  "subject": "CRITICAL: Production server down",
  "sender": "alerts@monitoring.ourcompany.com",
  "sender_trust_score": 0.91,
  "body": "...",
  "thread": [],
  "available_categories": ["billing","support","spam","urgent","general","newsletter","complaint"],
  "available_tags": ["billing","payment","invoice","..."],
  "emails_to_rank": [],
  "step": 0,
  "done": false,
  "info": {}
}
```

## 🕹️ Action Space

```json
// 🏷️ email_classification
{"task": "email_classification", "action": {"category": "urgent", "is_adversarial": false}}

// 📊 inbox_prioritization
{"task": "inbox_prioritization", "action": {"ranking": ["E015","E004","E007","E003","E005"]}}

// 🔖 email_tagging
{"task": "email_tagging", "action": {"tags": ["urgent","critical","security"]}}

// ✍️ reply_drafting
{"task": "reply_drafting", "action": {"reply": "Dear John,\n\nThank you for..."}}

// 📝 email_summarization
{"task": "email_summarization", "action": {"summary": "Production server down affecting 45K users."}}

// 🗂️ thread_classification
{"task": "thread_classification", "action": {"category": "billing", "key_issue": "Customer disputes invoice amount."}}
```

---

## 🏆 Reward Function

All scores returned as `float` strictly in `(0.0, 1.0)`:

| 📌 Task | 📊 Scoring Method |
|---------|------------------|
| 🏷️ Classification | Exact match → 0.90, related → 0.45, wrong → 0.05 |
| 📊 Prioritization | Kendall tau mapped from [-1,1] to (0,1) |
| 🔖 Tagging | F1 score scaled to (0.05, 0.95) |
| ✍️ Reply Drafting | Talking-point coverage + professionalism markers |
| 📝 Summarization | Length compliance + key-term coverage |
| 🗂️ Thread Classification | Category match (50%) + keyword coverage (50%) |

---

## 📈 Baseline Scores

| 📌 Task | 🎯 Score |
|---------|---------|
| 🏷️ email_classification | **0.90** |
| 📊 inbox_prioritization | **0.78** |
| 🔖 email_tagging | **0.74** |
| ✍️ reply_drafting | **0.71** |
| 📝 email_summarization | **0.80** |
| 🗂️ thread_classification | **0.68** |

---

## 🚀 Setup

### 🐳 Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### 💻 Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

---

## 🤖 Running Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_URL=http://localhost:7860

python inference.py
```

---

## 🔌 API Endpoints

| Method | 🛣️ Path | 📝 Description |
|--------|---------|---------------|
| GET | `/health` | ❤️ Liveness probe |
| GET | `/tasks` | 📋 List all tasks |
| POST | `/reset` | 🔄 Reset, get initial observation |
| POST | `/step` | 👣 Take action, get reward |
| GET | `/state` | 📊 Current environment state |
| GET | `/docs` | 📖 Swagger UI |
| GET | `/redoc` | 📚 ReDoc documentation |

---

## 🗂️ Project Structure

```
email_triage_env/
├── 📁 app/
│   ├── __init__.py
│   ├── main.py        # ⚡ FastAPI routes
│   ├── models.py      # 🧱 Typed Pydantic models
│   ├── data.py        # 📧 Email + thread dataset
│   ├── graders.py     # 🎯 Scoring functions
│   └── environment.py # 🌍 Core env logic
├── 📁 static/
│   └── index.html     # 🎨 Beautiful interactive homepage
├── inference.py       # 🤖 Baseline inference script
├── openenv.yaml       # 📋 OpenEnv metadata
├── Dockerfile         # 🐳 Container setup
├── requirements.txt   # 📦 Dependencies
└── README.md          # 📖 This file
```

---

Built with by [Apoorva Karoshi](https://huggingface.co/appuk10) · Powered by [OpenEnv](https://openenv.dev) & [Hugging Face](https://huggingface.co)