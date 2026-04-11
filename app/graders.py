"""
Grader functions for all 6 EmailTriageEnv tasks.

CRITICAL: Every public grade_*() function MUST return a value strictly between
0 and 1 (exclusive). We enforce this via the clamp() helper which maps any
raw score to the open interval (EPSILON, 1 - EPSILON).
"""

from __future__ import annotations

from typing import List

# ── Score clamping ─────────────────────────────────────────────────────────────

EPSILON = 0.001  # guarantees strict (0, 1)


def clamp(score: float) -> float:
    """Return score strictly between 0 and 1 — endpoints are excluded."""
    return max(EPSILON, min(1.0 - EPSILON, float(score)))


# ── Task 1: Email Classification ───────────────────────────────────────────────

# Semantic similarity groups — misclassifying within a group is partially correct.
_CATEGORY_GROUPS: list[set[str]] = [
    {"billing", "newsletter"},       # both commercial, low confusion penalty
    {"support", "complaint"},        # both customer-facing
    {"spam"},
    {"urgent"},
    {"general"},
]


def grade_classification(
    predicted: str,
    ground_truth: str,
    is_adversarial: bool = False,
    flagged_adversarial: bool = False,
) -> float:
    """
    Score email classification.

    - Adversarial email correctly flagged → 0.95
    - Adversarial email missed (not flagged as spam/phishing) → 0.05
    - Exact category match → 0.90
    - Same semantic group → 0.45
    - Wrong → 0.05
    """
    predicted = (predicted or "").strip().lower()
    ground_truth = (ground_truth or "").strip().lower()

    if is_adversarial:
        raw = 0.92 if flagged_adversarial else 0.05
        return clamp(raw)

    if predicted == ground_truth:
        return clamp(0.90)

    for group in _CATEGORY_GROUPS:
        if predicted in group and ground_truth in group:
            return clamp(0.45)

    return clamp(0.05)


# ── Task 2: Inbox Prioritization ──────────────────────────────────────────────

def _kendall_tau(predicted_order: List[str], ground_truth_order: List[str]) -> float:
    """
    Compute Kendall tau between two orderings of the same items.
    Returns a value in [-1, 1].
    """
    n = len(ground_truth_order)
    if n <= 1:
        return 1.0

    # Map each id → rank (lower rank = more urgent)
    gt_rank = {email_id: i for i, email_id in enumerate(ground_truth_order)}
    pred_rank = {email_id: i for i, email_id in enumerate(predicted_order)}

    concordant = discordant = 0
    ids = list(gt_rank.keys())
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            # In ground truth a < b (a more urgent)
            gt_order = gt_rank[a] < gt_rank[b]
            # In prediction
            if a in pred_rank and b in pred_rank:
                pred_order = pred_rank[a] < pred_rank[b]
                if gt_order == pred_order:
                    concordant += 1
                else:
                    discordant += 1

    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def grade_prioritization(
    predicted_order: List[str],
    ground_truth_order: List[str],
) -> float:
    """
    Score inbox prioritization using Kendall tau.
    tau ∈ [-1, 1] → mapped to (EPSILON, 1-EPSILON).
    """
    if not predicted_order:
        return clamp(0.05)

    # Normalise casing
    predicted_order = [x.strip().upper() for x in predicted_order]
    ground_truth_order = [x.strip().upper() for x in ground_truth_order]

    # Accept partial lists — fill missing items at the end
    missing = [x for x in ground_truth_order if x not in predicted_order]
    predicted_order = predicted_order + missing
    # Trim to same length as ground truth
    predicted_order = predicted_order[: len(ground_truth_order)]

    tau = _kendall_tau(predicted_order, ground_truth_order)
    # tau in [-1, 1] → score in [0, 1]
    raw = (tau + 1.0) / 2.0
    return clamp(raw)


# ── Task 3: Email Tagging ─────────────────────────────────────────────────────

def grade_tagging(
    predicted_tags: List[str],
    ground_truth_tags: List[str],
) -> float:
    """
    Score tag prediction via F1.
    F1 ∈ [0, 1] → mapped to (0.05, 0.95) then clamped to (EPSILON, 1-EPSILON).
    """
    if not ground_truth_tags:
        return clamp(0.50)

    pred_set = {t.strip().lower() for t in predicted_tags if t}
    truth_set = {t.strip().lower() for t in ground_truth_tags}

    if not pred_set:
        return clamp(0.05)

    overlap = pred_set & truth_set
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(truth_set)

    if precision + recall == 0:
        return clamp(0.05)

    f1 = 2 * precision * recall / (precision + recall)
    # Scale F1 into (0.05, 0.95) to avoid touching endpoints
    raw = 0.05 + f1 * 0.90
    return clamp(raw)


# ── Task 4: Reply Drafting ────────────────────────────────────────────────────

def grade_reply(
    reply: str,
    required_points: List[str],
    sender_name: str = "",
) -> float:
    """
    Score a drafted reply based on:
    - Coverage of required talking points (65%)
    - Professionalism markers (20%)
    - Appropriate length (10%)
    - Sender personalisation (5%)
    """
    if not reply or len(reply.strip()) < 10:
        return clamp(0.05)

    reply_lower = reply.lower()

    # Talking-point coverage
    if required_points:
        covered = 0
        for point in required_points:
            keywords = [kw.strip() for kw in point.split(",")]
            if any(kw in reply_lower for kw in keywords):
                covered += 1
        coverage = covered / len(required_points)
    else:
        coverage = 0.80  # No points required → moderate score

    # Professionalism markers
    has_greeting = any(g in reply_lower for g in ["dear", "hello", "hi ", "greetings", "good morning", "good afternoon"])
    has_closing = any(c in reply_lower for c in ["regards", "sincerely", "best", "thank you", "thanks", "cheers"])
    # Simple check: reply has multiple sentences
    has_structure = reply.count(".") >= 2 or reply.count("\n") >= 2
    professionalism = (has_greeting + has_closing + has_structure) / 3.0

    # Length score (ideal: 30-350 words)
    word_count = len(reply.split())
    if word_count < 10:
        length_score = 0.10
    elif word_count < 30:
        length_score = 0.50
    elif word_count <= 350:
        length_score = 0.95
    else:
        length_score = max(0.40, 0.95 - (word_count - 350) * 0.001)

    # Sender personalisation bonus
    first_name = sender_name.split()[0].lower() if sender_name and sender_name.split() else ""
    personal_bonus = 0.05 if first_name and first_name in reply_lower else 0.0

    raw = (
        coverage * 0.65
        + professionalism * 0.20
        + length_score * 0.10
        + personal_bonus
    )
    return clamp(0.05 + raw * 0.90)


# ── Task 5: Email Summarization ───────────────────────────────────────────────

def grade_summarization(
    summary: str,
    key_terms: List[str],
) -> float:
    """
    Score a summary based on:
    - Length compliance (10-60 words)  [40%]
    - Key-term coverage                [60%]
    """
    if not summary or len(summary.strip()) < 3:
        return clamp(0.05)

    words = summary.split()
    word_count = len(words)

    # Length scoring
    if word_count < 10:
        length_score = 0.10
    elif word_count <= 60:
        length_score = 0.95
    else:
        # Penalise excess verbosity
        excess = word_count - 60
        length_score = max(0.10, 0.95 - excess * 0.015)

    # Key-term coverage
    summary_lower = summary.lower()
    if key_terms:
        matched = sum(1 for t in key_terms if t.lower() in summary_lower)
        coverage = matched / len(key_terms)
    else:
        coverage = 0.70

    raw = length_score * 0.40 + coverage * 0.60
    return clamp(0.05 + raw * 0.90)


# ── Task 6: Thread Classification ────────────────────────────────────────────

def grade_thread_classification(
    predicted_category: str,
    key_issue: str,
    ground_truth_category: str,
    ground_truth_keywords: List[str],
) -> float:
    """
    Score thread classification:
    - Category match  [50%]
    - Key-issue extraction quality (keyword coverage)  [50%]
    """
    predicted_category = (predicted_category or "").strip().lower()
    ground_truth_category = (ground_truth_category or "").strip().lower()

    # Category score (same semantic-group logic as classification)
    if predicted_category == ground_truth_category:
        cat_score = 0.92
    else:
        cat_score = 0.05
        for group in _CATEGORY_GROUPS:
            if predicted_category in group and ground_truth_category in group:
                cat_score = 0.45
                break

    # Key-issue keyword coverage
    if key_issue and ground_truth_keywords:
        issue_lower = key_issue.lower()
        matched = sum(1 for kw in ground_truth_keywords if kw.lower() in issue_lower)
        kw_coverage = matched / len(ground_truth_keywords)
        issue_score = 0.05 + kw_coverage * 0.90
    elif key_issue:
        issue_score = 0.40  # provided but can't verify
    else:
        issue_score = 0.05

    raw = cat_score * 0.50 + issue_score * 0.50
    return clamp(raw)
