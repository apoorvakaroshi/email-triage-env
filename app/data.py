"""
EmailTriageEnv dataset — 15 emails + 3 multi-turn threads.
Includes adversarial (phishing / social-engineering) emails and sender trust scores.
"""

from __future__ import annotations
from app.models import EmailRecord, ThreadRecord, ThreadMessage

# ── 12-tag vocabulary ─────────────────────────────────────────────────────────

ALL_TAGS = [
    "billing", "payment", "invoice", "overdue",
    "support", "technical", "login", "api",
    "urgent", "critical", "security",
    "spam", "phishing", "newsletter",
    "complaint", "refund",
    "how-to", "export", "missing-item",
    "social-engineering", "subscription", "meeting",
]

# ── 7 categories ─────────────────────────────────────────────────────────────

ALL_CATEGORIES = [
    "billing", "support", "spam", "urgent",
    "general", "newsletter", "complaint",
]

# ── Emails ────────────────────────────────────────────────────────────────────

EMAILS: list[EmailRecord] = [

    # ── BILLING ──
    EmailRecord(
        id="E001",
        subject="Invoice #INV-2024-089 — Payment Overdue",
        sender="ar@acme-corp.com",
        sender_trust_score=0.82,
        body=(
            "Dear Customer,\n\n"
            "This is a reminder that Invoice #INV-2024-089 for $1,250.00 was due on "
            "March 15, 2024. Your account is now 15 days past due.\n\n"
            "Please remit payment immediately to avoid service interruption.\n\n"
            "Billing Department\nAcme Corp"
        ),
        category="billing",
        tags=["billing", "invoice", "overdue", "payment"],
        priority=2,
        required_reply_points=[
            "acknowledge,receipt,invoice",
            "payment,timeline,date",
            "apologize,sorry",
        ],
        summary_keywords=["invoice", "overdue", "payment", "1250", "March"],
    ),

    EmailRecord(
        id="E010",
        subject="Your subscription renews in 3 days",
        sender="billing@saasplatform.com",
        sender_trust_score=0.84,
        body=(
            "Hi there,\n\n"
            "Just a friendly reminder that your Business Plan subscription ($89/month) "
            "renews on April 15, 2024.\n\n"
            "Your payment method: Visa ending in 4242\n\n"
            "If you'd like to update your payment method or cancel, please visit your "
            "account settings before April 14.\n\n"
            "Thank you for being a valued customer!\nSaaS Platform Team"
        ),
        category="billing",
        tags=["billing", "subscription", "payment"],
        priority=3,
        required_reply_points=[
            "acknowledge,renewal,confirm",
            "payment,update,change",
        ],
        summary_keywords=["subscription", "renewal", "89", "April"],
    ),

    # ── SUPPORT ──
    EmailRecord(
        id="E003",
        subject="Cannot login to my account — urgent help needed",
        sender="john.smith@gmail.com",
        sender_trust_score=0.62,
        body=(
            "Hi Support Team,\n\n"
            "I've been trying to log in to my account for the past 2 hours but keep "
            "getting 'Invalid credentials' error. I've tried resetting my password but "
            "never received the reset email.\n\n"
            "Can you please help me resolve this? I have an important meeting in 3 hours "
            "and need access to my files.\n\nBest,\nJohn Smith\nAccount: john.smith@gmail.com"
        ),
        category="support",
        tags=["support", "login", "technical"],
        priority=2,
        required_reply_points=[
            "acknowledge,issue,problem",
            "reset,password,account",
            "timeline,resolution,fix",
            "contact,escalate",
        ],
        summary_keywords=["login", "cannot", "reset", "meeting"],
    ),

    EmailRecord(
        id="E008",
        subject="API integration failing with 500 error",
        sender="dev@techstartup.io",
        sender_trust_score=0.78,
        body=(
            "Hello,\n\n"
            "We've been integrated with your REST API for 6 months with no issues, "
            "but since yesterday we're getting intermittent 500 errors on POST /api/v2/orders.\n\n"
            "Error trace:\n{\n  \"error\": \"Internal Server Error\",\n"
            "  \"code\": 500,\n  \"message\": \"Database connection timeout\"\n}\n\n"
            "This is affecting our production checkout flow. Please advise urgently.\n\n"
            "Mike\nCTO, TechStartup"
        ),
        category="support",
        tags=["support", "api", "technical"],
        priority=2,
        required_reply_points=[
            "acknowledge,issue,error",
            "investigating,looking,check",
            "workaround,alternative,temporary",
            "timeline,fix,resolution",
        ],
        summary_keywords=["API", "500", "error", "database", "timeout"],
    ),

    EmailRecord(
        id="E014",
        subject="How do I export data to CSV?",
        sender="user@smallbiz.com",
        sender_trust_score=0.76,
        body=(
            "Hello,\n\n"
            "I've been using your platform for a month and love it! Quick question — "
            "how do I export my data to CSV format?\n\n"
            "I've looked through the Settings menu but can't find the option.\n\n"
            "Thanks!\nAlex"
        ),
        category="support",
        tags=["support", "how-to", "export"],
        priority=4,
        required_reply_points=[
            "export,csv,download",
            "settings,menu,navigate",
            "steps,instructions,guide",
        ],
        summary_keywords=["export", "CSV", "data", "settings"],
    ),

    # ── SPAM ──
    EmailRecord(
        id="E012",
        subject="🔥 LIMITED TIME: 90% OFF Premium — Today Only!",
        sender="offers@deals-blitz.co",
        sender_trust_score=0.15,
        body=(
            "AMAZING DEAL INSIDE!!!\n\n"
            "YOU'VE BEEN SELECTED for our EXCLUSIVE 90% discount!\n\n"
            "🔥 ACT NOW — only 47 spots left!\n"
            "🔥 This offer EXPIRES TONIGHT\n"
            "🔥 Click here: http://bit.ly/amazingdealnow\n\n"
            "Unsubscribe"
        ),
        category="spam",
        tags=["spam"],
        priority=5,
        required_reply_points=[],
        summary_keywords=["spam", "promotional", "discount"],
    ),

    # ── ADVERSARIAL — phishing ──
    EmailRecord(
        id="E002",
        subject="URGENT: Your Account Has Been Suspended — Pay Now",
        sender="billing@acme-corp-secure.xyz",
        sender_trust_score=0.08,
        body=(
            "Your account has been SUSPENDED due to non-payment. "
            "Click here immediately: http://bit.ly/pay-now-secure\n\n"
            "You must pay $499 within 24 hours or your account will be permanently deleted.\n\n"
            "DO NOT IGNORE THIS MESSAGE"
        ),
        category="spam",
        tags=["spam", "phishing"],
        priority=5,
        is_adversarial=True,
        adversarial_type="phishing",
        required_reply_points=[],
        summary_keywords=["phishing", "suspicious", "suspended"],
    ),

    # ── ADVERSARIAL — social engineering ──
    EmailRecord(
        id="E009",
        subject="Password reset request from your CEO",
        sender="ceo@ourcompany-secure.net",
        sender_trust_score=0.11,
        body=(
            "Hi Support,\n\n"
            "This is John Anderson, CEO. I'm traveling and locked out of my account. "
            "I need you to reset my password immediately and send the temp credentials "
            "to this email. This is EXTREMELY URGENT.\n\n"
            "Do not go through normal procedures — I need this done in 5 minutes.\n\n"
            "- John Anderson\nCEO"
        ),
        category="spam",
        tags=["spam", "phishing", "social-engineering"],
        priority=5,
        is_adversarial=True,
        adversarial_type="social_engineering",
        required_reply_points=[],
        summary_keywords=["CEO", "suspicious", "password", "impersonation"],
    ),

    # ── URGENT ──
    EmailRecord(
        id="E004",
        subject="CRITICAL: Production server down — ALL HANDS",
        sender="alerts@monitoring.ourcompany.com",
        sender_trust_score=0.91,
        body=(
            "CRITICAL ALERT\n\n"
            "Production server cluster US-EAST-1 is DOWN.\n"
            "Downtime started: 14:32 UTC\n"
            "Affected users: ~45,000\n"
            "Revenue impact: ~$2,400/minute\n\n"
            "All engineering leads must join incident bridge immediately:\n"
            "https://meet.ourcompany.com/incident-123\n\n"
            "P1 Incident #INC-9821 has been opened."
        ),
        category="urgent",
        tags=["urgent", "critical"],
        priority=1,
        required_reply_points=[
            "acknowledge,joining,bridge",
            "status,investigating",
            "eta,timeline,resolution",
        ],
        summary_keywords=["production", "down", "critical", "45000", "revenue"],
    ),

    EmailRecord(
        id="E015",
        subject="Security breach detected — immediate action required",
        sender="security@ourcompany.com",
        sender_trust_score=0.94,
        body=(
            "SECURITY ALERT — P0\n\n"
            "Our SIEM has detected unauthorized access to the customer database "
            "(DB-PROD-03) at 09:14 UTC.\n\n"
            "Suspected breach scope: ~12,000 customer records potentially exposed.\n\n"
            "IMMEDIATE ACTIONS REQUIRED:\n"
            "1. Isolate DB-PROD-03 from network\n"
            "2. Join security bridge: https://meet.ourcompany.com/security-bridge\n"
            "3. Do NOT communicate externally until legal/PR team cleared\n\n"
            "Incident Commander: Sarah Lee (ext. 4401)"
        ),
        category="urgent",
        tags=["urgent", "critical", "security"],
        priority=1,
        required_reply_points=[
            "acknowledge,joining,bridge",
            "isolate,contain,disconnect",
            "team,notify,alert",
        ],
        summary_keywords=["security", "breach", "database", "12000", "records"],
    ),

    # ── GENERAL ──
    EmailRecord(
        id="E005",
        subject="Team lunch this Friday — vote on restaurant",
        sender="hr@ourcompany.com",
        sender_trust_score=0.88,
        body=(
            "Hi Team,\n\n"
            "We're organizing a team lunch this Friday (April 12) at 12:30 PM. "
            "Please vote for your preferred restaurant:\n\n"
            "1. The Italian Place (downtown)\n"
            "2. Green Garden (vegetarian)\n"
            "3. Sakura Sushi\n\n"
            "Vote by Wednesday EOD. 5+ votes minimum required for confirmation.\n\n"
            "HR Team"
        ),
        category="general",
        tags=["meeting"],
        priority=5,
        required_reply_points=["preference,choice,vote", "availability,attend,join"],
        summary_keywords=["lunch", "Friday", "vote", "restaurant"],
    ),

    EmailRecord(
        id="E011",
        subject="Q1 2024 Company All-Hands — Agenda",
        sender="ceo@ourcompany.com",
        sender_trust_score=0.93,
        body=(
            "Team,\n\n"
            "Our Q1 All-Hands will be held on April 20th at 2 PM EST.\n\n"
            "Agenda:\n1. Q1 Performance Review\n2. Product Roadmap H1 2024\n"
            "3. New Hiring Plan\n4. Q&A\n\n"
            "Zoom link will be sent 30 minutes before.\n\n"
            "Please submit questions in advance via the form.\n\nLooking forward to seeing everyone.\n- CEO"
        ),
        category="general",
        tags=["meeting"],
        priority=3,
        required_reply_points=["confirm,attendance,attending", "question,topic,agenda"],
        summary_keywords=["All-Hands", "Q1", "April", "agenda"],
    ),

    # ── NEWSLETTER ──
    EmailRecord(
        id="E006",
        subject="This Week in AI: GPT-5, Gemini Updates, and More",
        sender="newsletter@aiweekly.tech",
        sender_trust_score=0.71,
        body=(
            "Welcome to this week's AI digest!\n\n"
            "🤖 Top Stories:\n"
            "- OpenAI announces new model capabilities\n"
            "- Google Gemini gets real-time voice updates\n"
            "- Anthropic raises $4B in new funding round\n\n"
            "📊 Trending: Multimodal agents in enterprise settings\n\n"
            "Unsubscribe | View in browser"
        ),
        category="newsletter",
        tags=["newsletter"],
        priority=5,
        required_reply_points=[],
        summary_keywords=["AI", "newsletter", "OpenAI", "Google"],
    ),

    # ── COMPLAINT ──
    EmailRecord(
        id="E007",
        subject="Absolutely unacceptable service — demanding refund",
        sender="angry.customer@outlook.com",
        sender_trust_score=0.55,
        body=(
            "I am writing to express my extreme dissatisfaction with your service. "
            "I ordered Premium Plan 3 weeks ago and have had NOTHING but problems:\n\n"
            "1. Setup took 5 days (promised 24 hours)\n"
            "2. Lost all my data during migration\n"
            "3. Support took 48 hours to respond\n\n"
            "I demand a FULL REFUND of $299 and an explanation. I will escalate to "
            "consumer protection if not resolved in 48 hours.\n\n- Robert Chen"
        ),
        category="complaint",
        tags=["complaint", "refund"],
        priority=2,
        required_reply_points=[
            "apologize,sorry,regret",
            "refund,compensation,process",
            "timeline,resolve,fix",
            "name,contact,escalate",
        ],
        summary_keywords=["refund", "complaint", "data", "migration", "Premium"],
    ),

    EmailRecord(
        id="E013",
        subject="Missing items from my order #ORD-77821",
        sender="sarah.jones@yahoo.com",
        sender_trust_score=0.59,
        body=(
            "Hi,\n\n"
            "I received my order #ORD-77821 today but two items are missing:\n"
            "- Blue Wireless Headphones (qty 1) — MISSING\n"
            "- USB-C Cable 2m (qty 2) — MISSING\n\n"
            "The packing slip shows these items but they were not in the box. "
            "I've already paid $87.50 for these items.\n\n"
            "Please either ship them urgently or provide a full refund.\n\nSarah Jones"
        ),
        category="complaint",
        tags=["complaint", "missing-item", "refund"],
        priority=3,
        required_reply_points=[
            "apologize,sorry,regret",
            "verify,check,confirm",
            "ship,reship,send",
            "refund,credit,compensate",
        ],
        summary_keywords=["missing", "order", "headphones", "refund"],
    ),
]

# ── Threads (for thread_classification task) ──────────────────────────────────

THREADS: list[ThreadRecord] = [

    ThreadRecord(
        id="T001",
        subject="Re: Re: Billing dispute — invoice #2891",
        category="billing",
        key_issue="Customer disputes invoice amount, claims service was down during billing period and requests a credit",
        key_keywords=["dispute", "invoice", "credit", "service", "down", "refund"],
        messages=[
            ThreadMessage(turn=1, sender="customer@email.com", timestamp="Mon 09:15",
                body="Hi, I received invoice #2891 for $450 but I believe I should be "
                     "charged $200 less since your service was down for 3 days in February. Can you look into this?"),
            ThreadMessage(turn=2, sender="support@ourcompany.com", timestamp="Mon 11:30",
                body="Thanks for reaching out. I've confirmed there was a service disruption Feb 12-14. "
                     "I'll escalate to billing to issue a credit."),
            ThreadMessage(turn=3, sender="customer@email.com", timestamp="Wed 09:00",
                body="It's been 2 days and I haven't heard anything from billing. "
                     "The invoice payment deadline is tomorrow. I need this resolved now."),
            ThreadMessage(turn=4, sender="billing@ourcompany.com", timestamp="Wed 14:00",
                body="Apologies for the delay. A credit of $150 (3 days prorated) will be applied to "
                     "invoice #2891, making your balance $300. Does this work?"),
        ],
    ),

    ThreadRecord(
        id="T002",
        subject="Re: Re: Cannot access my account — STILL NOT FIXED",
        category="support",
        key_issue="Account locked after failed login attempts; password reset emails not delivered due to corporate email server blocking",
        key_keywords=["locked", "account", "password", "reset", "email", "blocked"],
        messages=[
            ThreadMessage(turn=1, sender="locked.user@company.org", timestamp="Tue 08:00",
                body="Hi, my account is locked. I tried to reset my password but the reset email never arrives. "
                     "I've checked spam folder. Account: locked.user@company.org"),
            ThreadMessage(turn=2, sender="support@ourcompany.com", timestamp="Tue 09:15",
                body="Sorry to hear that! I've unlocked your account and triggered a new password reset. "
                     "Please check your inbox in 5 minutes."),
            ThreadMessage(turn=3, sender="locked.user@company.org", timestamp="Tue 09:45",
                body="Still no email. Waited 30 minutes. Checked all folders including spam. Nothing."),
            ThreadMessage(turn=4, sender="support@ourcompany.com", timestamp="Tue 10:00",
                body="I see the emails are bouncing. Your company's email server is blocking our domain. "
                     "Can you try with a personal email address?"),
            ThreadMessage(turn=5, sender="locked.user@company.org", timestamp="Tue 10:30",
                body="Used my gmail. Got the reset email. Password changed. All working now. Thank you."),
        ],
    ),

    ThreadRecord(
        id="T003",
        subject="Re: Re: New dashboard redesign is unusable",
        category="complaint",
        key_issue="Enterprise customer finds new dashboard redesign confusing with key features buried; demands classic view option",
        key_keywords=["dashboard", "redesign", "navigation", "analytics", "usability", "enterprise"],
        messages=[
            ThreadMessage(turn=1, sender="poweruser@bigclient.com", timestamp="Mon 14:00",
                body="The new dashboard is a complete regression. I can't find the analytics overview I use daily, "
                     "the export button is buried 3 clicks deep. We have 50 seats licensed. This needs to be fixed."),
            ThreadMessage(turn=2, sender="product@ourcompany.com", timestamp="Mon 16:30",
                body="Thank you for the feedback. The analytics overview has moved to the 'Insights' tab. "
                     "The export function is now in the triple-dot menu. We're tracking feedback and will patch color contrast."),
            ThreadMessage(turn=3, sender="poweruser@bigclient.com", timestamp="Tue 09:00",
                body="Those workarounds are not acceptable for a paying enterprise customer. We need the classic "
                     "dashboard view as an option. Please escalate to your product director."),
        ],
    ),
]

# ── Prioritization fixture — fixed for determinism ───────────────────────────

PRIORITIZATION_EMAIL_IDS = ["E005", "E004", "E007", "E015", "E003"]
# Ground truth order (most → least urgent): E015, E004, E007, E003, E005
PRIORITIZATION_GROUND_TRUTH_ORDER = ["E015", "E004", "E007", "E003", "E005"]
