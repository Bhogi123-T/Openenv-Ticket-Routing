# OpenEnv — Ticket Routing Environment

> A real-world **customer support ticket routing** environment built to the [OpenEnv](https://openenv.dev) specification.  
> Agents must classify, prioritize, and escalate tickets across three difficulty tiers using context, sentiment, and customer tier signals.

---

## 📁 Project Structure

```
openenv-ticket-routing/
├── env/
│   ├── __init__.py          # Package marker
│   ├── models.py            # Pydantic schemas: Ticket, Observation, Action, Reward
│   ├── environment.py       # TicketEnv — OpenEnv-compliant step/reset/state loop
│   ├── tasks.py             # Task definitions with metadata per difficulty tier
│   └── grader.py            # Episode grader with detailed breakdown + agent comparison
├── inference/
│   └── run_baseline.py      # Heuristic + LLM (Gemini 1.5 Flash) baseline agents
├── openenv.yaml             # OpenEnv specification manifest
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container setup for HuggingFace Spaces / cloud
├── .env                     # API key configuration (not committed)
├── .gitignore
└── README.md
```

---

## ⚙️ OpenEnv Compliance

| Feature               | Status |
|-----------------------|--------|
| `reset()` → `Observation` | ✅ |
| `step(Action)` → `(Observation, Reward, done, info)` | ✅ |
| `state()` → dict | ✅ |
| Pydantic models for all schemas | ✅ |
| `openenv.yaml` manifest | ✅ |
| Three task tiers (easy / medium / hard) | ✅ |
| Programmatic grader | ✅ |

---

## 🧠 Observation Space (`Observation`)

| Field              | Type             | Description                              |
|--------------------|------------------|------------------------------------------|
| `ticket`           | `Ticket \| None` | Current ticket to route                  |
| `history`          | `List[str]`      | Log of all previous agent actions        |
| `remaining_tickets`| `int`            | How many tickets remain in episode       |

Each **`Ticket`** includes:
- `id`, `text`, `priority`, `category`
- `customer_tier` — `Standard`, `Premium`, or `VIP`
- `sentiment` — `Positive`, `Neutral`, `Negative`, or `Angry`

---

## 🎮 Action Space (`Action`)

| Field                | Type            | Description                                               |
|----------------------|-----------------|-----------------------------------------------------------|
| `assign_to`          | `str`           | `billing_team`, `tech_team`, `general_team`, `escalation_team` |
| `mark_priority`      | `str`           | `high`, `medium`, or `low`                                |
| `escalate_to_manager`| `bool`          | Whether to escalate immediately                           |
| `automated_response` | `str \| None`   | Optional canned reply to the customer                     |

---

## 🏆 Reward Design

Per step, reward is computed as:

| Criterion                                       | Points  |
|-------------------------------------------------|---------|
| Correct team assignment                         | +0.5    |
| Incorrect team assignment                       | −0.5    |
| Correct priority (with VIP override logic)      | +0.3    |
| Incorrect priority                              | −0.3    |
| Escalated angry VIP/Premium customer            | +0.2    |
| Missed escalation for angry VIP/Premium         | −0.2    |
| Unnecessary escalation                          | −0.2    |

Final score = `sum(rewards) / num_steps` (normalized to `[0.0, 1.0]`)

---

## 🎯 Task Tiers

### Easy (3 tickets)
Clear text signals, explicit categories, standard routing. Baseline: **1.0**

### Medium (4 tickets)
VIP priority overrides, escalation business rules must apply. Baseline: **1.0**

### Hard (4 tickets)
Ambiguous text, conflicting signals, context inference required. Baseline: **0.90**

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional — LLM agent)
```bash
# Edit .env:
GEMINI_API_KEY=your-gemini-api-key-here
```
> Without a key, the heuristic agent runs automatically.

### 3. Run baseline
```bash
python -m inference.run_baseline
# or from IDE:
python inference/run_baseline.py
```

---

## 🐳 Docker

```bash
docker build -t ticket-env .
docker run --env-file .env ticket-env
```

---

## 📊 Baseline Results

| Task   | Agent     | Score   |
|--------|-----------|---------|
| Easy   | Heuristic | 0.8667  |
| Medium | Heuristic | 0.9000  |
| Hard   | Heuristic | 0.9000  |
| Easy   | LLM (Gemini 1.5 Flash) | ~1.0 |
| Medium | LLM (Gemini 1.5 Flash) | ~1.0 |
| Hard   | LLM (Gemini 1.5 Flash) | ~0.95 |

---

## 🤖 Agents

### Heuristic Agent
Rule-based: matches keywords → routes to team, applies VIP/sentiment overrides, auto-escalates Angry VIP/Premium customers.

### LLM Agent (Gemini 1.5 Flash)  
Sends a structured prompt to Gemini 1.5 Flash, enforcing business rules via system instructions. Falls back to heuristic if API key is missing or call fails.

---

## 📄 License
MIT