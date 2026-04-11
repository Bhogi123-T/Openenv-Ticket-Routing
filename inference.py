import os
import json
import requests
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")           # No default — must be supplied

# Optional — used when deploying via from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI-compatible client ───────────────────────────────────────────────────
# All LLM calls use this client configured via the variables above
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-token",
)

# ── Environment server URL ─────────────────────────────────────────────────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(ticket: dict, obs: dict) -> str:
    return f"""You are an expert customer support ticket routing agent.

Current Ticket:
{json.dumps(ticket, indent=2)}

History so far: {obs.get('history', [])}
Remaining tickets after this one: {obs.get('remaining_tickets', 0)}

Routing rules:
- category "billing"  → assign_to: "billing_team"
- category "tech"     → assign_to: "tech_team"
- category "general"  → assign_to: "general_team"
- If VIP customer     → always mark_priority: "high"
- If Angry + (VIP or Premium) customer → escalate_to_manager: true
- Otherwise escalate_to_manager: false

Respond with ONLY valid JSON — no extra text:
{{
  "assign_to": "billing_team|tech_team|general_team|escalation_team",
  "mark_priority": "high|medium|low",
  "escalate_to_manager": true or false,
  "automated_response": "brief reply to the customer or null"
}}"""


# ── Heuristic fallback ─────────────────────────────────────────────────────────
def heuristic_action(ticket: dict) -> dict:
    category  = ticket.get("category", "general")
    tier      = ticket.get("customer_tier", "Standard")
    sentiment = ticket.get("sentiment", "Neutral")
    priority  = ticket.get("priority", "medium")

    team_map = {"billing": "billing_team", "tech": "tech_team"}
    team = team_map.get(category, "general_team")

    if tier == "VIP":
        priority = "high"

    escalate = sentiment == "Angry" and tier in ("VIP", "Premium")

    return {
        "assign_to": team,
        "mark_priority": priority,
        "escalate_to_manager": escalate,
        "automated_response": f"Thank you for reaching out. Your ticket has been assigned to our {team}.",
    }


# ── Parse LLM response safely ──────────────────────────────────────────────────
def parse_action(raw: str, ticket: dict) -> dict:
    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        action = json.loads(cleaned.strip())
        # Validate required keys exist
        assert "assign_to" in action and "mark_priority" in action
        return action
    except Exception:
        return heuristic_action(ticket)


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(task: str = "easy") -> None:
    print("START")

    # Reset environment
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    reset_resp.raise_for_status()
    data = reset_resp.json()
    obs  = data["observation"]
    done = data["done"]
    step_num = 0

    while not done:
        step_num += 1
        ticket = obs.get("ticket") or {}

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(ticket, obs)}],
                max_tokens=256,
                temperature=0.0,
            )
            raw    = completion.choices[0].message.content
            action = parse_action(raw, ticket)
        except Exception as llm_err:
            # Fall back to heuristic if LLM is unavailable
            print(f"STEP {step_num} | task={task} | llm_error={llm_err} | using heuristic fallback")
            action = heuristic_action(ticket)

        # ── Step environment ───────────────────────────────────────────────────
        step_resp = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        result = step_resp.json()

        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        info   = result.get("info", {})

        print(f"STEP {step_num} | task={task} | reward={reward} | action={json.dumps(action)} | reason={info.get('reward_reason', '')}")

    print("END")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_episode(task)
