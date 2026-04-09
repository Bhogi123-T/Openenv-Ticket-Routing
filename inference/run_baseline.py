import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows PowerShell Unicode encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from dotenv import load_dotenv
load_dotenv()

from env.environment import TicketEnv
from env.models import Action
from env.grader import grade_episode, grade_episode_detailed, compare_agents

# ─────────────────────────────────────────
# HEURISTIC AGENT
# ─────────────────────────────────────────
def heuristic_agent(ticket):
    text = ticket.text.lower()

    if "payment" in text or "billing" in text or "cancel" in text or "invoice" in text or "refund" in text:
        assign_to = "billing_team"
    elif "crash" in text or "login" in text or "down" in text or "security" in text or "fraud" in text or "update" in text or "blocked" in text or "typo" in text or "bug" in text:
        assign_to = "tech_team"
    elif "manager" in text or "enterprise" in text or "ceo" in text or "licensing" in text:
        assign_to = "general_team"
    else:
        category_map = {
            "billing": "billing_team",
            "tech": "tech_team",
            "general": "general_team"
        }
        assign_to = category_map.get(ticket.category, "general_team")

    if ticket.customer_tier == "VIP":
        priority = "high"
    elif ticket.sentiment == "Angry":
        priority = "high"
    elif ticket.priority == "high":
        priority = "high"
    elif ticket.priority == "medium":
        priority = "medium"
    else:
        priority = "low"

    escalate = ticket.sentiment == "Angry" and ticket.customer_tier in ["VIP", "Premium"]

    return Action(
        assign_to=assign_to,
        mark_priority=priority,
        escalate_to_manager=escalate,
        automated_response="Thank you for reaching out, your ticket has been routed."
    )


# ─────────────────────────────────────────
# GEMINI LLM AGENT
# ─────────────────────────────────────────
def llm_agent(ticket):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  [INFO] GEMINI_API_KEY not found — using heuristic agent.")
        return heuristic_agent(ticket)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")

        prompt = f"""
You are an automated customer support ticket router.

Analyze the ticket below and return ONLY valid JSON (no markdown, no explanation) with exactly these keys:
{{
  "assign_to": one of ["billing_team", "tech_team", "general_team"],
  "mark_priority": one of ["high", "medium", "low"],
  "escalate_to_manager": true or false
}}

BUSINESS RULES (mandatory):
1. VIP customers → always mark_priority = "high"
2. Angry VIP or Premium customers → escalate_to_manager = true
3. Billing/payment/invoice/refund/cancel issues → assign_to = "billing_team"
4. Tech/crash/login/bug/down/security issues → assign_to = "tech_team"
5. All others → assign_to = "general_team"

Ticket:
  ID       : {ticket.id}
  Text     : {ticket.text}
  Category : {ticket.category}
  Priority : {ticket.priority}
  Sentiment: {ticket.sentiment}
  Tier     : {ticket.customer_tier}
"""

        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return Action(
            assign_to=data.get("assign_to", "general_team"),
            mark_priority=data.get("mark_priority", "low"),
            escalate_to_manager=data.get("escalate_to_manager", False),
            automated_response="Routed by Gemini LLM Agent."
        )

    except Exception as e:
        print(f"  [WARN] Gemini agent failed: {e} -- falling back to heuristic.")
        return heuristic_agent(ticket)


# ─────────────────────────────────────────
# MAIN RUN LOOP
# ─────────────────────────────────────────
def run():
    print("=" * 52)
    print("  OpenEnv -- Ticket Routing System")
    api_key = os.getenv("GEMINI_API_KEY")
    agent_type = "Gemini 1.5 Flash (LLM)" if api_key else "Heuristic (fallback)"
    print(f"  Agent   : {agent_type}")
    print("=" * 52)

    all_results = {"heuristic": {}, "llm": {}}

    for agent_key, current_agent in [("heuristic", heuristic_agent), ("llm", llm_agent)]:
        print(f"\n{'='*52}")
        print(f"  Evaluating Agent: {agent_key.upper()}")
        print(f"{'='*52}")

        for task_name in ["easy", "medium", "hard"]:
            env = TicketEnv(task_name=task_name)
            obs = env.reset()
            rewards = []

            while True:
                action = current_agent(obs.ticket)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break

            detail = grade_episode_detailed(rewards)
            all_results[agent_key][task_name] = detail

    print(compare_agents(all_results))


if __name__ == "__main__":
    run()