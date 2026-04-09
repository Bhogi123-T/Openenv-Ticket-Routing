from typing import List, Dict, Any

TASKS: List[Dict[str, Any]] = [
    {
        "name": "easy",
        "description": (
            "Route straightforward tickets to the correct team. "
            "Each ticket has a clear category (billing, tech, general), "
            "explicit customer sentiment, and unambiguous priority. "
            "Agent must correctly assign team and set priority."
        ),
        "difficulty": "easy",
        "num_tickets": 3,
        "routing_complexity": "single-factor",
        "metrics": ["routing_accuracy", "priority_accuracy"],
        "reward_breakdown": {
            "correct_team": 0.5,
            "correct_priority": 0.3,
            "correct_escalation": 0.2
        }
    },
    {
        "name": "medium",
        "description": (
            "Route a mixed queue of tickets where customer tier impacts priority rules. "
            "VIP customers must always receive high priority. "
            "Premium + Angry customers must be escalated to a manager. "
            "Agent must apply business rules on top of category-routing."
        ),
        "difficulty": "medium",
        "num_tickets": 4,
        "routing_complexity": "multi-factor",
        "metrics": ["routing_accuracy", "priority_accuracy", "escalation_accuracy"],
        "reward_breakdown": {
            "correct_team": 0.5,
            "correct_priority": 0.3,
            "correct_escalation": 0.2
        }
    },
    {
        "name": "hard",
        "description": (
            "Handle ambiguous, emotionally charged tickets where the text alone "
            "is insufficient to determine routing. Agent must infer intent from "
            "context, customer tier, sentiment, and partial information. "
            "4 tickets with overlapping signals and edge cases (e.g., VIP complaining "
            "about a non-tech issue, or enterprise licensing requests misrouted as general). "
            "Escalation decisions carry higher weight at this difficulty."
        ),
        "difficulty": "hard",
        "num_tickets": 4,
        "routing_complexity": "context-sensitive",
        "metrics": ["routing_accuracy", "priority_accuracy", "escalation_accuracy", "context_sensitivity"],
        "reward_breakdown": {
            "correct_team": 0.5,
            "correct_priority": 0.3,
            "correct_escalation": 0.2
        }
    }
]

def get_task(name: str) -> Dict[str, Any]:
    """Retrieve a task configuration by name."""
    for task in TASKS:
        if task["name"] == name:
            return task
    raise ValueError(f"Unknown task name: '{name}'. Choose from: {[t['name'] for t in TASKS]}")