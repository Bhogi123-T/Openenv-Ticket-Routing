from typing import List, Dict, Any
from env.models import Reward


def grade_episode(rewards: List[Reward]) -> float:
    """Compute normalized score over a full episode."""
    if not rewards:
        return 0.0
    total = sum(r.value for r in rewards)
    max_score = len(rewards)
    return round(total / max_score, 4)


def grade_episode_detailed(rewards: List[Reward]) -> Dict[str, Any]:
    """Return a detailed breakdown of episode performance."""
    if not rewards:
        return {
            "normalized_score": 0.0,
            "total_reward": 0.0,
            "num_steps": 0,
            "max_possible": 0.0,
            "perfect_steps": 0,
            "failed_steps": 0,
            "reasons": []
        }

    total = sum(r.value for r in rewards)
    max_possible = len(rewards)  # each step has max reward of 1.0
    perfect_steps = sum(1 for r in rewards if r.value >= 0.99)
    failed_steps = sum(1 for r in rewards if r.value == 0.0)

    return {
        "normalized_score": round(total / max_possible, 4),
        "total_reward": round(total, 4),
        "num_steps": len(rewards),
        "max_possible": float(max_possible),
        "perfect_steps": perfect_steps,
        "failed_steps": failed_steps,
        "reasons": [r.reason for r in rewards]
    }


def compare_agents(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Compare multiple agents' graded results.
    `results` is a dict of {agent_name: {task_name: grade_episode_detailed output}}
    Returns a formatted comparison string.
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"{'AGENT COMPARISON REPORT':^60}")
    lines.append(f"{'='*60}")

    agents = list(results.keys())
    tasks = list(next(iter(results.values())).keys()) if agents else []

    for task in tasks:
        lines.append(f"\n  Task: {task.upper()}")
        lines.append(f"  {'Agent':<20} | {'Score':>6} | {'Perfect':>7} | {'Failed':>6}")
        lines.append(f"  {'-'*50}")
        for agent_name in agents:
            data = results[agent_name].get(task, {})
            score = data.get("normalized_score", 0.0)
            perfect = data.get("perfect_steps", 0)
            failed = data.get("failed_steps", 0)
            lines.append(f"  {agent_name:<20} | {score:>6.4f} | {perfect:>7} | {failed:>6}")

    lines.append(f"\n{'='*60}\n")
    return "\n".join(lines)