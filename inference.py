from flask import Flask, request, jsonify
from env.environment import TicketEnv
from env.models import Action

app = Flask(__name__)

# Global environment instance — default task is "easy"
env = TicketEnv(task_name="easy")


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the environment. Optionally accepts {"task": "easy"|"medium"|"hard"}."""
    global env
    data = request.get_json(silent=True) or {}
    task = data.get("task", "easy")

    # Validate task name
    if task not in ("easy", "medium", "hard"):
        task = "easy"

    env = TicketEnv(task_name=task)
    obs = env.reset()

    return jsonify({
        "observation": obs.model_dump(),
        "reward": 0,
        "done": False,
        "info": {"task": task}
    })


@app.route("/step", methods=["POST"])
def step():
    """Advance one step. Accepts Action JSON body."""
    global env
    data = request.get_json(silent=True) or {}

    # Build Action from request body; use safe defaults if missing fields
    try:
        action = Action(
            assign_to=data.get("assign_to", "general_team"),
            mark_priority=data.get("mark_priority", "medium"),
            escalate_to_manager=bool(data.get("escalate_to_manager", False)),
            automated_response=data.get("automated_response", None),
        )
    except Exception as e:
        return jsonify({"error": f"Invalid action: {e}"}), 400

    obs, reward, done, info = env.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": {**info, "reward_reason": reward.reason}
    })


@app.route("/state", methods=["GET"])
def state():
    """Return current environment state."""
    return jsonify(env.state())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
