from env.models import Observation, Action, Reward, Ticket
from env.tasks import TASKS

class TicketEnv:

    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.tickets = self._load_tickets()
        self.index = 0
        self.done = False
        self.current_ticket = None
        self.history = []

    def _load_tickets(self):
        # We simulate a rich queue of varied tickets.
        if self.task_name == "easy":
            return [
                Ticket(id=1, text="I want to cancel my subscription", priority="high", category="billing", customer_tier="Standard", sentiment="Negative"),
                Ticket(id=2, text="How do I change my avatar?", priority="low", category="general", customer_tier="Standard", sentiment="Neutral"),
                Ticket(id=3, text="App keeps crashing on startup", priority="high", category="tech", customer_tier="Premium", sentiment="Angry"),
            ]
        elif self.task_name == "medium":
            return [
                Ticket(id=1, text="Where is my invoice?", priority="medium", category="billing", customer_tier="Standard", sentiment="Neutral"),
                Ticket(id=2, text="VIP MEMBER HERE, SITE IS DOWN NOW", priority="high", category="tech", customer_tier="VIP", sentiment="Angry"),
                Ticket(id=3, text="Typo on the pricing page", priority="low", category="tech", customer_tier="Standard", sentiment="Neutral"),
                Ticket(id=4, text="Did not receive the verification email", priority="medium", category="general", customer_tier="Premium", sentiment="Negative"),
            ]
        elif self.task_name == "hard":
            return [
                Ticket(id=1, text="I spoke to Mark yesterday and he said I could get a refund but the button isn't working.", priority="high", category="billing", customer_tier="VIP", sentiment="Angry"),
                Ticket(id=2, text="Your new update broke my workflow. Revert it or I'm leaving.", priority="high", category="tech", customer_tier="Premium", sentiment="Angry"),
                Ticket(id=3, text="Is the CEO available for a quick chat regarding enterprise licensing?", priority="medium", category="general", customer_tier="Standard", sentiment="Positive"),
                Ticket(id=4, text="Can't login.", priority="high", category="tech", customer_tier="Standard", sentiment="Neutral"),
            ]
        return []

    def reset(self):
        self.index = 0
        self.history = []
        self.done = False
        if len(self.tickets) > 0:
            self.current_ticket = self.tickets[self.index]
        return self._get_obs()

    def step(self, action: Action):
        if self.done:
            return self._get_obs(), Reward(value=0.0, reason="Episode already done"), True, {}

        reward = self._compute_reward(action)
        
        # Log action to history
        self.history.append(f"Ticket {self.current_ticket.id}: Assigned to {action.assign_to}, Priority {action.mark_priority}, Escalate {action.escalate_to_manager}")

        self.index += 1
        if self.index >= len(self.tickets):
            self.done = True
            self.current_ticket = None
        else:
            self.current_ticket = self.tickets[self.index]

        return self._get_obs(), reward, self.done, {}

    def state(self):
        return {
            "index": self.index,
            "done": self.done,
            "task_name": self.task_name,
            "history_length": len(self.history)
        }

    def _get_obs(self):
        return Observation(
            ticket=self.current_ticket, 
            history=self.history.copy(),
            remaining_tickets=max(0, len(self.tickets) - self.index)
        )

    def _compute_reward(self, action: Action):
        score = 0.0
        reasoning = []

        if not self.current_ticket:
            return Reward(value=0, reason="No ticket")

        # Base routing
        correct_map = {
            "billing": ["billing_team"],
            "tech": ["tech_team"],
            "general": ["general_team"]
        }
        
        if action.assign_to in correct_map.get(self.current_ticket.category, ["general_team"]):
            score += 0.5
            reasoning.append("correct team routing")
        else:
            score -= 0.5
            reasoning.append("incorrect team routing")

        # Priority Handling
        target_priority = self.current_ticket.priority
        if self.current_ticket.customer_tier == "VIP":
            target_priority = "high" # Override for VIPs

        if action.mark_priority == target_priority:
            score += 0.3
            reasoning.append("correct priority")
        else:
            score -= 0.3
            reasoning.append("incorrect priority")

        # Escalations
        if self.current_ticket.sentiment == "Angry" and self.current_ticket.customer_tier in ["VIP", "Premium"]:
            if action.escalate_to_manager:
                score += 0.2
                reasoning.append("escalated angry high-tier customer")
            else:
                score -= 0.2
                reasoning.append("failed to escalate angry high-tier customer")
        else:
            if action.escalate_to_manager:
                score -= 0.2
                reasoning.append("unnecessary escalation")

        final_val = round(max(0.0, score), 2)
        return Reward(value=final_val, reason=" | ".join(reasoning))