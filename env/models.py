from pydantic import BaseModel, Field
from typing import List, Optional

class Ticket(BaseModel):
    id: int
    text: str
    priority: str
    category: str
    customer_tier: str = Field(default="Standard", description="Can be Standard, Premium, or VIP")
    sentiment: str = Field(default="Neutral", description="Customer sentiment: Positive, Neutral, Negative, or Angry")

class Observation(BaseModel):
    ticket: Optional[Ticket]
    history: List[str]
    remaining_tickets: int

class Action(BaseModel):
    assign_to: str = Field(description="Team to assign to: billing_team, tech_team, general_team, or escalation_team")
    mark_priority: str = Field(description="high, medium, or low")
    escalate_to_manager: bool = Field(default=False, description="Whether to escalate this immediately")
    automated_response: Optional[str] = Field(default=None, description="Automated reply to the customer")

class Reward(BaseModel):
    value: float
    reason: str