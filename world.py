from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Something Tommy can perceive nearby."""
    id: str
    type: str  # "car", "pedestrian", "police", "weapon", ...
    distance: str  # "close", "medium", "far"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TommyState(BaseModel):
    location: str = "Ocean Beach"
    cash: int = 500
    wanted_level: int = 0
    current_vehicle: Optional[str] = None
    health: int = 100
    armed: bool = True
    weapon: Optional[str] = "pistol"


class WorldState(BaseModel):
    tick: int = 0
    tommy: TommyState = Field(default_factory=TommyState)
    nearby_entities: List[Entity] = Field(default_factory=list)
    event_log: List[str] = Field(default_factory=list)
    last_decision: Optional[Dict[str, Any]] = None

    def log(self, message: str) -> None:
        self.event_log.append(f"[tick {self.tick}] {message}")