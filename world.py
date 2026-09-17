from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Something Tommy can perceive nearby."""
    id: str
    type: str  # "car", "pedestrian", "police", "weapon", ...
    distance: str  # "close", "medium", "far"
    attributes: dict[str, Any] = Field(default_factory=dict)


class TommyState(BaseModel):
    location: str = "Ocean Beach"
    cash: int = 500
    wanted_level: int = 0
    current_vehicle: str | None = None
    health: int = 100
    armed: bool = True
    weapon: str | None = "pistol"


class WorldState(BaseModel):
    tick: int = 0
    tommy: TommyState = Field(default_factory=TommyState)
    nearby_entities: list[Entity] = Field(default_factory=list)
    event_log: list[str] = Field(default_factory=list)
    last_decision: dict[str, Any] | None = None

    def log(self, message: str) -> None:
        self.event_log.append(f"[tick {self.tick}] {message}")
