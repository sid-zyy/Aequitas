"""Core value types shared by every stage. Kept free of heavy imports so tests stay fast."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class Category(str, Enum):
    PERSON_DOWN = "person_down"
    FIGHTING = "fighting"
    TAILGATING = "tailgating"
    CROWD_INCIDENT = "crowd_incident"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class Status(str, Enum):
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    REJECTED = "rejected"


CATEGORY_DEFINITIONS: dict[Category, str] = {
    Category.PERSON_DOWN: "a person has fallen or is lying on the ground and stays down",
    Category.FIGHTING: "two or more people are physically fighting (hitting, pushing, grappling)",
    Category.TAILGATING: "two or more people pass through an entrance one right behind the other",
    Category.CROWD_INCIDENT: "an area is overcrowded, or a crowd suddenly scatters",
    Category.SUSPICIOUS_ACTIVITY: "a person lingers in one place for a long time, or a bag is left unattended",
}


@dataclass(frozen=True)
class TrackObs:
    """One tracked object in one frame."""

    track_id: int
    label: str
    conf: float
    box: Box

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def feet(self) -> tuple[float, float]:
        """Bottom-centre of the box: the point used for zone membership and line crossing."""
        return ((self.box[0] + self.box[2]) / 2, self.box[3])


@dataclass
class Frame:
    camera_id: str
    index: int
    ts_ms: int  # stream time, UTC epoch ms
    wall_ns: int  # time.perf_counter_ns() when captured, for latency
    image: np.ndarray  # BGR


@dataclass(frozen=True)
class PoseResult:
    torso_angle_deg: float  # 0 = upright, 90 = horizontal
    conf: float


@dataclass
class Candidate:
    """A trigger firing, before VLM verification."""

    camera_id: str
    category: Category
    subtype: str | None
    onset_ms: int
    trigger_ms: int
    score: float
    track_ids: tuple[int, ...]
    zone: str | None = None
    key: tuple = ()  # rule-level identity, used for cooldown and end_ts extension
    detail: dict = field(default_factory=dict)
