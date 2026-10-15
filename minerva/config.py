"""Settings and camera config, loaded from YAML and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Point = tuple[float, float]  # normalised [0, 1] image coordinates


class DetectorCfg(BaseModel):
    weights: str = "yolo11n.pt"
    pose_weights: str = "yolo11n-pose.pt"
    backend: Literal["torch", "openvino"] = "torch"
    imgsz: int = 640
    conf: float = 0.25
    min_person_conf: float = 0.4
    min_person_height_px: float = 40


class TrackerCfg(BaseModel):
    activation_threshold: float = 0.25
    lost_track_buffer: int = 60  # frames; 15 s at 4 fps
    matching_threshold: float = 0.8


class VlmCfg(BaseModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:3b"
    num_ctx: int = 4096
    timeout_s: float = 300
    keep_alive: str = "30m"
    tile_px: int = 448
    max_queue_per_camera: int = 8


class ChatCfg(BaseModel):
    model: str | None = None  # None = reuse the VLM model
    timezone: str | None = None  # IANA name; None = system local time


class FusionCfg(BaseModel):
    confirmed_base: float = 0.6
    confirmed_gain: float = 0.4
    disputed_gain: float = 0.4
    rejected_gain: float = 0.2


class PersonDownCfg(BaseModel):
    enabled: bool = True
    upright_ratio: float = 1.4  # h/w at or above = upright
    upright_window_s: float = 3.0
    lying_ratio: float = 1.2  # w/h at or above = horizontal (fallback when pose is unsure)
    torso_angle_deg: float = 60.0
    min_pose_conf: float = 0.5
    sustain_s: float = 3.0
    max_drift_frac: float = 0.3  # centroid drift allowed, as a fraction of box width


class FightingCfg(BaseModel):
    enabled: bool = True
    proximity_frac: float = 0.8  # centre distance < this x mean box height
    motion_ratio: float = 3.0
    min_erratic_tracks: int = 1
    erratic_window_s: float = 2.0
    sustain_s: float = 1.5


class TailgatingCfg(BaseModel):
    enabled: bool = True
    max_gap_s: float = 2.0


class CrowdCfg(BaseModel):
    enabled: bool = True
    sustain_s: float = 10.0


class LoiteringCfg(BaseModel):
    enabled: bool = True
    min_dwell_s: float = 60.0
    displacement_window_s: float = 30.0
    max_displacement_heights: float = 2.0


class AbandonedCfg(BaseModel):
    enabled: bool = True
    labels: list[str] = ["backpack", "handbag", "suitcase"]
    stationary_iou: float = 0.7
    unattended_s: float = 30.0
    radius_person_heights: float = 1.5


class TriggersCfg(BaseModel):
    cooldown_s: float = 30.0
    release_s: float = 1.0  # condition may drop out this long without resetting an episode
    person_down: PersonDownCfg = PersonDownCfg()
    fighting: FightingCfg = FightingCfg()
    tailgating: TailgatingCfg = TailgatingCfg()
    crowd: CrowdCfg = CrowdCfg()
    loitering: LoiteringCfg = LoiteringCfg()
    abandoned_object: AbandonedCfg = AbandonedCfg()


class Settings(BaseModel):
    device: str = "auto"  # auto | cpu | cuda:0 | mps
    sample_fps: float = 4.0
    ring_buffer_s: float = 12.0
    ring_buffer_max_width: int = 960
    data_dir: Path = Path("data")
    detector: DetectorCfg = DetectorCfg()
    tracker: TrackerCfg = TrackerCfg()
    vlm: VlmCfg = VlmCfg()
    chat: ChatCfg = ChatCfg()
    fusion: FusionCfg = FusionCfg()
    triggers: TriggersCfg = TriggersCfg()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "minerva.db"


class EntryLine(BaseModel):
    p1: Point
    p2: Point
    # Which side of p1->p2 counts as "inside". Flip it if entries are detected as exits.
    entering: Literal["positive", "negative"] = "positive"


class CrowdZone(BaseModel):
    polygon: list[Point] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    max_people: int = 10
    name: str = "crowd_zone"


class Zones(BaseModel):
    entry_line: EntryLine | None = None
    crowd_zone: CrowdZone | None = None
    watch_zones: dict[str, list[Point]] = Field(default_factory=dict)


class CameraCfg(BaseModel):
    camera_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source: str  # rtsp://..., http://..., a file path, or a webcam index like "0"
    realtime: bool = True  # pace file sources at native speed (simulated live)
    start_time: str | None = None  # ISO time a file source starts at; None = now
    zones: Zones = Zones()

    @field_validator("source", mode="before")
    @classmethod
    def _source_str(cls, v):
        return str(v)


class CamerasFile(BaseModel):
    cameras: list[CameraCfg]


def load_settings(path: str | Path | None) -> Settings:
    if path is None or not Path(path).exists():
        return Settings()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)


def load_cameras(path: str | Path) -> list[CameraCfg]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CamerasFile.model_validate(data).cameras


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
