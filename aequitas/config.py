"""Settings and camera config, loaded from YAML and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from aequitas.authenticity import AuthenticityCfg
from aequitas.taxonomy import DEFAULT_CATEGORIES, validate

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


Backend = Literal["ollama", "openai"]  # openai = any OpenAI-compatible LOCAL server (llama.cpp, vLLM, LM Studio)


class LlmEndpoint(BaseModel):
    backend: Backend = "ollama"
    host: str = "http://127.0.0.1:11434"
    model: str = "gemma4:e2b"
    api_key_env: str | None = None  # env var holding a key, for local servers that require one
    allow_remote: bool = False  # refuse non-local hosts so no cloud API is used by accident
    # Ollama 'think' flag. False keeps reasoning models (Gemma 4, Qwen3) from spending time on hidden
    # reasoning; harmless for others. None = omit the flag (for servers that reject it).
    think: bool | None = False
    num_ctx: int = 4096
    timeout_s: float = 300
    keep_alive: str = "30m"
    # OpenAI-compatible backend: classify with vLLM structured outputs (`choice`) + logprobs. Turn off
    # for servers without it (they then get a JSON-schema enum and no confidence).
    choice_decoding: bool = True
    # Output caps. Decode time grows with every output token, so the summary is kept short.
    max_tokens: int = 384  # any structured call (chat filter, grounded answer)
    summary_max_tokens: int = 128  # stage 1 JSON: <=35-word summary + <=8-word evidence
    concise_summary: bool = True  # False = the original longer prompt (only to measure the time saved)
    guided_checks: bool = False  # stage 1 also answers grouped check questions (ASK-Hint style)


class VlmCfg(LlmEndpoint):
    tile_px: int = 448
    max_queue_per_camera: int = 8
    # Worker threads sending requests at once. 1 suits Ollama; vLLM batches concurrent requests,
    # so 4-16 cuts total time there.
    concurrency: int = Field(default=1, ge=1, le=64)


class EndpointOverride(BaseModel):
    """Another model for one stage. Any field left null falls back to the VLM endpoint, so by
    default one model serves every stage."""

    backend: Backend | None = None
    host: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    think: bool | None = None

    def endpoint(self, vlm: VlmCfg) -> LlmEndpoint:
        base = vlm.model_dump(include=set(LlmEndpoint.model_fields))
        own = {k: v for k, v in self.model_dump(include={"backend", "host", "model", "api_key_env", "think"}).items()
               if v is not None}
        return LlmEndpoint(**{**base, **own})

    @property
    def overridden(self) -> bool:
        return any(v is not None for v in (self.backend, self.host, self.model, self.api_key_env, self.think))


class EscalationCfg(EndpointOverride):
    """Cascade: a bigger model re-checks windows the fast model flags (model=None: off). Falls back to the
    VLM endpoint for anything left unset, so `model: gemma4:e4b-it-qat` alone is enough with Ollama."""

    violence_band_low: float = 0.35  # also re-check when the clip model scores at least this


class ChatCfg(EndpointOverride):
    summarize: bool = True  # add an LLM-written summary that may only cite retrieved event IDs
    timezone: str | None = None  # IANA name; None = system local time


class AnalysisCfg(BaseModel):
    # triggered: YOLO + rules decide when the VLM runs (default, cheap).
    # periodic: no YOLO; the VLM reviews every window of sampled frames.
    mode: Literal["triggered", "periodic"] = "triggered"
    window_s: float = 4.0
    confirm_votes: int = 2
    max_gap_s: float = 10.0  # windows further apart than this never merge into one event
    # Latency: skip the VLM for windows where almost nothing moves (mean frame difference, 0-1), e.g. an
    # empty corridor. 0 = off. Even when skipping, the VLM looks at least every max_skip_s.
    skip_static_below: float = 0.0
    max_skip_s: float = 30.0
    # Stage 2 also sees the previous window's summary from the same camera (sequences such as an arrest).
    # Measured (scripts/bench_variants.py, 72 windows): arrest 7->8/12, fire 6->8/10, 0 false alarms, +0.03 s.
    label_context: bool = True
    # A window where the violence clip model reaches models.violence_alone_threshold also votes "fighting"
    # towards confirmation, whatever label the VLM gave. Replay of the six saved demo runs: +1 to +3 confirmed
    # incidents per run, no new false alarm (results/improvements/README.md, scripts/replay_confirm.py; chosen on those clips).
    clip_votes: bool = True
    # Coarse stage: a SigLIP scene gate skips the VLM on windows that clearly look normal (aequitas/understand/gate.py).
    # None = off. Example: google/siglip-base-patch16-224 (Apache-2.0, ~0.8 GB).
    gate_model: str | None = None
    gate_below: float = 0.3


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
    # Overhead cameras: a person lying along the viewing direction still looks "upright" in the image,
    # but their box height collapses. Down if height <= this fraction of their recent upright height.
    height_drop_frac: float = 0.6
    # Trackers often switch ID during a fall; a new track counts as "was upright" if someone stood
    # upright within this many person-heights of the same spot, within upright_window_s.
    spatial_match_heights: float = 1.0


class ModelsCfg(BaseModel):
    """Optional trained models (kaggle/ notebooks). Unset = not used."""

    violence_dir: Path | None = None  # folder with violence_clip.onnx + metadata.json
    violence_stride_s: float = 0.5
    violence_threshold: float | None = None  # None = the calibrated threshold from metadata.json
    # The clip model alone (VLM saw nothing) raises "fighting" only at this score. Chosen on our demo street
    # clips (2 of 123 normal windows >= 0.9, every faculty clip has windows >= 0.95): a demo setting, not accuracy.
    violence_alone_threshold: float | None = 0.9
    hazard_dir: Path | None = None  # folder with hazard_detector.pt/.onnx + metadata.json
    hazard_conf: float = 0.35


class FightingCfg(BaseModel):
    enabled: bool = True
    proximity_frac: float = 0.8  # centre distance < this x mean box height
    motion_ratio: float = 3.0
    min_erratic_tracks: int = 1
    erratic_window_s: float = 2.0
    sustain_s: float = 1.5
    # With a violence clip model loaded, a sustained score at/above its threshold also counts,
    # with no need for two people to be tracked close together.
    use_clip_model: bool = True
    min_persons_for_clip: int = 1


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


class AuthCfg(BaseModel):
    required: bool = True  # police endpoints need a signed-in account; citizen endpoints never do
    session_hours: float = 12
    cookie_secure: bool = False  # True when served over HTTPS
    max_failures: int = 5
    lockout_s: float = 900
    # Fresh clone = empty database = nobody can sign in. Create admin/admin when there are no users at all
    # (flagged in the audit log). Demo only: the server listens on 127.0.0.1; set false for real use.
    demo_admin: bool = True


class CitizenCfg(BaseModel):
    max_upload_mb: int = 500
    reports_per_hour_per_ip: int = 5


class LiveCfg(BaseModel):
    localities_file: Path = Path("config/bangalore.yaml")  # localities and camera points (demo footage)


class Settings(BaseModel):
    # Stage 2 label set: name -> one-line definition shown to the classifier (aequitas/taxonomy.py).
    categories: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_CATEGORIES))
    device: str = "auto"  # auto | cpu | cuda:0 | mps
    sample_fps: float = 4.0
    ring_buffer_s: float = 12.0
    ring_buffer_max_width: int = 960
    data_dir: Path = Path("data")
    analysis: AnalysisCfg = AnalysisCfg()
    models: ModelsCfg = ModelsCfg()
    detector: DetectorCfg = DetectorCfg()
    tracker: TrackerCfg = TrackerCfg()
    vlm: VlmCfg = VlmCfg()
    classifier: EndpointOverride = EndpointOverride()  # stage 2, text only
    escalation: EscalationCfg = EscalationCfg()  # cascade re-check by a bigger model
    chat: ChatCfg = ChatCfg()
    fusion: FusionCfg = FusionCfg()
    triggers: TriggersCfg = TriggersCfg()
    auth: AuthCfg = AuthCfg()
    citizen: CitizenCfg = CitizenCfg()
    authenticity: AuthenticityCfg = AuthenticityCfg()
    live: LiveCfg = LiveCfg()

    @field_validator("categories")
    @classmethod
    def _categories(cls, v):
        return validate(v)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "aequitas.db"


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
    loop: bool = False  # replay a file source forever (demo footage standing in for a live camera)
    view: Literal["cctv", "close"] = "cctv"  # close: a webcam near the people; gestures described in detail
    window_s: float | None = None  # periodic window for this camera (None = analysis.window_s)
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
