"""Skeleton action recognition for close-range cameras (the live webcam, bodycam-style views).

YOLO pose gives 17 COCO keypoints per person per frame; STGCN++ (Duan et al., PYSKL 2022) classifies the last
few seconds of keypoints into the 120 NTU RGB+D 120 actions. NTU is filmed indoors at close range, and its
classes separate what the VLM confuses on the webcam: hand waving, thumb up, sitting down vs punching, pushing,
falling. Weights: PYSKL `stgcnpp_ntu120_xsub_hrnet` joint (j.pth, 84.4 % top-1) and bone (b.pth, 84.8 %)
streams, trained on HRNet 2D keypoints (COCO-17, the same layout as YOLO pose). NTU RGB+D is released for
non-commercial research.

This is a plain-PyTorch port of the inference path only (no mmcv): the module names match the checkpoints, so
`load_state_dict(strict=True)` checks the port. Preprocessing follows the PYSKL test pipeline: PreNormalize2D,
GenSkeFeat, 100 frames, 2 persons. Two differences: our frames come at ~4 fps, so the window is resampled to
100 frames by linear interpolation (NTU: 30 fps, uniformly sampled); and a 4:3 webcam frame is placed on a
16:9 canvas, the shape NTU's 1920x1080 keypoints were normalised with.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

NTU120 = (
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'sitting down',
    'standing up (from sitting position)', 'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket',
    'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
    'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket',
    'hopping (one foot jumping)', 'jump up', 'make a phone call/answer phone', 'playing with phone/tablet',
    'typing on a keyboard', 'pointing to something with finger', 'taking a selfie', 'check time (from watch)',
    'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute', 'put the palms together',
    'cross hands in front (say stop)', 'sneeze/cough', 'staggering', 'falling', 'touch head (headache)',
    'touch chest (stomachache/heart pain)', 'touch back (backache)', 'touch neck (neckache)',
    'nausea or vomiting condition', 'use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person', 'kicking other person', 'pushing other person',
    'pat on back of other person', 'point finger at the other person', 'hugging other person',
    'giving something to other person', "touch other person's pocket", 'handshaking', 'walking towards each other',
    'walking apart from each other', 'put on headphone', 'take off headphone', 'shoot at the basket',
    'bounce ball', 'tennis bat swing', 'juggling table tennis balls', 'hush (quite)', 'flick hair', 'thumb up',
    'thumb down', 'make ok sign', 'make victory sign', 'staple book', 'counting money', 'cutting nails',
    'cutting paper (using scissors)', 'snapping fingers', 'open bottle', 'sniff (smell)', 'squat down',
    'toss a coin', 'fold paper', 'ball up paper', 'play magic cube', 'apply cream on face',
    'apply cream on hand back', 'put on bag', 'take off bag', 'put something into a bag',
    'take something out of a bag', 'open a box', 'move heavy objects', 'shake fist', 'throw up cap/hat',
    'hands up (both hands)', 'cross arms', 'arm circles', 'arm swings', 'running on the spot',
    'butt kicks (kick backward)', 'cross toe touch', 'side kick', 'yawn', 'stretch oneself', 'blow nose',
    'hit other person with something', 'wield knife towards other person',
    'knock over other person (hit with body)', "grab other person's stuff", 'shoot at other person with a gun',
    'step on foot', 'high-five', 'cheers and drink', 'carry something with other person',
    'take a photo of other person', 'follow other person', "whisper in other person's ear",
    'exchange things with other person', 'support somebody with hand',
    'finger-guessing game (playing rock-paper-scissors)',
)

# NTU action -> our category. Every other action (waving, thumbs up, sitting, phone, hugging...) is ordinary.
INCIDENT_ACTIONS = {
    "falling": "person_down",
    "punching/slapping other person": "fighting", "kicking other person": "fighting",
    "pushing other person": "fighting", "hit other person with something": "fighting",
    "knock over other person (hit with body)": "fighting",
    "wield knife towards other person": "weapon", "shoot at other person with a gun": "weapon",
    "grab other person's stuff": "theft", "touch other person's pocket": "theft",
}

CLIP_LEN = 100   # frames per sample (PYSKL test pipeline)
NUM_PERSON = 2
# COCO-17 graph and bone pairs, as in pyskl/utils/graph.py and JointToBone (dataset='coco')
COCO_INWARD = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6),
               (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
COCO_BONES = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0),
              (12, 0), (13, 11), (14, 12), (15, 13), (16, 14)]


def coco_adjacency() -> np.ndarray:
    """Graph(layout='coco', mode='spatial').A: identity, inward and outward, column-normalised. (3, 17, 17)"""
    n = 17

    def edge2mat(links):
        a = np.zeros((n, n))
        for i, j in links:
            a[j, i] = 1
        return a

    def normalize(a):
        d = a.sum(0)
        dn = np.diag([1 / x if x > 0 else 0 for x in d])
        return a @ dn

    return np.stack([edge2mat([(i, i) for i in range(n)]), normalize(edge2mat(COCO_INWARD)),
                     normalize(edge2mat([(j, i) for i, j in COCO_INWARD]))]).astype(np.float32)


# ---------- preprocessing (numpy, testable without torch) ----------

def normalize_2d(kp: np.ndarray, score: np.ndarray, img_wh: tuple[int, int], threshold: float = 0.01) -> np.ndarray:
    """PreNormalize2D (mode 'fix') on a 16:9 canvas around the frame. kp (M,T,V,2) pixels, score (M,T,V).
    Returns (M,T,V,3): x, y in about [-1, 1] and the score; joints at or below `threshold` are zeroed."""
    w, h = img_wh
    cw, ch = max(w, h * 16 / 9), max(h, w * 9 / 16)  # NTU frames are 1920x1080
    out = np.concatenate([kp.astype(np.float32), score[..., None].astype(np.float32)], -1)
    out[..., 0] = (out[..., 0] - w / 2) / (cw / 2)
    out[..., 1] = (out[..., 1] - h / 2) / (ch / 2)
    off = out[..., 2] <= threshold
    out[..., 0][off] = 0
    out[..., 1][off] = 0
    return out


def to_bone(j: np.ndarray) -> np.ndarray:
    """JointToBone for COCO with a score channel: bone = joint - parent, score = mean of the two."""
    b = np.zeros_like(j)
    for v1, v2 in COCO_BONES:
        b[..., v1, :2] = j[..., v1, :2] - j[..., v2, :2]
        b[..., v1, 2] = (j[..., v1, 2] + j[..., v2, 2]) / 2
    return b


def resample(x: np.ndarray, length: int = CLIP_LEN) -> np.ndarray:
    """Linear interpolation along time (axis 1) to `length` frames. x (M,T,V,C)."""
    t = x.shape[1]
    if t == length:
        return x
    if t == 1:
        return np.repeat(x, length, axis=1)
    pos = np.linspace(0, t - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    f = (pos - lo)[None, :, None, None].astype(np.float32)
    return x[:, lo] * (1 - f) + x[:, hi] * f


def pad_persons(x: np.ndarray, m: int = NUM_PERSON) -> np.ndarray:
    """FormatGCNInput(num_person=2, mode='zero')."""
    if x.shape[0] >= m:
        return x[:m]
    return np.concatenate([x, np.zeros((m - x.shape[0],) + x.shape[1:], x.dtype)], 0)


def order_people(frames: list[tuple[np.ndarray, np.ndarray]], m: int = NUM_PERSON) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame detections -> fixed person slots over time. frames: list of (kp (P,17,2), score (P,17)), people
    sorted largest first. Slot k keeps the person whose centre is nearest to slot k's centre in the frame before
    (greedy), so two people do not swap slots between frames. Returns kp (m,T,17,2), score (m,T,17)."""
    t = len(frames)
    kp = np.zeros((m, t, 17, 2), np.float32)
    sc = np.zeros((m, t, 17), np.float32)
    prev: list[np.ndarray | None] = [None] * m
    for ti, (k, s) in enumerate(frames):
        k, s = k[:m], s[:m]
        centres = [kk[ss > 0.3].mean(0) if (ss > 0.3).any() else kk.mean(0) for kk, ss in zip(k, s)]
        free = list(range(len(k)))
        for slot in range(m):
            if not free:
                break
            if prev[slot] is None:
                pick = free[0]
            else:
                pick = min(free, key=lambda i: float(np.linalg.norm(centres[i] - prev[slot])))
            free.remove(pick)
            kp[slot, ti], sc[slot, ti] = k[pick], s[pick]
            prev[slot] = centres[pick]
    return kp, sc


def prepare(kp: np.ndarray, score: np.ndarray, img_wh: tuple[int, int], feat: str) -> np.ndarray:
    """(m,T,17,2) + (m,T,17) -> model input (1, M=2, 100, 17, 3) for feat 'j' (joints) or 'b' (bones)."""
    x = normalize_2d(kp, score, img_wh)
    if feat == "b":
        x = to_bone(x)
    return pad_persons(resample(x))[None].astype(np.float32)


def incident_from_probs(probs: np.ndarray) -> tuple[str | None, float, str]:
    """Sum the NTU probabilities per category. Returns (best incident category or None, its probability,
    top NTU action name)."""
    per: dict[str, float] = {}
    for i, name in enumerate(NTU120):
        cat = INCIDENT_ACTIONS.get(name)
        if cat:
            per[cat] = per.get(cat, 0.0) + float(probs[i])
    top = NTU120[int(np.argmax(probs))]
    if not per:
        return None, 0.0, top
    cat = max(per, key=per.get)
    return cat, per[cat], top


# ---------- model (torch) ----------

def _build_model(num_classes: int = 120):
    import torch
    import torch.nn as nn

    class UnitTCN(nn.Module):
        def __init__(self, cin, cout, k=9, stride=1, dilation=1, norm=True):
            super().__init__()
            pad = (k + (k - 1) * (dilation - 1) - 1) // 2
            self.conv = nn.Conv2d(cin, cout, (k, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
            self.bn = nn.BatchNorm2d(cout) if norm else nn.Identity()

        def forward(self, x):
            return self.bn(self.conv(x))

    class MSTCN(nn.Module):
        CFG = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"]

        def __init__(self, cin, cout, stride=1):
            super().__init__()
            nb = len(self.CFG)
            mid = cout // nb
            rem = cout - mid * (nb - 1)
            act = nn.ReLU()
            branches = []
            for i, cfg in enumerate(self.CFG):
                bc = rem if i == 0 else mid
                if cfg == "1x1":
                    branches.append(nn.Conv2d(cin, bc, 1, stride=(stride, 1)))
                elif cfg[0] == "max":
                    branches.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), act,
                                                  nn.MaxPool2d((cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                else:
                    branches.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), act,
                                                  UnitTCN(bc, bc, cfg[0], stride, cfg[1], norm=False)))
            self.branches = nn.ModuleList(branches)
            tin = mid * (nb - 1) + rem
            self.transform = nn.Sequential(nn.BatchNorm2d(tin), act, nn.Conv2d(tin, cout, 1))
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return self.bn(self.transform(torch.cat([b(x) for b in self.branches], 1)))

    class UnitGCN(nn.Module):  # adaptive='init', conv_pos='pre', with_res=True
        def __init__(self, cin, cout, A):
            super().__init__()
            self.k = A.shape[0]
            self.A = nn.Parameter(A.clone())
            self.conv = nn.Conv2d(cin, cout * self.k, 1)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.ReLU()
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout)) if cin != cout else None

        def forward(self, x):
            n, c, t, v = x.shape
            res = self.down(x) if self.down is not None else x
            x = self.conv(x).view(n, self.k, -1, t, v)
            x = torch.einsum("nkctv,kvw->nctw", x, self.A).contiguous()
            return self.act(self.bn(x) + res)

    class Block(nn.Module):
        def __init__(self, cin, cout, A, stride=1, residual=True):
            super().__init__()
            self.gcn = UnitGCN(cin, cout, A)
            self.tcn = MSTCN(cout, cout, stride)
            self.relu = nn.ReLU()
            if not residual:
                self.residual = None
            elif cin == cout and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = UnitTCN(cin, cout, 1, stride)

        def forward(self, x):
            res = 0 if self.residual is None else self.residual(x)
            return self.relu(self.tcn(self.gcn(x)) + res)

    class Backbone(nn.Module):  # STGCN(in_channels=3, base_channels=64, num_stages=10, inflate/down [5, 8])
        def __init__(self):
            super().__init__()
            A = torch.tensor(coco_adjacency())
            self.data_bn = nn.BatchNorm1d(3 * 17)
            blocks, base = [Block(3, 64, A, 1, residual=False)], 64
            for i in range(2, 11):
                stride = 1 + (i in (5, 8))
                cout = base * 2 if i in (5, 8) else base
                blocks.append(Block(base, cout, A, stride))
                base = cout
            self.gcn = nn.ModuleList(blocks)

        def forward(self, x):  # N, M, T, V, C
            n, m, t, v, c = x.shape
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = self.data_bn(x.view(n * m, v * c, t))
            x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
            for blk in self.gcn:
                x = blk(x)
            return x.reshape((n, m) + x.shape[1:])

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_cls = nn.Linear(256, num_classes)

        def forward(self, x):  # N, M, C, T, V
            return self.fc_cls(x.mean(dim=(3, 4)).mean(dim=1))

    class STGCNPP(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.cls_head = Head()

        def forward(self, x):
            return self.cls_head(self.backbone(x))

    return STGCNPP()


STREAMS = {"j": "stgcnpp_ntu120_xsub_hrnet_j.pth", "b": "stgcnpp_ntu120_xsub_hrnet_b.pth"}


class SkeletonActionModel:
    """Joint and bone streams (whichever files are in `model_dir`); softmax scores averaged."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        import torch

        self.torch = torch
        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.nets: dict[str, object] = {}
        for feat, name in STREAMS.items():
            p = Path(model_dir) / name
            if not p.exists():
                continue
            net = _build_model()
            sd = torch.load(p, map_location="cpu", weights_only=True)
            net.load_state_dict(sd.get("state_dict", sd), strict=True)
            self.nets[feat] = net.to(self.device).eval()
        if not self.nets:
            raise FileNotFoundError(f"no STGCN++ checkpoint ({', '.join(STREAMS.values())}) in {model_dir}")
        log.info("skeleton model: streams %s on %s", "+".join(self.nets), self.device)

    def probs(self, kp: np.ndarray, score: np.ndarray, img_wh: tuple[int, int]) -> np.ndarray:
        """kp (m,T,17,2) pixels, score (m,T,17) -> 120 action probabilities."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for feat, net in self.nets.items():
                x = torch.from_numpy(prepare(kp, score, img_wh, feat)).to(self.device)
                out.append(net(x).softmax(-1)[0].float().cpu().numpy())
        return np.mean(out, 0)


class PoseSequence:
    """YOLO pose on whole frames; keeps the keypoints of the current window."""

    def __init__(self, weights: str = "yolo11n-pose.pt", device: str = "cpu", conf: float = 0.3):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = 0 if device.startswith("cuda") else "cpu"
        self.conf = conf
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []
        self.wh: tuple[int, int] | None = None

    def add(self, image_bgr: np.ndarray) -> None:
        r = self.model.predict(image_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        self.wh = (image_bgr.shape[1], image_bgr.shape[0])
        if r.keypoints is None or len(r.boxes) == 0:
            self.frames.append((np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)))
            return
        xy = r.keypoints.xy.cpu().numpy()
        sc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2], np.float32)
        area = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
        order = np.argsort(-area)
        self.frames.append((xy[order], sc[order]))

    def take(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
        """The window's keypoints in person slots, then start a new window. None if nobody was seen."""
        frames, self.frames = self.frames, []
        if not frames or self.wh is None or not any(len(k) for k, _ in frames):
            return None
        kp, sc = order_people(frames)
        return kp, sc, self.wh
