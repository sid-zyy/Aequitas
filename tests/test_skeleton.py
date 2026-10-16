import numpy as np
import pytest

from aequitas.vision.skeleton import (INCIDENT_ACTIONS, NTU120, coco_adjacency, incident_from_probs, normalize_2d,
                                      order_people, prepare, resample, to_bone)


def test_incident_actions_are_ntu_classes():
    assert len(NTU120) == 120
    assert set(INCIDENT_ACTIONS) <= set(NTU120)
    assert "hand waving" not in INCIDENT_ACTIONS and "thumb up" not in INCIDENT_ACTIONS


def test_adjacency_shape_and_normalisation():
    a = coco_adjacency()
    assert a.shape == (3, 17, 17)
    assert np.allclose(a[0], np.eye(17))
    for k in (1, 2):  # columns with an edge sum to 1
        s = a[k].sum(0)
        assert np.allclose(s[s > 0], 1)


def test_normalize_puts_frame_centre_at_zero_and_uses_a_16_9_canvas():
    kp = np.array([[[[320, 240], [640, 240], [0, 0]]]], np.float32)  # M=1, T=1, V=3
    sc = np.array([[[0.9, 0.9, 0.0]]], np.float32)
    out = normalize_2d(kp, sc, (640, 480))
    assert out[0, 0, 0, :2].tolist() == [0, 0]
    # 480 high -> canvas 853 wide: the right edge of a 640 frame is at 320 / 426.7
    assert out[0, 0, 1, 0] == pytest.approx(320 / (480 * 16 / 9 / 2))
    assert out[0, 0, 2].tolist() == [0, 0, 0]  # zero score: zeroed


def test_bone_is_joint_minus_parent():
    j = np.zeros((1, 1, 17, 3), np.float32)
    j[0, 0, 5] = [1, 2, 1]   # left shoulder, parent nose (0)
    j[0, 0, 7] = [3, 5, 0.5]  # left elbow, parent left shoulder
    b = to_bone(j)
    assert b[0, 0, 7, :2].tolist() == [2, 3]
    assert b[0, 0, 7, 2] == pytest.approx(0.75)
    assert b[0, 0, 0].tolist() == [0, 0, 0]


def test_resample_keeps_endpoints_and_interpolates():
    x = np.zeros((1, 3, 1, 1), np.float32)
    x[0, :, 0, 0] = [0, 10, 20]
    r = resample(x, 5)
    assert r[0, :, 0, 0].tolist() == [0, 5, 10, 15, 20]
    assert resample(x[:, :1], 4).shape == (1, 4, 1, 1)


def test_order_people_keeps_slots_when_detection_order_swaps():
    a = np.full((17, 2), 100, np.float32)
    b = np.full((17, 2), 500, np.float32)
    s = np.ones((17,), np.float32)
    frames = [(np.stack([a, b]), np.stack([s, s])), (np.stack([b, a]), np.stack([s, s]))]
    kp, _ = order_people(frames)
    assert kp[0, 0, 0, 0] == 100 and kp[0, 1, 0, 0] == 100
    assert kp[1, 0, 0, 0] == 500 and kp[1, 1, 0, 0] == 500


def test_prepare_shape_and_one_person_padding():
    kp = np.random.rand(1, 16, 17, 2).astype(np.float32) * 400
    sc = np.ones((1, 16, 17), np.float32)
    x = prepare(kp, sc, (640, 480), "b")
    assert x.shape == (1, 2, 100, 17, 3)
    assert not x[0, 1].any()


def test_incident_from_probs_sums_per_category():
    p = np.zeros(120)
    p[NTU120.index("hand waving")] = 0.5
    p[NTU120.index("punching/slapping other person")] = 0.2
    p[NTU120.index("pushing other person")] = 0.15
    p[NTU120.index("falling")] = 0.1
    cat, score, top = incident_from_probs(p)
    assert (cat, round(score, 2), top) == ("fighting", 0.35, "hand waving")


def test_port_has_the_checkpoint_layout():
    torch = pytest.importorskip("torch")
    from aequitas.vision.skeleton import _build_model

    net = _build_model().eval()
    keys = net.state_dict().keys()
    assert "backbone.gcn.0.gcn.down.0.weight" in keys and "backbone.gcn.4.residual.conv.weight" in keys
    assert "backbone.gcn.0.tcn.branches.0.3.conv.weight" in keys and "cls_head.fc_cls.weight" in keys
    assert sum(1 for k in keys if k.endswith(".A")) == 10
    with torch.no_grad():
        out = net(torch.zeros(1, 2, 100, 17, 3))
    assert out.shape == (1, 120)
