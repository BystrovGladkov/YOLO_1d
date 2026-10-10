"""Тесты M7: голова с n_leads наборами предсказаний, лосс MultiLeadLoss, Trainer в режиме MC."""
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.nets.criterion import ComputeLoss, MultiLeadLoss  # noqa: E402
from yolo_1d.nets.export import ExportWrapper  # noqa: E402
from yolo_1d.nets.nn import get_model  # noqa: E402

L, NC, LEN = 12, 3, 5120
ARCH = dict(strides="B", bias_init="1d", expected_objects=24)


def model(n_leads=1, channels=1, **kw):
    torch.manual_seed(0)
    return get_model("nano", NC, input_channels=channels, input_length=LEN, n_leads=n_leads, **ARCH, **kw)


def head_inputs(m, batch=2, channels=1):
    x = torch.randn(batch, channels, LEN)
    return list(m.fpn(m.net(x)))


# ---------------------------------------------------------------------------------------------- голова
def test_single_lead_head_matches_original_layout():
    """n_leads = 1: выход головы = cat(box, cls) по каналам, как в исходной голове; имена/формы весов прежние."""
    m = model().eval()
    feats = head_inputs(m)
    out = m.head.train()([f.clone() for f in feats])
    for f, o, box, cls in zip(feats, out, m.head.box, m.head.cls):
        assert torch.equal(o, torch.cat((box(f), cls(f)), dim=1))
    assert m.head.box[0][-1].out_channels == 32 and m.head.cls[0][-1].out_channels == NC


@pytest.mark.skipif(not (REPO / "experiments" / "final" / "fold0_seed2" / "train1" / "best.pt").is_file(),
                    reason="нет чекпоинта E4")
def test_existing_checkpoint_loads_strictly():
    """Чекпоинты режима S (E2–E4) загружаются в новую голову без изменений."""
    ckpt = torch.load(REPO / "experiments" / "final" / "fold0_seed2" / "train1" / "best.pt", weights_only=False,
                      map_location="cpu")
    model().load_state_dict(ckpt["model"], strict=True)


def test_multilead_shapes_and_rows():
    """Выход MC — «батч отведений»: строка b·12 + k = срез выходных каналов отведения k."""
    m = model(L, L)
    feats = head_inputs(m, batch=2, channels=L)
    out = m.head.train()([f.clone() for f in feats])
    for f, o, box, cls in zip(feats, out, m.head.box, m.head.cls):
        b, c = box(f), cls(f)
        assert o.shape == (2 * L, 32 + NC, f.shape[-1])
        for bi in range(2):
            for k in (0, 5, 11):
                row = o[bi * L + k]
                assert torch.equal(row[:32], b[bi, k * 32:(k + 1) * 32])
                assert torch.equal(row[32:], c[bi, k * NC:(k + 1) * NC])
    m.eval()
    y = m(torch.randn(2, L, LEN))
    assert y.shape[0] == 2 * L and y.shape[1] == 2 + NC
    wrapped = ExportWrapper(m, input_length=LEN, resize=False)(torch.randn(2, L, 5000))
    assert wrapped.shape[:3] == (2, L, 2 + NC)


def test_bias_init_per_lead_and_width_mult():
    m = model(L, L, head_width_mult=2.0)
    for cls in m.head.cls:
        bias = cls[-1].bias.data
        assert bias.numel() == NC * L and torch.allclose(bias, bias[0].expand_as(bias))
    assert m.head.box[0][0].conv.out_channels == 128  # max(64, 128 // 4) · 2


# ---------------------------------------------------------------------------------------------- лосс
def mc_batch(m, batch=2):
    """Выход MC-модели (train) и цели: по 3 волны в каждом отведении."""
    out = m.train()(torch.randn(batch, L, LEN))
    cls, box, idx, lead = [], [], [], []
    for b in range(batch):
        for k in range(L):
            for j, c in enumerate((0, 1, 2)):
                cls.append([c])
                box.append([0.2 + 0.25 * j + 0.003 * k, 0.02])
                idx.append(b)
                lead.append(k)
    targets = {"cls": torch.tensor(cls, dtype=torch.float32), "box": torch.tensor(box, dtype=torch.float32),
               "idx": torch.tensor(idx, dtype=torch.float32), "lead": torch.tensor(lead, dtype=torch.float32),
               "valid": torch.ones(batch, L, dtype=torch.bool)}
    return out, targets


def flat(targets):
    return {"cls": targets["cls"], "box": targets["box"],
            "idx": (targets["idx"] * L + targets["lead"]).float()}


def test_joint_loss_is_compute_loss_over_lead_rows():
    m = model(L, L)
    out, targets = mc_batch(m)
    criterion = ComputeLoss(m, 7.5, 0.5, 1.5)
    got = MultiLeadLoss(criterion, L, "joint")(out, targets)
    ref = criterion(out, flat(targets))
    for a, b in zip(got, ref):
        assert torch.allclose(a, b)


def test_per_lead_loss_is_sum_of_lead_losses():
    m = model(L, L)
    out, targets = mc_batch(m)
    criterion = ComputeLoss(m, 7.5, 0.5, 1.5)
    got = MultiLeadLoss(criterion, L, "per_lead")(out, targets)
    ref = [0.0, 0.0, 0.0]
    for k in range(L):
        rows = torch.tensor([r % L == k for r in range(2 * L)])
        keep = targets["lead"] == k
        sub = {"cls": targets["cls"][keep], "box": targets["box"][keep], "idx": targets["idx"][keep]}
        losses = criterion([o[rows] for o in out], sub)
        ref = [a + b for a, b in zip(ref, losses)]
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, rtol=1e-5)


def test_invalid_lead_excluded_from_loss():
    """Отведение с valid = False не влияет на лосс и не получает градиента."""
    m = model(L, L)
    out, targets = mc_batch(m)
    targets["valid"][1, 4] = False
    keep = ~((targets["idx"] == 1) & (targets["lead"] == 4))  # как в данных: разметка непригодного отведения не пишется
    targets = {**{k: targets[k][keep] for k in ("cls", "box", "idx", "lead")}, "valid": targets["valid"]}
    loss = MultiLeadLoss(ComputeLoss(m, 7.5, 0.5, 1.5), L, "joint")
    out = [o.detach().requires_grad_(True) for o in out]
    total = sum(loss(out, targets))
    total.backward()
    row = 1 * L + 4
    assert all(o.grad[row].abs().sum() == 0 for o in out)
    perturbed = [o.detach().clone() for o in out]
    for o in perturbed:
        o[row] += 5.0
    assert torch.allclose(sum(loss(perturbed, targets)), total.detach())


# ---------------------------------------------------------------------------------------------- Trainer
needs_data = pytest.mark.skipif(not (REPO / "data" / "processed" / "ludb_mc" / "leads.csv").is_file(),
                                reason="нет data/processed/ludb_mc")


def mc_settings(tmp, **over):
    from yolo_1d.utils.settings import TrainingSettings
    config = yaml.safe_load((REPO / "experiments" / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    training = {**config["training"], "input_channels": L, "n_leads": L, "num_workers": 0, "batch_size": 4, **over}
    return TrainingSettings(**training, dataset=str(REPO / "data" / "splits" / "fold_0_mc.yaml"),
                            results_path=tmp, project="t", seed=0)


@needs_data
def test_trainer_collects_per_lead_predictions():
    from yolo_1d.yolo_1d import Trainer
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(mc_settings(tmp))
        samples, _ = trainer.collect_predictions("val")
    s_ids = (REPO / "data" / "splits" / "fold_0" / "val.txt").read_text(encoding="utf-8").split()
    assert [s["id"] for s in samples] == s_ids  # тот же порядок и набор отведений, что в режиме S
    for s in samples[:24]:
        assert s["ref"].shape[1] == 3 and len(s["ref"]) > 0 and s["pred"].shape[1] == 4


@needs_data
def test_trainer_rejects_inconsistent_mode():
    from yolo_1d.yolo_1d import Trainer
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(mc_settings(tmp, n_leads=1, input_channels=12))
        with pytest.raises(ValueError):
            trainer.collect_predictions("val")
