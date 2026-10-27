"""Тесты раннера абляции ECRv1 (run_ec_ablation.py, задача D7): конфиги, проверка полей, сохранение
предсказаний, сводка, зашумлённый test. Обучение здесь не запускается (смоук с Trainer — E1)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
import run_ec_ablation as r  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings, TrainingSettings  # noqa: E402

AUG_KEYS = ["random_resize", "random_crop", "random_reverse", "phase_spectrum_noise", "random_jitter",
            "random_invert", "channel_scaling", "time_masking", "channel_masking", "mosaic", "mix_up",
            "sine_noise", "square_noises"]


def test_cumulative_groups():
    a0, a8 = r.ablation_augmentations("A0"), r.ablation_augmentations("A8")
    assert all(not a0[k] for k in AUG_KEYS)
    assert all(a8[k] for k in AUG_KEYS)
    prev = a0
    for gid, _, delta in r.GROUPS[1:]:
        cur = r.ablation_augmentations(gid)
        changed = {k for k in cur if cur[k] != prev[k]}
        assert changed == set(delta), gid           # каждый шаг добавляет ровно свою группу
        prev = cur
    assert a8["square_noises"] == 0.1 and a8["sine_noise"] == 0.5 and a8["mosaic"] == 0.5


def test_configs_are_valid_training_settings(tmp_path):
    paths = r.write_configs(tmp_path)
    assert [p.stem for p in paths] == [g[0] for g in r.GROUPS] + ["IN_raw", "IN_filtered", "A8_noRev", "A8_noMask"]
    for gid in [g[0] for g in r.GROUPS]:
        cfg = r.load_config(gid, config_dir=tmp_path)
        s = TrainingSettings(**cfg["training"], dataset="x.yaml", seed=1)
        assert s.model_size == "large" and s.batch_size == 50 and s.weight_box == 10 and s.weight_dfl == 3
        assert s.best_metric == "mAP"
        assert s.augmentations.pipeline == "eddy" and s.augmentations.augment_apply_p == 0.5
        assert cfg["seeds"] == r.SEEDS == [0] and cfg["dataset"] == "data/ecr_v1.yaml"
    assert r.load_config("A3", config_dir=tmp_path, epochs=2)["training"]["epochs"] == 2


def test_input_configs_use_base(tmp_path):
    r.write_configs(tmp_path, in_base="A6")
    cfg = yaml.safe_load((tmp_path / "IN_raw.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["input_subset"] == "raw"
    assert cfg["training"]["augmentations"] == r.ablation_augmentations("A6")


def test_unknown_and_forbidden_keys():
    with pytest.raises(ValueError, match="random_crpo"):
        r.check_training_keys({"names": ["a"], "augmentations": {"random_crpo": True}})
    with pytest.raises(ValueError, match="seed"):
        r.check_training_keys({"names": ["a"], "seed": 3})
    r.check_training_keys(r.make_config("A8")["training"])


def test_predictions_roundtrip(tmp_path):
    per_sample = [(np.zeros((0, 4)), np.array([[1.0]]), np.array([[0.5, 0.1]])),
                  (np.ones((2, 4)), np.array([[0.0], [2.0]]), np.array([[0.2, 0.1], [0.7, 0.05]]))]
    r.save_predictions(tmp_path / "p.npz", ["a", "b"], per_sample)
    p = r.load_predictions(tmp_path / "p.npz")
    assert list(p["ids"]) == ["a", "b"]
    assert p["det"][0].shape == (0, 4) and p["det"][1].shape == (2, 4)
    np.testing.assert_array_equal(p["box"][1], per_sample[1][2].astype(np.float32))


def test_metrics_per_class_names():
    names = ["a", "b", "c"]
    m = r._metrics({"mAP": 0.5, "mAP@50": 0.7, "Precision": 0.6, "Recall": 0.4},
                   ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.3, 0.4, 0.5]), names)
    assert m["mAP"] == 0.5 and m["per_class"]["b"] == {"AP": 0.4, "AP@50": 0.8, "Precision": 0.2, "Recall": 0.5}
    m = r._metrics({}, ([0.1], [0.2], [0.3], [0.4]), names)
    assert "per_class_unnamed" in m


def fake_run(runs, cid, seed, mAP, noisy=None, epochs=None, val_curve=None):
    d = runs / (f"{cid}_e{epochs}_s{seed}" if epochs else f"{cid}_s{seed}")
    d.mkdir(parents=True)
    if epochs:
        (d / "config.yaml").write_text(yaml.safe_dump({"name": cid, "training": {"epochs": epochs}}))
    if val_curve:
        (d / "train1").mkdir()
        lines = ["epoch,box_train,mAP", "001,1.0,"] + [f"{i + 2:03d},1.0,{v}" for i, v in enumerate(val_curve)]
        (d / "train1" / "step.csv").write_text("\n".join(lines) + "\n")
    split = {"mAP": mAP, "mAP@50": mAP + 0.1, "Precision": 0.5, "Recall": 0.5,
             "per_class": {n: {"AP": mAP, "AP@50": mAP, "Precision": 0, "Recall": 0} for n in r.NAMES}}
    (d / "metrics.json").write_text(json.dumps({"config": cid, "seed": seed, "val": split, "test": split}))
    if noisy is not None:
        (d / "noisy_test.json").write_text(json.dumps({"test_noisy": {**split, "mAP": noisy}}))


def test_summary(tmp_path):
    runs = tmp_path / "runs"
    fake_run(runs, "A1", 0, 0.4)
    fake_run(runs, "A1", 1, 0.6, noisy=0.3)
    fake_run(runs, "A0", 0, 0.2)
    agg = r.summarize(runs, tmp_path)
    assert [a["config"] for a in agg] == ["A0", "A1"]               # порядок групп, а не алфавит
    a1 = agg[1]
    assert a1["n_seeds"] == 2 and a1["test_mAP_mean"] == pytest.approx(0.5)
    assert a1["test_mAP_sd"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert a1["test_noisy_mAP_mean"] == pytest.approx(0.3)
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").startswith("config,seed,")
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| A1 | — | — | 50.00 | 60.00 |" in md


def test_budget_run_names_do_not_collide():
    main = r.make_config("A8")
    budget = {**r.make_config("A8"), "budget": 50}
    assert r.run_name(main, 0) == "A8_s0" and r.run_name(budget, 0) == "A8_e50_s0"
    assert r.default_runs_dir(main) == r.RUNS_DIR and r.default_runs_dir(budget) == r.RUNS_BUDGET_DIR
    assert r.RUNS_DIR != r.RUNS_BUDGET_DIR
    assert r.BASE_TRAINING["early_stopping_epochs"] == 0


def test_load_config_epochs_marks_budget(tmp_path):
    r.write_configs(tmp_path)
    assert "budget" not in r.load_config("A0", tmp_path)
    c = r.load_config("A0", tmp_path, epochs=50)
    assert c["budget"] == 50 and c["training"]["epochs"] == 50


def test_training_progress(tmp_path):
    fake_run(tmp_path, "A8", 0, 0.5, epochs=5, val_curve=[0.1, 0.4, 0.4, 0.3])
    p = r.training_progress(tmp_path / "A8_e5_s0")
    assert p == {"stopped_epoch": 5, "best_epoch": 3}               # первый максимум, эпоха 1 без mAP
    assert r.training_progress(tmp_path / "missing") == {}


def test_budget_summary_groups_by_epochs(tmp_path):
    runs = tmp_path / "runs_budget"
    fake_run(runs, "A8", 0, 0.70, epochs=100, val_curve=[0.5, 0.7])
    fake_run(runs, "A8", 0, 0.60, epochs=50, val_curve=[0.6, 0.5])
    fake_run(runs, "A0", 0, 0.30, epochs=50, val_curve=[0.3])
    agg = r.summarize(runs, tmp_path)
    assert [(a["config"], a["epochs"]) for a in agg] == [("A0", 50), ("A8", 50), ("A8", 100)]
    assert [a["n_seeds"] for a in agg] == [1, 1, 1]
    assert agg[2]["test_mAP_mean"] == pytest.approx(0.70) and agg[1]["best_epoch_mean"] == 2
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").startswith("config,epochs,seed,")
    assert "| A8 | 100 | 3 | 70.00 | 80.00 |" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_noisy_test_transform_deterministic(tmp_path):
    s = AugmentationSettings(regular_length=6000, input_length=4096)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(np.round(rng.normal(0, 40, (15, 6000))).astype(np.float32))
    box, cls = torch.tensor([[0.5, 0.1]]), torch.tensor([[1.0]])
    t = r.noisy_test_transform(s, None, noise_seed=0)
    y1, _, _ = t(x.clone(), box.clone(), cls)
    np.random.seed(123)                                              # глобальное состояние не влияет
    y2, _, _ = t(x.clone(), box.clone(), cls)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    y3, _, _ = r.noisy_test_transform(s, None, noise_seed=1)(x.clone(), box.clone(), cls)
    assert not torch.equal(y1, y3)
    from yolo_1d.utils.transforms import EddyPreprocessing, build_validation_transform
    clean, _, _ = build_validation_transform(s, EddyPreprocessing())(x.clone(), box.clone(), cls)
    assert y1.shape == clean.shape == (30, 4096) and not torch.equal(y1, clean)
    # с кэшем — отдельная папка, результат совпадает с точностью float16
    yc, _, _ = r.noisy_test_transform(s, str(tmp_path), noise_seed=0)(x.clone(), box.clone(), cls)
    assert list((tmp_path / "noisy_test_s0" / "preprocessed").rglob("*.npy"))
    torch.testing.assert_close(yc, y1, rtol=1e-3, atol=1e-2)



def test_noisy_test_transform_picklable():
    """Трансформ зашумлённого test передаётся воркерам DataLoader (spawn) — должен pickle'иться."""
    import pickle
    tr = r.noisy_test_transform(AugmentationSettings(**r.ablation_augmentations("A0")), None, noise_seed=3)
    x = torch.from_numpy(np.random.default_rng(0).normal(0, 100, (15, 6000)).astype(np.float32))
    a = tr(x.clone(), torch.zeros(0, 2), torch.zeros(0, 1))[0]
    b = pickle.loads(pickle.dumps(tr))(x.clone(), torch.zeros(0, 2), torch.zeros(0, 1))[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_leave_out_configs():
    a8 = r.make_config("A8")["training"]["augmentations"]
    no_rev = r.make_config("A8_noRev")["training"]["augmentations"]
    no_mask = r.make_config("A8_noMask")["training"]["augmentations"]
    assert no_rev == {**a8, "random_reverse": 0} and a8["random_reverse"] > 0
    assert no_mask == {**a8, "time_masking": 0, "channel_masking": 0} and a8["time_masking"] > 0
    assert r.all_config_ids()[-2:] == ["A8_noRev", "A8_noMask"]


def test_fixed_conf_metrics(tmp_path):
    """P / R при пороге уверенности: детекции ниже порога отбрасываются до сопоставления с разметкой."""
    run = tmp_path / "A8_s0"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"augmentations": {"input_length": 1000}}}))
    # сигнал 1: crossing [100, 300] найден (0.9), ложный crossing (0.6), слабый дубль ниже порога (0.3)
    # сигнал 2: axle counter [500, 520] не найден (детекция 0.4 < 0.5)
    det = [np.array([[100, 300, 0.9, 4], [600, 700, 0.6, 4], [110, 290, 0.3, 4]]), np.array([[500, 520, 0.4, 3]])]
    cls = [np.array([4.0]), np.array([3.0])]
    box = [np.array([[0.2, 0.2]]), np.array([[0.51, 0.02]])]
    r.save_predictions(run / "predictions_test.npz", ["a", "b"], list(zip(det, cls, box)))
    m = r.fixed_conf_metrics(run, conf=0.5)
    assert m["per_class"]["crossing"] == {"P": 0.5, "R": 1.0, "n": 1}
    assert m["per_class"]["axle counter"] == {"P": 0.0, "R": 0.0, "n": 1}
    assert m["P"] == pytest.approx(0.25) and m["R"] == pytest.approx(0.5)      # макро по классам разметки
    assert r.fixed_conf_metrics(run, conf=0.35)["per_class"]["axle counter"]["R"] == 1.0
