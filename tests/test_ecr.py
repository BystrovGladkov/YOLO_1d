"""Тесты источника данных и манифеста ECRv1 (yolo_1d.utils.ecr, задача D2).

Проверка реальной локальной копии пропускается, если датасет недоступен.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils import ecr  # noqa: E402


def make_layout(root: Path, n=2, orphan_label=False, orphan_data=False):
    """Мини-копия структуры ECRv1."""
    for split in ecr.SPLITS:
        (root / split / "data").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            name = f"passage1_left_{i:04d}"
            np.save(root / split / "data" / f"{name}.npy", np.zeros((15, 100), dtype=np.int16))
            (root / split / "labels" / f"{name}.txt").write_text("2 10 30\n")
    if orphan_label:
        (root / "train" / "labels" / "passage9_right_0001.txt").write_text("1 0 5\n")
    if orphan_data:
        np.save(root / "val" / "data" / "passage9_right_0002.npy", np.zeros((15, 10), dtype=np.int16))
    return root


def test_validate_layout_ok(tmp_path):
    stats = ecr.validate_layout(make_layout(tmp_path, n=3))
    assert stats == {s: {"data": 3, "labels": 3, "pairs": 3} for s in ecr.SPLITS}


@pytest.mark.parametrize("kw", [{"orphan_label": True}, {"orphan_data": True}])
def test_validate_layout_mismatch(tmp_path, kw):
    root = make_layout(tmp_path, **kw)
    with pytest.raises(ValueError):
        ecr.validate_layout(root)
    stats = ecr.validate_layout(root, strict=False)
    assert all(s["pairs"] == 2 for s in stats.values())


def test_validate_layout_missing_split(tmp_path):
    root = make_layout(tmp_path)
    for p in (root / "test" / "labels").iterdir():
        p.unlink()
    (root / "test" / "labels").rmdir()
    with pytest.raises(ValueError, match="test"):
        ecr.validate_layout(root)


def test_get_local(tmp_path):
    root = make_layout(tmp_path)
    assert ecr.get_ecr_dataset(root) == root


@pytest.mark.parametrize("source", ["hf://Nasochec/ECRv1", "Nasochec/ECRv1"])
def test_get_hf_calls_snapshot_download(tmp_path, monkeypatch, source):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub
    root = make_layout(tmp_path / "snap")
    calls = {}

    def fake_snapshot_download(**kw):
        calls.update(kw)
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    got = ecr.get_ecr_dataset(source, cache_dir=tmp_path / "cache", revision="v1")
    assert got == root
    assert calls == {"repo_id": "Nasochec/ECRv1", "repo_type": "dataset", "revision": "v1",
                     "local_dir": str(tmp_path / "cache")}


def test_get_unknown_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        ecr.get_ecr_dataset(tmp_path / "no_such_dir")


def test_write_manifest(tmp_path):
    root = make_layout(tmp_path / "ds")
    out = ecr.write_manifest(root, tmp_path / "cfg" / "ecr.yaml")
    m = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert m["format"] == "ecr_npy"
    assert m["names"] == {0: "flashbutt", 1: "aluminothermic", 2: "rail end", 3: "axle counter",
                          4: "crossing", 5: "manual arc"}
    assert m["class_map"] == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 12: 5}
    for s in ecr.SPLITS:
        assert Path(m[f"{s}_data"]) == (root / s / "data").resolve()
        assert Path(m[f"{s}_labels"]) == (root / s / "labels").resolve()
    assert Path(m["preprocess_cache"]) == (tmp_path / "cfg" / "cache").resolve()


def test_prepare_script(tmp_path):
    sys.path.insert(0, str(REPO / "data"))
    import prepare_ecr
    root = make_layout(tmp_path / "ds")
    prepare_ecr.main(["--source", str(root), "--out", str(tmp_path / "m.yaml")])
    assert (tmp_path / "m.yaml").exists()


@pytest.mark.skipif(not ecr.DEFAULT_LOCAL_ROOT.exists(), reason="локальная копия ECRv1 недоступна")
def test_real_local_copy():
    stats = ecr.validate_layout(ecr.DEFAULT_LOCAL_ROOT)
    assert {s: v["pairs"] for s, v in stats.items()} == {"train": 2099, "val": 560, "test": 294}
