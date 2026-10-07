"""Тесты чтения LUDB (yolo_1d.utils.ludb).

Тесты парсера используют синтетические последовательности символов. Интеграционные
тесты на реальных данных пропускаются, если LUDB не скачан (python data/download_ludb.py).
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils import ludb  # noqa: E402
from yolo_1d.utils.ludb import Wave, parse_annotation  # noqa: E402

LUDB_ROOT = REPO / "data" / "raw" / "ludb"


def kinds(issues):
    return [(i.kind, i.kept) for i in issues]


def test_normal_triplets():
    waves, issues = parse_annotation([10, 20, 30, 40, 60, 80], ["(", "p", ")", "(", "N", ")"])
    assert waves == [Wave("P", 10, 20, 30), Wave("QRS", 40, 60, 80)]
    assert issues == []


def test_peak_before_onset_is_fixed():
    waves, issues = parse_annotation([100, 101, 130], ["N", "(", ")"])
    assert waves == [Wave("QRS", 100, 100, 130)]
    assert kinds(issues) == [(ludb.ISSUE_PEAK_BEFORE_ONSET, True)]


def test_peak_far_before_onset_is_not_fixed():
    # пик на 10 отсчётов раньше onset — не описка, а сбой: пик без onset, затем пара без пика
    waves, issues = parse_annotation([100, 110, 130], ["N", "(", ")"])
    assert waves == []
    assert kinds(issues) == [(ludb.ISSUE_UNPARSED, False), (ludb.ISSUE_NO_PEAK, False)]


def test_nested_duplicates_are_merged_by_median():
    x = [368, 388, 392, 418, 419, 424, 450, 455, 470]
    s = ["(", "(", "(", "t", "t", "t", ")", ")", ")"]
    waves, issues = parse_annotation(x, s)
    assert waves == [Wave("T", 388, 419, 455)]
    assert kinds(issues) == [(ludb.ISSUE_MERGED_DUPLICATES, True)]


def test_missing_onset_offset_and_peak_are_dropped():
    x = [10, 30, 50, 60, 90, 100, 120, 140, 150, 170]
    s = ["N", ")", "(", "t", "(", "p", ")", "(", ")", ")"]
    waves, issues = parse_annotation(x, s)
    assert waves == [Wave("P", 90, 100, 120)]
    assert kinds(issues) == [(ludb.ISSUE_MISSING_ONSET, False), (ludb.ISSUE_MISSING_OFFSET, False),
                             (ludb.ISSUE_NO_PEAK, False), (ludb.ISSUE_UNPARSED, False)]


def test_waves_sorted_by_onset():
    waves, _ = parse_annotation([50, 60, 70, 10, 20, 30], ["(", "t", ")", "(", "p", ")"])
    assert [w.onset for w in waves] == [10, 50]


def test_rhythm_group():
    assert ludb.rhythm_group({"Rhythms": ["Sinus rhythm"], "Cardiac pacing": []}) == "sinus"
    assert ludb.rhythm_group({"Rhythms": ["Atrial fibrillation"]}) == "af_flutter"
    assert ludb.rhythm_group({"Rhythms": ["Atrial flutter, typical"]}) == "af_flutter"
    assert ludb.rhythm_group({"Rhythms": ["Sinus rhythm"], "Cardiac pacing": ["Unipolar ventricular pacing"]}) == "paced"


needs_data = pytest.mark.skipif(not (LUDB_ROOT / "RECORDS").is_file(), reason="LUDB не скачан")


@needs_data
def test_load_record_1():
    rec = ludb.load_record(LUDB_ROOT, "1")
    assert rec.signal.shape == (12, 5000)
    assert set(rec.waves) == set(ludb.LEADS)
    assert rec.waves["ii"][0] == Wave("QRS", 644, 662, 682)  # первая тройка из R1
    start, end = rec.zone("ii")
    assert start == 644 and end > start
    assert rec.meta["Sex"] == ["F"] and rec.meta["Age"] == ["51"]
    assert rhythm_of(rec) == "Sinus bradycardia"


def rhythm_of(rec):
    return rec.meta["Rhythms"][0]


@needs_data
def test_record_104_drops_unlabelled_qrs():
    rec = ludb.load_record(LUDB_ROOT, "104")
    assert not any(w.cls == "QRS" for w in rec.waves["ii"])
    assert all(i.kind == ludb.ISSUE_MISSING_ONSET for i in rec.dropped("ii"))
    assert len(rec.dropped("ii")) > 0


@needs_data
def test_record_names_and_metadata():
    names = ludb.record_names(LUDB_ROOT)
    assert names[:3] == ["1", "2", "3"] and len(names) == 200
    meta = ludb.load_metadata(LUDB_ROOT)
    assert len(meta) == 200
