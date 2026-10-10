"""R5 §2.1–2.2: разброс размеченных зон между отведениями записи и непригодные отведения LUDB.

    python docs/research/scripts/r5_zones.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
rows = list(csv.DictReader(open(REPO / "data" / "processed" / "ludb" / "manifest.csv", encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[r["record"]].append(r)
bad = sorted({k for k, v in by.items() if any(x["usable"] != "1" for x in v)}, key=int)
print(f"отведений: {len(rows)}, непригодных: {sum(r['usable'] != '1' for r in rows)} в записях {bad}")
sp_s, sp_e, inter, union = [], [], [], []
for v in by.values():
    u = [x for x in v if x["usable"] == "1"]
    s = np.array([int(x["annot_start"]) for x in u])
    e = np.array([int(x["annot_end"]) for x in u])
    sp_s.append(np.ptp(s) * 2)
    sp_e.append(np.ptp(e) * 2)
    inter.append((e.min() - s.max()) / 500)
    union.append((e.max() - s.min()) / 500)
for name, x in (("разброс начала зоны, мс", sp_s), ("разброс конца зоны, мс", sp_e)):
    print(f"{name}: медиана {np.median(x):.0f}, p90 {np.percentile(x, 90):.0f}, max {np.max(x):.0f}")
print(f"пересечение зон, с: min {min(inter):.2f}, медиана {np.median(inter):.2f}; объединение, медиана {np.median(union):.2f}")
