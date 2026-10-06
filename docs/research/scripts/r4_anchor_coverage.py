"""Покрытие волн LUDB якорями при разных страйдах (R4).

TAL-ассайнер (nets/criterion.py) берёт только якоря, лежащие внутри GT; DFL с 16 бинами
ограничивает расстояние от якоря до границы величиной 15*stride.
Использование: python r4_anchor_coverage.py <папка LUDB>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from r1_ludb_stats import LEADS, parse  # noqa: E402

REG = 15


def main(root):
    widths = {'P': [], 'QRS': [], 'T': []}
    for r in range(1, 201):
        for lead in LEADS:
            for c, on, _, off in parse(os.path.join(root, 'data', str(r)), lead)[0]:
                widths[c].append(off - on)
    print('stride | class | mean anchors inside | % >=1 | % >=3 | % regressable (w/2 <= 15*s)')
    for s in [2, 4, 8, 16, 32, 64, 128]:
        for c, w in widths.items():
            w = np.array(w, float)
            print(f'{s:4d} | {c:3s} | {(w / s).mean():6.1f} | {(np.floor(w / s) >= 1).mean() * 100:5.1f} | '
                  f'{(np.floor(w / s) >= 3).mean() * 100:5.1f} | {((w / 2) <= REG * s).mean() * 100:5.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
