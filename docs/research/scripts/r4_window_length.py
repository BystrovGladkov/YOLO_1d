"""Почему не фиксированное окно внутри размеченной зоны (R1 §4.2, R4 §2.4).

Для длины окна L: доля отведений/записей, у которых размеченная зона >= L, и доля всех
размеченных волн, попадающих целиком в окно L, поставленное по центру зоны.
Использование: python r4_window_length.py <папка LUDB>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from r1_ludb_stats import LEADS, parse  # noqa: E402


def main(root):
    waves, lead_zone, rec_zone = {}, {}, {}
    for r in map(str, range(1, 201)):
        firsts, lasts = [], []
        for lead in LEADS:
            w = parse(os.path.join(root, 'data', r), lead)[0]
            waves[(r, lead)] = w
            if w:
                lead_zone[(r, lead)] = (w[0][1], w[-1][3])
                firsts.append(w[0][1]); lasts.append(w[-1][3])
        rec_zone[r] = (max(firsts), min(lasts))  # пересечение зон 12 отведений (режим M)
    span = np.array([b - a for a, b in lead_zone.values()])
    rspan = np.array([b - a for a, b in rec_zone.values()])
    print('lead zone, samples: min %d p5 %d p50 %d' % (span.min(), *np.percentile(span, [5, 50])))
    print('record zone (intersection): min %d p5 %d p50 %d' % (rspan.min(), *np.percentile(rspan, [5, 50])))
    total = sum(len(w) for w in waves.values())
    print('L | s | % leads with zone >= L | % waves inside centred window')
    for length in [2048, 2560, 3072, 3328, 3584, 4096, 5120]:
        fits, inside = 0, 0
        for k, (a, b) in lead_zone.items():
            c = (a + b) / 2
            lo, hi = c - length / 2, c + length / 2
            fits += (b - a) >= length
            inside += sum(t[1] >= lo and t[3] <= hi for t in waves[k])
        print(f'{length} | {length / 500:.2f} | {fits / len(lead_zone) * 100:5.1f} | {inside / total * 100:5.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
