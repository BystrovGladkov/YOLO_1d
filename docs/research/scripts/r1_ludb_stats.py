"""Статистика разметки LUDB для задачи R1 (см. docs/research/R1_ludb.md).

Использование:
    python docs/research/scripts/r1_ludb_stats.py <путь к папке LUDB с data/ и ludb.csv>

Требует пакет `wfdb`.
"""
import collections
import csv
import os
import sys

import numpy as np
import wfdb

LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FS = 500
WAVE = {'p': 'P', 'N': 'QRS', 't': 'T'}


def parse(path, lead):
    """Строгий разбор тройек '(' <peak> ')'. Возвращает волны и число нераспознанных символов."""
    ann = wfdb.rdann(path, lead)
    s, x = ann.symbol, ann.sample
    waves, bad, i = [], 0, 0
    while i < len(s):
        if s[i] == '(' and i + 2 < len(s) and s[i + 1] in WAVE and s[i + 2] == ')':
            waves.append((WAVE[s[i + 1]], x[i], x[i + 1], x[i + 2]))
            i += 3
        else:
            bad += 1
            i += 1
    return waves, bad


def anomaly_patterns(path, lead):
    """Классификация нарушений структуры '(' peak ')'."""
    s = wfdb.rdann(path, lead).symbol
    seq = ''.join({'(': '(', ')': ')'}.get(c, 'w') for c in s)
    cat, i = collections.Counter(), 0
    while i < len(seq):
        if seq[i:i + 3] == '(w)':
            i += 3
        elif seq[i:i + 2] == '()':
            cat['bracket pair without peak'] += 1; i += 2
        elif seq[i:i + 3] == 'w()':
            cat['peak before onset'] += 1; i += 3
        elif seq[i:i + 2] == 'w)':
            cat['missing onset'] += 1; i += 2
        elif seq[i:i + 3] == '(w(':
            cat['missing offset'] += 1; i += 2
        else:
            cat['other'] += 1; i += 1
    return cat


def main(root):
    data = os.path.join(root, 'data')
    rows = {r['ID']: r for r in csv.DictReader(open(os.path.join(root, 'ludb.csv'), encoding='utf-8'))}
    cnt, dur = collections.Counter(), collections.defaultdict(list)
    first, last, rec_first, rec_last = [], [], [], []
    count_diff, spread = collections.Counter(), collections.defaultdict(list)
    anomalies, bad_rec = collections.Counter(), collections.Counter()
    first_type, last_type = collections.Counter(), collections.Counter()
    overlap, all_waves = 0, {}

    for r in map(str, range(1, 201)):
        path = os.path.join(data, r)
        per = {}
        for lead in LEADS:
            w, b = parse(path, lead)
            per[lead] = w
            if b:
                bad_rec[r] += b
                anomalies.update(anomaly_patterns(path, lead))
            for c, on, _, off in w:
                cnt[c] += 1
                dur[c].append((off - on) * 1000 / FS)
            if w:
                first.append(w[0][1] / FS); last.append(w[-1][3] / FS)
                first_type[w[0][0]] += 1; last_type[w[-1][0]] += 1
            overlap += sum(b_[1] < a_[3] for a_, b_ in zip(w, w[1:]))
        all_waves[r] = per
        rec_first.append(min(w[0][1] for w in per.values() if w) / FS)
        rec_last.append(max(w[-1][3] for w in per.values() if w) / FS)
        for c in WAVE.values():
            n = [sum(t[0] == c for t in per[lead]) for lead in LEADS]
            count_diff[(c, max(n) - min(n) > 0)] += 1
            # разброс границ одной и той же волны между 12 отведениями (сопоставление по пику, окно 150 мс)
            for t in (t for t in per['ii'] if t[0] == c):
                ons, offs = [], []
                for lead in LEADS:
                    cand = [u for u in per[lead] if u[0] == c and abs(u[2] - t[2]) < 75]
                    if cand:
                        u = min(cand, key=lambda u: abs(u[2] - t[2]))
                        ons.append(u[1]); offs.append(u[3])
                if len(ons) == 12:
                    spread[c].append((np.ptp(ons), np.ptp(offs)))

    print('waves parsed:', sum(cnt.values()), dict(cnt))
    print('malformed symbols per record:', dict(bad_rec))
    print('anomaly patterns:', dict(anomalies))
    for c, v in dur.items():
        v = np.array(v)
        print(f'{c} duration ms: mean {v.mean():.0f} sd {v.std():.0f} min {v.min():.0f} '
              f'p1 {np.percentile(v, 1):.0f} p50 {np.median(v):.0f} p99 {np.percentile(v, 99):.0f} max {v.max():.0f}')
    f, l = np.array(first), np.array(last)
    print(f'per-lead first onset s: min {f.min():.2f} p50 {np.median(f):.2f} p90 {np.percentile(f, 90):.2f} max {f.max():.2f}')
    print(f'per-lead last offset s: min {l.min():.2f} p10 {np.percentile(l, 10):.2f} p50 {np.median(l):.2f} max {l.max():.2f}')
    span = l - f
    print(f'per-lead annotated span s: min {span.min():.2f} p5 {np.percentile(span, 5):.2f} p50 {np.median(span):.2f}')
    print('first wave type:', dict(first_type), 'last wave type:', dict(last_type))
    print('overlapping consecutive waves in one lead:', overlap)
    print('records where wave count differs between leads (class, differs):', sorted(count_diff.items()))
    for c, v in spread.items():
        v = np.array(v) * 1000 / FS
        print(f'{c}: n={len(v)} max-min over 12 leads, ms: onset p50 {np.median(v[:, 0]):.0f} '
              f'p90 {np.percentile(v[:, 0], 90):.0f}; offset p50 {np.median(v[:, 1]):.0f} p90 {np.percentile(v[:, 1], 90):.0f}')
    no_p = [r for r in all_waves if not any(t[0] == 'P' for t in all_waves[r]['ii'])]
    print('records without P in lead ii:', len(no_p),
          dict(collections.Counter(rows[r]['Rhythms'].strip().replace('\n', '; ') for r in no_p)))
    print('paced records:', [r for r in rows if rows[r]['Cardiac pacing'].strip()])
    print('rhythms:', dict(collections.Counter(rows[r]['Rhythms'].strip().replace('\n', '; ') for r in rows)))
    q = np.array([sum(t[0] == 'QRS' for t in all_waves[r]['ii']) for r in all_waves])
    print(f'QRS per record (lead ii): min {q.min()} p50 {np.median(q):.0f} max {q.max()}')


if __name__ == '__main__':
    main(sys.argv[1])
