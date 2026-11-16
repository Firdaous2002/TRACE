"""Summarize finished TRACE runs: mean ± std over seeds per dataset/model/task/method.
Usage: python ours/analysis/summarize.py [results_dir] [--csv out.csv]"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith('--')]
res_dir = args[0] if args else 'results'
csv_out = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None

PAT = re.compile(r'explain_(sasrec|bert4rec|gru4rec)_(.+)_(beam|ga|greedy|random|gradient)_'
                 r'(targeted_r\d+_\d+|pushout_top\d+)_seed(\d+)\.json$')
NAME = {'beam': 'TRACE', 'ga': 'GECE', 'greedy': 'Greedy', 'random': 'Random', 'gradient': 'Gradient'}

rows, groups = [], defaultdict(list)
for f in sorted(glob.glob(os.path.join(res_dir, 'explain_*.json'))):
    m = PAT.search(os.path.basename(f))
    if not m:
        continue
    model, ds, meth, task, seed = m.groups()
    d = json.load(open(f))
    mt, pu = d.get('metrics', {}), d.get('per_user', [])
    hides = [u['hides_competitor'] for u in pu if 'hides_competitor' in u]
    row = dict(dataset=ds, model=model, task=task, method=NAME[meth], seed=int(seed), users=len(pu),
               validity=mt.get('validity'), queries=mt.get('queries'), hamming=mt.get('hamming_mean'),
               dnll=mt.get('delta_nll'), time_s=mt.get('time_seconds'),
               hides_comp=(np.mean(hides) if hides else None))
    rows.append(row)
    groups[(ds, model, task, row['method'])].append(row)

def ms(vals, fmt):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return '-'
    return (fmt % np.mean(v)) + ('±' + fmt % np.std(v) if len(v) > 1 else '')

print(f"{len(rows)} result files in {res_dir}\n")
hdr = f"{'dataset':8} {'model':9} {'task':16} {'method':7} {'seeds':5} {'validity':>13} {'queries':>15} {'hamming':>11} {'ΔNLL':>13} {'hides':>6}"
print(hdr); print('-' * len(hdr))
order = {'TRACE': 0, 'GECE': 1, 'Greedy': 2}
for k in sorted(groups, key=lambda k: (k[0], k[1], k[2], order.get(k[3], 9))):
    g = groups[k]
    print(f"{k[0]:8} {k[1]:9} {k[2]:16} {k[3]:7} {len(g):5} "
          f"{ms([r['validity'] for r in g], '%.3f'):>13} {ms([r['queries'] for r in g], '%.0f'):>15} "
          f"{ms([r['hamming'] for r in g], '%.2f'):>11} {ms([r['dnll'] for r in g], '%.3f'):>13} "
          f"{ms([r['hides_comp'] for r in g], '%.2f'):>6}")

if csv_out:
    import csv
    with open(csv_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nper-run rows written to {csv_out}")
