#!/usr/bin/env python3
"""Compare warm ANE (including Metal conversions/fences) and Metal using test-backend-ops.

Example: python3 scripts/ane-sweep.py --output build/ane-sweep --repeats 2
Measurements exclude model compilation. Each backend runs alone; no concurrent GPU load.
Repeated tensors can stay in cache; confirm thresholds with a model benchmark.
"""

import argparse
import csv
import ctypes
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess


def summarize(rows, thresholds=None):
    groups = {}
    for row in rows:
        groups.setdefault((row['type'], row['m'], row['n'], row['k']), []).append(row)
    cases = []
    for group in groups.values():
        case = dict(group[0])
        for key in ('metal_us', 'ane_us'):
            case[key] = statistics.median(float(row[key]) for row in group)
        for key in ('macs', 'intensity'):
            case[key] = float(case[key])
        cases.append(case)
    macs = sorted({0, *(c['macs'] for c in cases), math.nextafter(max(c['macs'] for c in cases), math.inf)})
    intensities = sorted({0, *(c['intensity'] for c in cases), math.nextafter(max(c['intensity'] for c in cases), math.inf)})
    best = None
    candidates = [thresholds] if thresholds is not None else itertools.product(macs, intensities)
    for min_macs, min_intensity in candidates:
        selected = [c for c in cases if c['macs'] >= min_macs and c['intensity'] >= min_intensity]
        # Penalize regressions twice as much as equally large gains.
        deltas = [c['ane_us'] - c['metal_us'] for c in selected]
        score = sum(d if d < 0 else 2*d for d in deltas)
        if best is None or score < best['score_us']:
            best = dict(min_macs=min_macs, min_intensity=min_intensity, score_us=score,
                        selected=len(selected), regressions=sum(d > 0 for d in deltas),
                        routed_us=sum(c['metal_us'] for c in cases) + sum(deltas))
    best.update(cases=len(cases), ane_wins=sum(c['ane_us'] < c['metal_us'] for c in cases),
                metal_us=sum(c['metal_us'] for c in cases),
                oracle_us=sum(min(c['ane_us'], c['metal_us']) for c in cases))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=Path('build'))
    parser.add_argument('--output', type=Path, default=Path('build/ane-sweep'))
    parser.add_argument('--types', default='Q4_K,Q6_K,Q8_0')
    parser.add_argument('--shapes', default='512x512,1024x1024,4096x4096,4096x12288,12288x4096,8192x4096', help='MxK pairs')
    parser.add_argument('--tokens', default='32,128,512,1024')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--analyze', type=Path, nargs='+', help='analyze existing samples.csv files without running benchmarks')
    parser.add_argument('--min-macs', type=float, help='evaluate a fixed policy instead of fitting thresholds')
    parser.add_argument('--min-intensity', type=float, help='MACs per estimated byte for the fixed policy')
    args = parser.parse_args()
    if (args.min_macs is None) != (args.min_intensity is None):
        parser.error('specify both --min-macs and --min-intensity')
    thresholds = None if args.min_macs is None else (args.min_macs, args.min_intensity)
    if thresholds and any(not math.isfinite(v) or v < 0 for v in thresholds):
        parser.error('thresholds must be finite and nonnegative')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.analyze:
        rows = []
        for path in args.analyze:
            with path.open() as f:
                rows.extend(csv.DictReader(f))
        print(json.dumps(summarize(rows, thresholds), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    samples = args.output / 'samples.csv'
    if samples.exists():
        parser.error(f'{samples} exists; use a fresh output directory')
    root = Path(__file__).resolve().parents[1]
    header = (root / 'ggml/include/ggml.h').read_text()
    header = re.sub(r'//[^\n]*|/\*.*?\*/', '', header, flags=re.S)
    types = {name: int(number) for name, number in re.findall(r'GGML_TYPE_(\w+)\s*=\s*(\d+)', header)}
    ops = re.findall(r'GGML_OP_\w+', re.search(r'enum ggml_op\s*\{(.*?)\};', header, re.S)[1])
    mul_mat = ops.index('GGML_OP_MUL_MAT')
    lib = ctypes.CDLL(str((args.build / 'bin/libggml-base.dylib').resolve()))
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_type_size.argtypes = [ctypes.c_int]
    lib.ggml_type_size.restype = ctypes.c_size_t
    quant_types = [name.strip().upper() for name in args.types.split(',')]
    shapes = [tuple(map(int, shape.split('x'))) for shape in args.shapes.split(',')]
    tokens = list(map(int, args.tokens.split(',')))
    cases = list(itertools.product(quant_types, shapes, tokens, range(args.repeats)))
    random.Random(1).shuffle(cases)
    env = dict(os.environ, GGML_ANE_MTLQUANTS=args.types, GGML_ANE_MIN_MACS='0', GGML_ANE_MIN_INTENSITY='0')
    (args.output / 'metadata.json').write_text(json.dumps(dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), platform=platform.platform(), hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        dirty=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        arguments={k: str(v) for k, v in vars(args).items()},
        method='test-backend-ops perf: warmup excluded, repeated operations, synchronized batch timing; ANE includes conversions and fences',
    ), indent=2))
    rows = []
    with samples.open('w') as f:
        writer = None
        for index, (name, (m, k), n, repeat) in enumerate(cases):
            t = types[name]
            w_row = lib.ggml_row_size(t, k)
            source_w = [t, k, m, 1, 1, lib.ggml_type_size(t), w_row, w_row*m, w_row*m]
            source_a = [types['F32'], k, n, 1, 1, 4, 4*k, 4*k*n, 4*k*n]
            stem = f'{name}-{m}-{n}-{k}-{repeat}'
            test_file = args.output / f'{stem}.txt'
            test_file.write_text(' '.join(map(str, [mul_mat, types['F32'], m, n, 1, 1, 0, 2, *source_w, *source_a, stem])) + '\n')
            timings = {}
            for backend in (['ANE', 'MTL0'] if index % 2 else ['MTL0', 'ANE']):
                log_file = args.output / f'{stem}-{backend}.log'
                cmd = [str(args.build / 'bin/test-backend-ops'), 'perf', '-b', backend, '-o', 'MUL_MAT', '--test-file', str(test_file)]
                with log_file.open('w') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                match = re.search(r'([\d.]+) us/run', log_file.read_text())
                if not match:
                    raise RuntimeError(f'No timing in {log_file}')
                timings[backend] = float(match[1])
            macs = m*n*k
            moved = w_row*m + (4*m*k if name != 'F16' else 0) + 8*n*k + 8*n*m + 8*n
            row = dict(type=name, m=m, n=n, k=k, repeat=repeat, macs=macs, bytes=moved, intensity=macs/moved,
                       metal_us=timings['MTL0'], ane_us=timings['ANE'], speedup=timings['MTL0']/timings['ANE'])
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f'{index + 1}/{len(cases)} {stem}: Metal {timings["MTL0"]:.2f} us, ANE {timings["ANE"]:.2f} us, speedup {row["speedup"]:.3f}', flush=True)
    summary = summarize(rows, thresholds)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
