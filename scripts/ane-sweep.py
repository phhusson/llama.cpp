#!/usr/bin/env python3
"""Compare warm ANE (including Metal conversions/fences) and Metal using test-backend-ops.

Example: python3 scripts/ane-sweep.py --output build/ane-sweep --repeats 2
Measurements exclude model compilation. Each backend runs alone; no concurrent GPU load.
Repeated tensors can stay in cache; confirm thresholds with a model benchmark.
Joules per matmul cover modeled chip and DRAM energy across all processes, not wall energy.
Use --objective energy or --objective edp to fit for energy or energy-delay product.
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
import socket
import statistics
import subprocess
import sys
import time


def measure(cmd, env, log, timeout, meter):
    parent, child = socket.socketpair()
    with parent, child:
        parent.settimeout(timeout)
        run_env = dict(env, GGML_TEST_PERF_FD=str(child.fileno()))
        with subprocess.Popen(cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(child.fileno(),)) as process:
            child.close()
            try:
                deadline = time.monotonic() + timeout
                if parent.recv(1) != b'B':
                    raise RuntimeError('Missing perf measurement hook; rebuild test-backend-ops')
                before = meter.read()
                start = time.monotonic()
                parent.sendall(b'B')
                parent.settimeout(max(0.001, deadline - time.monotonic()))
                if parent.recv(1) != b'E':
                    raise RuntimeError('Benchmark ended before the final energy sample')
                after = meter.read()
                elapsed = time.monotonic() - start
                parent.sendall(b'E')
                status = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if status:
                    raise subprocess.CalledProcessError(status, cmd)
            except BaseException:
                process.kill()
                process.wait()
                raise
    if before.keys() != after.keys() or any(after[k] < before[k] for k in before):
        raise RuntimeError('Energy counters changed or reset during the benchmark')
    joules = sum(after[k] - before[k] for k in before)
    return dict(before=before, after=after, joules=joules, elapsed_s=elapsed)


def summarize(rows, thresholds=None, objective="speed"):
    energy_complete = bool(rows) and all(row.get("metal_j") not in (None, "") and row.get("ane_j") not in (None, "") for row in rows)
    if objective != "speed" and not energy_complete:
        raise ValueError("energy objectives require joule measurements for every sample")
    groups = {}
    for row in rows:
        groups.setdefault((row['type'], row['m'], row['n'], row['k']), []).append(row)
    cases = []
    for group in groups.values():
        case = dict(group[0])
        for key in (('metal_us', 'ane_us', 'metal_j', 'ane_j') if energy_complete else ('metal_us', 'ane_us')):
            case[key] = statistics.median(float(row[key]) for row in group)
        for key in ('macs', 'intensity'):
            case[key] = float(case[key])
        cases.append(case)
    macs = sorted({0, *(c['macs'] for c in cases), math.nextafter(max(c['macs'] for c in cases), math.inf)})
    intensities = sorted({0, *(c['intensity'] for c in cases), math.nextafter(max(c['intensity'] for c in cases), math.inf)})
    def cost(case, backend):
        if objective == 'speed':
            return case[f'{backend}_us']
        if objective == 'energy':
            return case[f'{backend}_j']
        return case[f'{backend}_j'] * case[f'{backend}_us']

    best = None
    candidates = [thresholds] if thresholds is not None else itertools.product(macs, intensities)
    for min_macs, min_intensity in candidates:
        selected = [c for c in cases if c['macs'] >= min_macs and c['intensity'] >= min_intensity]
        # Penalize regressions twice as much as equally large gains.
        deltas = [cost(c, 'ane') - cost(c, 'metal') for c in selected]
        score = sum(d if d < 0 else 2*d for d in deltas)
        if best is None or score < best['objective_score']:
            time_deltas = [c['ane_us'] - c['metal_us'] for c in selected]
            best = dict(min_macs=min_macs, min_intensity=min_intensity, objective=objective, objective_score=score,
                        objective_score_unit={'speed': 'us', 'energy': 'J', 'edp': 'J*us'}[objective],
                        score_us=sum(d if d < 0 else 2*d for d in time_deltas),
                        selected=len(selected), regressions=sum(d > 0 for d in deltas),
                        routed_us=sum(c['metal_us'] for c in cases) + sum(time_deltas),
                        speed_regressions=sum(d > 0 for d in time_deltas))
            if energy_complete:
                energy_deltas = [c['ane_j'] - c['metal_j'] for c in selected]
                best.update(routed_j=sum(c['metal_j'] for c in cases) + sum(energy_deltas),
                            energy_regressions=sum(d > 0 for d in energy_deltas))
    best.update(cases=len(cases), ane_wins=sum(c['ane_us'] < c['metal_us'] for c in cases),
                metal_us=sum(c['metal_us'] for c in cases),
                oracle_us=sum(min(c['ane_us'], c['metal_us']) for c in cases))
    if energy_complete:
        best.update(metal_j=sum(c['metal_j'] for c in cases),
                    oracle_j=sum(min(c['ane_j'], c['metal_j']) for c in cases),
                    ane_energy_wins=sum(c['ane_j'] < c['metal_j'] for c in cases))
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
    parser.add_argument('--objective', choices=['speed', 'energy', 'edp'], default='speed', help='fit for latency, joules, or energy-delay product; regressions carry twice the weight of gains')
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
        print(json.dumps(summarize(rows, thresholds, args.objective), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    samples = args.output / 'samples.csv'
    if samples.exists():
        parser.error(f'{samples} exists; use a fresh output directory')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import joules as meter
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
        energy_method='IOReport chip + DRAM counter delta around warm perf loop / total runs; includes other processes and loop overhead; excludes compilation and wall supply losses',
        energy_channels=list(meter.read()),
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
            energy = {}
            runs = {}
            for backend in (['ANE', 'MTL0'] if index % 2 else ['MTL0', 'ANE']):
                log_file = args.output / f'{stem}-{backend}.log'
                cmd = [str(args.build / 'bin/test-backend-ops'), 'perf', '-b', backend, '-o', 'MUL_MAT', '--test-file', str(test_file)]
                with log_file.open('w') as log:
                    energy[backend] = measure(cmd, env, log, args.timeout, meter)
                (args.output / f'{stem}-{backend}-energy.json').write_text(json.dumps(energy[backend], indent=2))
                match = re.search(r'(\d+) runs\s+-\s+([\d.]+) us/run', log_file.read_text())
                if not match:
                    raise RuntimeError(f'No timing in {log_file}')
                runs[backend] = int(match[1])
                timings[backend] = float(match[2])
            macs = m*n*k
            moved = w_row*m + (4*m*k if name != 'F16' else 0) + 8*n*k + 8*n*m + 8*n
            row = dict(type=name, m=m, n=n, k=k, repeat=repeat, macs=macs, bytes=moved, intensity=macs/moved,
                       metal_us=timings['MTL0'], ane_us=timings['ANE'], speedup=timings['MTL0']/timings['ANE'])
            for backend, prefix in [('MTL0', 'metal'), ('ANE', 'ane')]:
                row.update({f'{prefix}_j': energy[backend]['joules']/runs[backend],
                            f'{prefix}_total_j': energy[backend]['joules'],
                            f'{prefix}_runs': runs[backend],
                            f'{prefix}_elapsed_s': energy[backend]['elapsed_s'],
                            f'{prefix}_watts': energy[backend]['joules']/energy[backend]['elapsed_s']})
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f'{index + 1}/{len(cases)} {stem}: Metal {timings["MTL0"]:.2f} us / {row["metal_j"]:.6f} J, ANE {timings["ANE"]:.2f} us / {row["ane_j"]:.6f} J, speedup {row["speedup"]:.3f}', flush=True)
    summary = summarize(rows, thresholds, args.objective)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
