#!/usr/bin/env python3
"""Run canonical VOLTA configurations against the original PTX corpus."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', help='V-case ID; use all-races or all-reductions for category runs')
    ap.add_argument('--iterations', type=int, default=1)
    ap.add_argument('--out-dir', type=Path, default=ROOT/'results/volta-local')
    ap.add_argument('--print-command', action='store_true')
    args = ap.parse_args()
    cases = json.loads((ROOT/'manifest.json').read_text())['cases']
    if args.case == 'all-reductions':
        codes = [subprocess.run([__import__('sys').executable, __file__, c['id'],
                               '--iterations', str(args.iterations),
                               *(['--print-command'] if args.print_command else [])]).returncode
                 for c in cases if c['id'].startswith(('V01', 'V02', 'V03'))]
        return max(codes)
    elif args.case == 'all-races':
        selector = ['category', 'race']
    else:
        c = next((c for c in cases if c['id']==args.case and c['id'].startswith('V')), None)
        if not c: ap.error('Select a V-case ID')
        if c['id'].startswith(('V01', 'V02', 'V03')):
            # Zenodo ships BLOCKSIZE=8 reductions, while current Rust benchmark
            # configs expect a later recompilation at 128. Analyze original PTX
            # directly with the artifact's specialization, without modifying it.
            ptx = ROOT/c['candidate']
            entry = re.search(r'\.entry\s+(\S+)\(', ptx.read_text()).group(1)
            n = 12 if c['id'].startswith('V01') else 8
            cmd = ['cargo', 'run', '--release', '-p', 'volta_cli', '--', 'analyze', str(ptx),
                   '-k', entry, '-b', '8', '-g', '1',
                   '--array', f'in:0x10000:4:{n}:in', '--array', 'out:0x20000:4:1:out',
                   '--param', 'ptr:in', '--param', 'ptr:out', '--dyn-shared', '2048']
            if args.print_command:
                print(json.dumps({'cwd':str(ROOT/'vendor/volta'), 'argv':cmd,
                                  'note':'Original 8-thread artifact; not the later 128-thread timing configuration. --iterations applies only to benchmark-harness runs.'}, indent=2))
                return 0
            return subprocess.run(cmd, cwd=ROOT/'vendor/volta').returncode
        selector = ['single', c['baseline_selector']]
    cmd = ['cargo', 'run', '--release', '-p', 'volta_bench', '--',
           '--kernels-dir', str(ROOT/'vendor/volta-benchmarks'),
           '--iterations', str(args.iterations), '--out-dir', str(args.out_dir.resolve()),
           'generate', *selector]
    if args.print_command:
        print(json.dumps({'cwd':str(ROOT/'vendor/volta'), 'argv':cmd}, indent=2))
        return 0
    return subprocess.run(cmd, cwd=ROOT/'vendor/volta').returncode

if __name__ == '__main__':
    raise SystemExit(main())
