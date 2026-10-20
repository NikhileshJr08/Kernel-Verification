#!/usr/bin/env python3
"""Replay the vendored Lethe audit on ONE original pair in a fresh process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', help='D-case ID, or UUID such as 1088')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    matches = [c for c in json.loads((ROOT/'manifest.json').read_text())['cases']
               if c['id'] == args.case or str(c.get('uuid')) == args.case]
    if len(matches) != 1 or not matches[0]['id'].startswith('D'):
        ap.error('Select one Triton D-case from manifest.json')
    c = matches[0]
    if not args.worker:
        start = time.perf_counter()
        cmd = [sys.executable, __file__, c['id'], '--device', args.device, '--worker']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            try:
                result = json.loads(r.stdout)
            except json.JSONDecodeError:
                result = {'status': 'worker_error', 'stderr': r.stderr[-6000:]}
            result['worker_exit_code'] = r.returncode
        except subprocess.TimeoutExpired:
            result = {'status': 'timeout', 'timeout_seconds': args.timeout}
        result.update(case=c['id'], wall_seconds_including_startup=time.perf_counter()-start,
                      device=args.device, candidate_sha256=c['sha256']['candidate'])
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text+'\n')
        return 0 if result.get('status') == 'gated' else 2
    # Optional dependencies are loaded only inside the worker.
    try:
        import torch
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; CPU fallbacks cannot validate Triton kernels')
        sys.path.insert(0, str(ROOT/'vendor/lethe/src'))
        from lethe.verifier.audit_harness import audit_worker
        result = audit_worker((ROOT/c['reference']).read_text(),
                              (ROOT/c['candidate']).read_text(), {'device': args.device})
        result['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(torch.device(args.device))}
        import triton
        result['environment']['triton'] = triton.__version__
        print(json.dumps(result))
        return 0
    except Exception as e:
        print(json.dumps({'status': 'environment_or_worker_error', 'error': f'{type(e).__name__}: {e}'}))
        return 2

if __name__ == '__main__':
    sys.exit(main())
