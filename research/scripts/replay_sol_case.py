#!/usr/bin/env python3
"""Run a locally downloaded problem with a selectable correctness seed."""
import argparse
import json
from pathlib import Path
import subprocess
import time

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('problem_dir',type=Path,help='Directory created by fetch_nvidia_references.py')
    ap.add_argument('--seed',type=int,default=200)
    ap.add_argument('--timeout',type=int,default=300)
    ap.add_argument('--print-command',action='store_true')
    args=ap.parse_args();d=args.problem_dir.resolve()
    for name in ['definition.json','workload.jsonl','solution.json']:
        if not (d/name).is_file():ap.error(f'Missing local file {name}; run fetch_nvidia_references.py first')
    cfg=d/f'config-seed-{args.seed}.json'
    cfg.write_text(json.dumps({'seed':args.seed,'warmup_runs':10,'iterations':50,'lock_clocks':False,'benchmark_reference':False},indent=2)+'\n')
    cmd=['sol-execbench','--definition',str(d/'definition.json'),'--workload',str(d/'workload.jsonl'),
         '--solution',str(d/'solution.json'),'--config',str(cfg),'--timeout',str(args.timeout),'--json']
    if args.print_command:print(json.dumps(cmd,indent=2));return 0
    start=time.perf_counter()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout+60)
        (d/f'results-seed-{args.seed}.jsonl').write_text(r.stdout)
        (d/f'stderr-seed-{args.seed}.log').write_text(r.stderr)
        print(json.dumps({'seed':args.seed,'exit_code':r.returncode,'wall_seconds':time.perf_counter()-start,'results':str(d/f'results-seed-{args.seed}.jsonl')},indent=2))
        return r.returncode
    except subprocess.TimeoutExpired:
        print(json.dumps({'seed':args.seed,'status':'timeout'}));return 2

if __name__=='__main__':raise SystemExit(main())
