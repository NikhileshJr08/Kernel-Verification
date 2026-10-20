#!/usr/bin/env python3
"""Run one recovered ABC verified test against its exact original CUDA source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('row_id', type=int, help='Sakana positional row ID, e.g. 3492')
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--out', type=Path, default=ROOT/'results/kernelbench-audit-local.json')
    args = ap.parse_args()
    records = json.loads((ROOT/'extensions/manifest.json').read_text())['records']
    c = next((r for r in records if r['collection']=='kernelbench-audit' and r['dataset_row']==args.row_id), None)
    if c is None: ap.error('No verified test for that row ID; see extensions/kernelbench-audit/INDEX.md')
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU and nvcc required')
    except (ImportError, RuntimeError) as exc:
        print(json.dumps({'status':'environment_error','error':str(exc)}))
        return 2
    with tempfile.TemporaryDirectory(prefix='kernelbench_audit_') as tmp:
        work = Path(tmp)
        cp = ROOT/c['path']
        relative_test = Path('tests_verified/level_1')/cp.name/f'test_{args.row_id}.py'
        (work/relative_test).parent.mkdir(parents=True)
        (work/'kernels').mkdir()
        shutil.copyfile(cp/'candidate.cu', work/'kernels'/f'{args.row_id}.cu')
        shutil.copyfile(cp/'test_verified.py', work/relative_test)
        shutil.copyfile(ROOT/'extensions/kernelbench-audit/upstream/utils.py', work/'utils.py')
        env = os.environ.copy(); env['KERNEL_ID'] = str(args.row_id)
        env['MAX_JOBS'] = '4'
        cmd = [sys.executable,'-m','pytest',str(relative_test),'--timeout=120',
               '--json-report','--json-report-file=pytest-report.json','-q']
        try:
            result = subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True,timeout=args.timeout)
            report = work/'pytest-report.json'
            data = {'case_id':c['id'],'exit_code':result.returncode,
                    'pytest_report':json.loads(report.read_text()) if report.exists() else None,
                    'stdout':result.stdout[-12000:],'stderr':result.stderr[-12000:],
                    'candidate_sha256':c['candidate_sha256'],
                    'note':'Inspect failed/skipped counts; a skip is not a passing correctness result.'}
        except subprocess.TimeoutExpired:
            data = {'case_id':c['id'],'status':'timeout','timeout_seconds':args.timeout}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(data,indent=2)+'\n')
        print(args.out)
        return data.get('exit_code',2)

if __name__=='__main__':
    raise SystemExit(main())
