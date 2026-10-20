#!/usr/bin/env python3
"""Direct local download of pinned SOL references; excluded from distribution."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='all', help='Full problem name or all')
    ap.add_argument('--revision', choices=['current','cursor-era'], default='current')
    ap.add_argument('--accept-nvidia-evaluation-license', action='store_true',
                    help='Confirm you accept the NVIDIA Evaluation Dataset License; evaluation-only, no redistribution or training use')
    ap.add_argument('--out', type=Path, default=ROOT/'local-data/sol-execbench')
    args=ap.parse_args()
    if not args.accept_nvidia_evaluation_license:
        ap.error('Read extensions/sol-execbench/NVIDIA_EVALUATION_LICENSE.txt, then explicitly accept its terms with --accept-nvidia-evaluation-license')
    import pyarrow.parquet as pq
    lock=json.loads((ROOT/'extensions/upstream-lock.json').read_text())
    records=[r for r in json.loads((ROOT/'extensions/manifest.json').read_text())['records'] if r['collection']=='sol-execbench']
    if args.case!='all': records=[r for r in records if r['name']==args.case]
    if not records: ap.error('Unknown SOL case')
    rev=lock['revisions']['sol_current' if args.revision=='current' else 'sol_cursor_era']
    cache={}
    for c in records:
        subset=c['subset']
        if args.revision=='cursor-era' and c['original_reference_revision'] is None:
            print('SKIP: no cursor-era pin recorded for',c['name']); continue
        if subset not in cache:
            url=f'https://huggingface.co/datasets/nvidia/SOL-ExecBench/resolve/{rev}/data/{subset}.parquet'
            with urllib.request.urlopen(url,timeout=120) as response: payload=response.read()
            key=f'sol-{subset}.parquet' if args.revision=='current' else f'sol-original-{subset}.parquet'
            assert hashlib.sha256(payload).hexdigest()==lock['downloaded_source_sha256'][key], 'Dataset bytes differ from recorded acquisition'
            cache[subset]={r['name']:r for r in pq.read_table(io.BytesIO(payload)).to_pylist()}
        row=cache[subset][c['name']]
        expected=c['reference_sha256'] if args.revision=='current' else c['original_reference_sha256']
        assert hashlib.sha256(row['reference'].encode()).hexdigest()==expected
        d=args.out/rev/subset/c['name'];d.mkdir(parents=True,exist_ok=True)
        (d/'reference.py').write_text(row['reference'])
        definition={'name':row['name'],'hf_id':row.get('hf_id'),'description':row['description'],
                    'axes':json.loads(row['axes']),'inputs':json.loads(row['inputs']),
                    'outputs':json.loads(row['outputs']),'custom_inputs_entrypoint':row.get('custom_inputs_entrypoint'),
                    'reference':row['reference']}
        (d/'definition.json').write_text(json.dumps(definition,indent=2)+'\n')
        workloads=json.loads(row['workloads'])
        (d/'workload.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in workloads))
        source=ROOT/c['path']/'solution.json'
        (d/'solution.json').write_bytes(source.read_bytes())
        print(d)

if __name__=='__main__':main()
