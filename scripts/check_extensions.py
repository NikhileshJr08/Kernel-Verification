#!/usr/bin/env python3
"""Offline validation of the specifically requested source collections."""
import ast
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    m=json.loads((ROOT/'extensions/manifest.json').read_text());rows=m['records']
    assert len(rows)==m['record_count']==162
    assert len({r['id'] for r in rows})==162
    counts=collections.Counter(r['collection'] for r in rows)
    assert dict(counts)==m['collection_counts']
    files_checked=0
    for r in rows:
        if r.get('candidate_sha256'):
            p=ROOT/r['candidate'];assert sha(p)==r['candidate_sha256'],p;files_checked+=1
            if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        if r.get('reference_sha256') and r.get('reference'):
            p=ROOT/r['reference'];assert sha(p)==r['reference_sha256'],p;files_checked+=1
            if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        if r['collection']=='kernelbench-audit':
            d=ROOT/r['path'];e=json.loads((d/'original_result.json').read_text())
            assert e['Correct'] is True and e['dataset_row_position']==r['dataset_row']
            assert r['candidate_matches_abc_bytes'] is True
            ast.parse((d/'test_verified.py').read_text())
        elif r['collection']=='sol-execbench':
            d=ROOT/r['path'];solution=json.loads((d/'solution.json').read_text())
            assert solution['definition']==r['name']
            for s in solution['sources']:
                p=d/'candidate'/s['path'];assert p.read_text()==s['content']
                assert sha(p)==r['candidate_source_hashes'][str(p.relative_to(ROOT))];files_checked+=1
                if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
            evidence=json.loads((d/'published_evaluations.json').read_text())
            assert all('workload' not in x for x in evidence)
            assert all(x['evaluation']['status']=='PASSED' for x in evidence)
            assert not (d/'reference.py').exists()
            assert not (d/'workload.jsonl').exists()
        elif r['collection'] in ['flashattention','megatron'] and r.get('pre_revision'):
            d=ROOT/r['path'];commit=json.loads((d/'commit.json').read_text())
            assert commit['sha']==r['fixed_revision']
            assert commit['parents'][0]['sha']==r['pre_revision']
            if r['collection']=='flashattention':
                ast.parse((d/'pre/tests/test_flash_attn.py').read_text())
                ast.parse((d/'post/tests/test_flash_attn.py').read_text())
    for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    assert not list((ROOT/'extensions').rglob('*.parquet'))
    result={'status':'extensions_integrity_pass','records':len(rows),'collection_counts':dict(counts),
            'hashed_source_files_checked':files_checked,'nvidia_reference_data_embedded':False,
            'gpu_executed':False,'source_revision_and_test_presence_validation_only':True}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
