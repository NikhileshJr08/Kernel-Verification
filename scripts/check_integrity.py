#!/usr/bin/env python3
"""Offline integrity/provenance checks; does not execute downloaded kernels."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    cases = manifest['cases']
    assert len(cases) == manifest['case_count'] == 20
    assert len({c['id'] for c in cases}) == 20
    total_files = 0
    for c in cases:
        assert json.loads((ROOT / 'cases' / c['id'] / 'case.json').read_text()) == c
        for key in ('reference', 'candidate'):
            p = ROOT / c[key]
            assert p.is_file(), p
            assert hashlib.sha256(p.read_bytes()).hexdigest() == c['sha256'][key], p
            if p.suffix == '.py':
                ast.parse(p.read_text(), filename=str(p))
            else:
                assert '.entry' in p.read_text(), p
            total_files += 1
        if c['id'].startswith('D'):
            d = ROOT / 'cases' / c['id']
            trajectory = json.loads((d / 'trajectory.json').read_text())
            passed = json.loads((d / 'numerical_pass.json').read_text())
            audit = json.loads((d / 'audit.json').read_text())
            assert str(trajectory['uuid']) == passed['id'] == audit['id']
            assert trajectory['best_round'] == c['round'] == 5
            assert passed['their_correct'] is True and passed['compiled'] is True
            assert audit['status'] == 'gated'
            assert any(g['status'] == 'fail' for g in audit['gates'].values())
            import re
            blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', trajectory['messages'][9]['content'], re.S)
            assert (d / 'candidate.py').read_text() == [b for b in blocks if 'class ModelNew' in b][-1]
            assert (d / 'reference.py').read_text() == trajectory['original_python_code']
        else:
            reports = json.loads((ROOT / c['failure_evidence']).read_text())
            assert any(r['record']['status'] == 'RACE' for r in reports), c['id']
            if 'pre-fix' in c['baseline_selector']:
                assert any(r['record']['status'] == 'OK' for r in reports), c['id']
    # Check all own Python entrypoints without importing optional dependencies.
    for p in (ROOT / 'scripts').glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    result = {'cases': len(cases), 'reference_candidate_files': total_files,
              'formal_race_reports': 8, 'numerical_pass_and_audit_failure_pairs': 12,
              'gpu_executed': False, 'status': 'integrity_pass'}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
