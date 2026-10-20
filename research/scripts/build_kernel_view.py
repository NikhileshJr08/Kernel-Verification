#!/usr/bin/env python3
"""Build the simple source-only browsing layout using relative symlinks."""
import json
import os
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent


def link(source, destination):
    assert source.exists(), source
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink():
        assert os.readlink(destination) == target, destination
    else:
        assert not destination.exists(), destination
        destination.symlink_to(target, target_is_directory=source.is_dir())


def main():
    core = json.loads((RESEARCH / 'manifest.json').read_text())['cases']
    extra = json.loads((RESEARCH / 'extensions/manifest.json').read_text())['records']
    groups = {}
    for record in core + extra:
        group = record.get('collection') or ('drkernel' if record['id'].startswith('D') else 'volta')
        if group == 'volta-bounds':
            continue  # No complete implementation recovered.
        folder = ROOT / 'kernels' / group / record['id']
        sources = []
        note = 'Reference and candidate source.'
        if group == 'sol-execbench':
            sources = [('candidate', RESEARCH / record['path'] / 'candidate')]
            note = 'Complete multi-file candidate bundle. Reference requires the separate NVIDIA download; see evidence.'
        elif group == 'gimlet' and record['id'] == 'G01':
            sources = [(name, RESEARCH / 'extensions/gimlet/G01' / name)
                       for name in ('reference.py', 'candidate.py', 'run.py')]
            note = 'Newly written reconstruction of the blog example; not recovered original LLM code. Run python3 run.py in this folder.'
        else:
            for role in ('reference', 'candidate'):
                path = record.get('source_' + role) or record.get(role)
                if isinstance(path, str) and (RESEARCH / path).is_file():
                    source = RESEARCH / path
                    sources.append((role + source.suffix, source))
            if group == 'gimlet':
                note = 'Public reference task only; original generated candidate not recovered.'
            elif group in ('megatron', 'flashattention', 'volta'):
                note = 'Historical implementations: candidate is faulty/pre-fix; reference is baseline/fixed. These are not labeled LLM-generated.'
            if record['id'] == 'T674_fp32_issue_reproducer':
                sources = [('reproducer.py', RESEARCH / record['candidate'])]
                note = 'Combined source and test with an embedded PyTorch reference; not a separate recovered pair.'
            if group == 'megatron':
                base = RESEARCH / record['path']
                for rev in ('pre', 'post'):
                    sources.append((rev + '_binding.cpp', base / rev / 'layer_norm_cuda.cpp'))
        if not sources:
            continue
        for name, source in sources:
            link(source, folder / name)
        fallback = ('extensions/gimlet/' if group == 'gimlet' else 'cases/') + record['id']
        origin = RESEARCH / (record.get('path') or fallback)
        link(origin, folder / 'evidence')
        (folder / 'README.md').write_text(
            f'# {record["id"]}\n\n{note}\n\n'
            + '\n'.join(f'- [{name}]({name})' for name, _ in sources)
            + '\n\n[evidence/](evidence/) contains the original metadata, attribution, tests, and supporting files.\n'
            + 'Source entries are links to the preserved files under research/; edits change those files directly.\n'
        )
        groups.setdefault(group, []).append((record['id'], note))
    lines = ['# Kernel implementations', '',
             'Pick a collection, then a case. Each case exposes reference and candidate sources directly.', '',
             'Some candidates require a CUDA toolchain, dependencies, or the original harness. This layout is for finding source; it does not promise standalone execution.', '',
             '| Collection | Cases |', '|---|---:|']
    for group, cases in sorted(groups.items()):
        lines.append(f'| [{group}]({group}/README.md) | {len(cases)} |')
        content = [f'# {group}', '', '| Case | Source status |', '|---|---|']
        content += [f'| [{case}]({case}/README.md) | {note} |' for case, note in sorted(cases)]
        (ROOT / 'kernels' / group / 'README.md').write_text('\n'.join(content) + '\n')
    (ROOT / 'kernels/README.md').write_text('\n'.join(lines) + '\n')
    count = sum(map(len, groups.values()))
    print(f'Created source view for {count} case records across {len(groups)} collections.')
    for path in (ROOT / 'kernels').rglob('*'):
        if path.is_symlink():
            assert path.exists(), f'Broken link: {path}'


if __name__ == '__main__':
    main()
