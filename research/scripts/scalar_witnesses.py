#!/usr/bin/env python3
"""Source-derived scalar checks, NOT verification of CUDA/Triton lowering."""
import json
import math
from decimal import Decimal, localcontext

def main():
    results = []
    # Actual D1088 source: C=10 and minimum autotune tile BLOCK_C=16.
    # Zero logits -> ten probabilities 1/10 and six masked probabilities 0.
    # The second exp/sum forgets the mask, adding six exp(0) terms.
    with localcontext() as ctx:
        ctx.prec = 70
        e = (Decimal(1) / 10).exp()
        ref_s = 10 * e
        cand_s = ref_s + 6
        ref = (ref_s**2 - 1) / (ref_s**2 + 1)
        cand = (cand_s**2 - 1) / (cand_s**2 + 1)
        assert cand > ref
        results.append({'case': 'D1088', 'kind': 'source_derived_exact_real_formula',
                        'input': 'all zero logits; C=10; BLOCK_C=16',
                        'reference': str(ref), 'candidate': str(cand),
                        'absolute_difference': str(cand - ref),
                        'passes_scalar_allclose_atol_rtol_1e_2': abs(cand-ref) <= Decimal('0.01') + Decimal('0.01')*abs(ref)})
    # Supplemental Gimlet reconstructions: independent witnesses.
    def softmax(x):
        es = [math.exp(v - max(x)) for v in x]
        return [v / sum(es) for v in es]
    a, b = softmax([-5., -5.]), softmax([-6., -7.])
    assert a != b
    results.append({'case': 'G01_missing_attention_clamp', 'reference': a, 'candidate': b})
    # Config bug under explicit positive scaling contract.
    x, s = 0.75, 1.0
    ref = min(max(min(max(x, 0), 1) * s, 0), 1) / s
    cand = min(max(x, 0), 0.5)
    assert ref != cand
    results.append({'case': 'G02_hardcoded_scaling', 'x': x, 'scaling_factor': s,
                    'reference': ref, 'candidate': cand})
    print(json.dumps({'status': 'scalar_witnesses_pass', 'gpu_executed': False,
                      'scope': 'scalar formulas only, not compiled GPU programs',
                      'results': results}, indent=2))

if __name__ == '__main__':
    main()
