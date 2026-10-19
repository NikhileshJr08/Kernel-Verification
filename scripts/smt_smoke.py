#!/usr/bin/env python3
"""Small real-arithmetic obligations; deliberately no claim of full-kernel proof."""
import json
import z3

def main():
    out = []
    # D1088: substitute E = exp(1/10) > 1; exp is not approximated by Z3.
    e = z3.Real('exp_one_tenth')
    sr, sc = 10 * e, 10 * e + 6
    # Equal tanh(log(S)) outputs iff the cross-multiplied fractions agree.
    s = z3.Solver(); s.set(timeout=10000)
    s.add(e > 1, (sr*sr - 1)*(sc*sc + 1) == (sc*sc - 1)*(sr*sr + 1))
    verdict = s.check()
    assert verdict == z3.unsat
    out.append({'case': 'D1088', 'query': 'equal_outputs_under_E_gt_1', 'result': str(verdict),
                'meaning': 'inequivalent for all E>1, under the source-derived scalar model'})
    # Supplemental folded clamp incorrectly assumes s=2.
    x, scale = z3.Reals('x scale')
    clamp = lambda t, lo, hi: z3.If(t < lo, lo, z3.If(t > hi, hi, t))
    reference = clamp(clamp(x, 0, 1)*scale, 0, 1)/scale
    candidate = clamp(x, 0, z3.RealVal('1/2'))
    s = z3.Solver(); s.set(timeout=10000)
    s.add(scale == 1, x == z3.RealVal('3/4'), reference != candidate)
    verdict = s.check(); assert verdict == z3.sat
    out.append({'case': 'G02', 'query': 'reference_ne_candidate', 'result': str(verdict), 'model': str(s.model())})
    # Gimlet's recovered FP32 reciprocal interpreted as an exact real coefficient.
    # This is not an IEEE-754 execution proof.
    x = z3.Real('x')
    ref = clamp((x + 3)/6, 0, 1)
    cand = clamp((x + 3)*z3.RealVal('11184811/67108864'), 0, 1)
    s = z3.Solver(); s.set(timeout=10000)
    s.add(x == 0, ref != cand)
    verdict = s.check(); assert verdict == z3.sat
    out.append({'case': 'G03', 'query': 'real_coefficient_models_differ', 'result': str(verdict),
                'warning': 'At x=0 the actual rounded FP32 outputs can agree; this witness belongs only to the exact-real coefficient model.'})
    print(json.dumps({'z3_version': z3.get_version_string(), 'scope': 'scalar_models_only', 'results': out}, indent=2))

if __name__ == '__main__':
    main()
