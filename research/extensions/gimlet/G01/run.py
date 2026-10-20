"""Compare reconstructed attention implementations on CPU or CUDA."""
import argparse
import math

import torch

from candidate import Model as Candidate
from reference import Model as Reference


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0 (no fallback)')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    reference, candidate = Reference().to(device), Candidate().to(device)

    # Small random inputs keep every score inside the clamp interval.
    q = torch.randn(2, 2, 8, 16, device=device) * 0.1
    k = torch.randn(2, 2, 8, 16, device=device) * 0.1
    v = torch.randn(2, 2, 8, 16, device=device)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    assert scores.abs().max().item() < 5.0
    expected, actual = reference(q, k, v), candidate(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f'PyTorch {torch.__version__}; device={device}')
    print(f'Small random inputs: PASS; max error={(actual - expected).abs().max().item():.8g}')

    # One query, two keys, head dimension 1: raw scores are [-6, -7].
    # Clamping makes both scores -5, so the reference averages values 0 and 1.
    q = torch.tensor([[[[1.0]]]], device=device)
    k = torch.tensor([[[[-6.0], [-7.0]]]], device=device)
    v = torch.tensor([[[[0.0], [1.0]]]], device=device)
    expected, actual = reference(q, k, v), candidate(q, k, v)
    torch.testing.assert_close(expected, torch.full_like(expected, 0.5))
    torch.testing.assert_close(actual, torch.full_like(actual, 1 / (1 + math.e)))
    assert not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    print('Counterexample scores: [-6, -7]; clamped scores: [-5, -5]')
    print(f'Reference output: {expected.item():.8f}')
    print(f'Candidate output: {actual.item():.8f}')
    print(f'Absolute error:   {(actual - expected).abs().item():.8f}')
    print('Missing clamp detected: PASS (the candidate is intentionally incorrect)')

    # Also exercise full batched, multi-head attention tensors. All 128 queries
    # have raw scores [-6, -7, -6, -7, ...], across 128 keys in each head.
    q = torch.zeros(2, 4, 128, 64, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[..., 0] = 8.0  # sqrt(64), so scaled dot products equal k[..., 0].
    k[..., 0::2, 0] = -6.0
    k[..., 1::2, 0] = -7.0
    v[..., 1::2, :] = 1.0
    expected, actual = reference(q, k, v), candidate(q, k, v)
    torch.testing.assert_close(expected, torch.full_like(expected, 0.5))
    torch.testing.assert_close(actual, torch.full_like(actual, 1 / (1 + math.e)))
    assert not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    print(f'Batched multi-head counterexample: Q/K/V shape={tuple(q.shape)}')
    print(f'Reference[0,0,0,0]: {expected[0,0,0,0].item():.8f}')
    print(f'Candidate[0,0,0,0]: {actual[0,0,0,0].item():.8f}')


if __name__ == '__main__':
    main()
