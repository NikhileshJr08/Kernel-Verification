#!/usr/bin/env python3
"""Reduced rounding witness for SOL L1/087, without NVIDIA reference code."""
import json
import struct

def bf16_round(x):
    bits=struct.unpack('>I',struct.pack('>f',x))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('>f',struct.pack('>I',bits))[0]

def main():
    sequential=0.0
    for _ in range(512): sequential=bf16_round(sequential+1.0)
    fp32_then_bf16=bf16_round(sum([1.0]*512))
    assert sequential==256.0 and fp32_then_bf16==512.0
    print(json.dumps({'status':'reduced_rounding_witness_pass',
                      'case':'S_L1_087_embedding_with_initial_layernorm_backward',
                      'contributions':512,'each_contribution':1.0,
                      'bf16_round_after_each_add':sequential,
                      'fp32_accumulation_then_bf16':fp32_then_bf16,
                      'scope':'accumulation mechanism only; not an execution of the original GPU kernel or a reproduction of the training experiment'},indent=2))

if __name__=='__main__':main()
