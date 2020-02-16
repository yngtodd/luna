#!/bin/sh

# Test PyTorch RPC on a single host using torch.distributed.launch

python -m torch.distributed.launch \
    --nproc_per_node 3 \
    --use_env \
    domain_decomp_1d.py
