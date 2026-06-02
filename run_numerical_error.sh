#!/usr/bin/env bash

cd /workspace/leet/LeetDecoding
python -m leetDecoding.test.test_error \
  --methods BCMV_vanilla FleetAttention FleetAttention_torch lightningAttention2_optimized lightningAttention2_torch causal_dot_product causal_dot_product_torch recursion blockbased \
  --dtypes float32 bfloat16 \
  --n 1024 \
  --trials 64 \
  --output-path ./outputs/error_summary_no_decay.json