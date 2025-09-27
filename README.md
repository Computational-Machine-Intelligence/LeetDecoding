<!-- ## An efficient Linear Attention Decoding package -->

# LeetDecoding: A PyTorch Library for Exponentially Decaying Causal Linear Attention with CUDA Implementations


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Computational-Machine-Intelligence/LeetDecoding/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/LibMOON.svg)](https://pypi.org/project/leetDecoding/)
[![Paper](https://img.shields.io/badge/arxiv-paper-blue)](https://arxiv.org/abs/2501.02573) 




``LeetDecoding`` is an open-source library built for efficient Linear Attention Decoding. 

### 1. Installation
#### 1. From Pypi 
```bash
conda create -n leetDecoding python==3.9
conda activate leetDecoding
pip install leetDecoding
```

The code has been test under the following environment:
```python
triton>=2.1.0
torch>=2.1.0
pycuda
pynvml
numpy<2
```
You can use the following command to install:
```python
pip install triton
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pycuda
pip install pynvml
pip install numpy
```
#### 2. From SourceCode
```python
conda create -n leetDecoding python==3.9
conda activate leetDecoding
python setup.py develop
```

### 2. usage

```python
import torch
from leetDecoding.efficient_linear_decoding import causal_linear_decoder

torch.cuda.set_device('cuda:0')

# Create input tensor
Q = torch.randn(2,32,1024,128,device='cuda:0')
K = torch.randn(2,32,1024,128,device='cuda:0')
V = torch.randn(2,32,1024,128,device='cuda:0')

# Inference using causal_linear_decoder
output = causal_linear_decoder(Q,K,V)

# If you want to input a mask with weight that values are exp(-gamma), set the is_mask_weight: True and is_need_exp:True
gamma = torch.full((32,),0.5,device='cuda:0')
output = causal_linear_decoder(Q,K,V,is_mask_weight=True,gamma=gamma,is_need_exp=True)

# If you just want to input a mask with weight, set the is_mask_weight: True and is_need_exp:False
gamma = torch.full((32,),0.5,device='cuda:0')
output = causal_linear_decoder(Q,K,V,is_mask_weight=True,gamma=gamma,is_need_exp=False)

# If you want to use a specified methods, such as FleetAttention, set the attn-method: 'FleetAttention'
gamma = torch.full((32,),0.5,device='cuda:0')
output = causal_linear_decoder(Q,K,V,is_mask_weight=False,attn_method='FleetAttention')

```


### 3. acknowledgement
|method|Title|Paper|Code|
|---|---|---|---|
|causal_dot_product|Fast Transformers with Clustered Attention|[arxiv](https://arxiv.org/abs/2007.04825) |[code](https://github.com/idiap/fast-transformers/tree/master/fast_transformers/causal_product)|
|Lighting Attention-2|Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models|[arxiv](https://arxiv.org/abs/2401.04658)|[code](https://github.com/OpenNLPLab/lightning-attention)
|block-based| Transformer-VQ: Linear-Time Transformers via Vector Quantization|[arxiv](https://arxiv.org/abs/2309.16354) | [code](https://github.com/transformer-vq/transformer_vq)|
|recursion| HyperAttention: Long-context Attention in Near-Linear Time|[arxiv](https://arxiv.org/abs/2310.05869) | [code](https://github.com/insuhan/hyper-attn)|
|causal_dot_product_torch|Rethinking Attention with Performers|[arxiv](https://arxiv.org/abs/2009.14794)|[code](https://github.com/google-research/google-research/tree/master/performer)|