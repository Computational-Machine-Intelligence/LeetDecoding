## An efficient Linear Attention Decoding package

### 1. installation

#### 1.1 Source Code Install
```bash
conda create -n efficient_linear_decoding python=3.9
conda activate efficient_linear_decoding
pip install requirements.txt
```

#### 1.2 pip Install
```bash
pip install efficient_linear_decoding
```

### 2. usage

```python
import torch
from efficient_linear_decoding import causal_linear_decoder

# Create input tensor
Q = torch.randn(2,32,1024,128,device='cuda:0')
K = torch.randn(2,32,1024,128,device='cuda:0')
V = torch.randn(2,32,1024,128,device='cuda:0')

# Inference using causal_linear_decoder
output = causal_linear_decoder(Q,K,V)

# If you want to input a mask with weight, set the is_mask_weight: True
gamma = torch.full((32,),0.5,device='cuda:0')
output = causal_linear_decoder(Q,K,V,is_mask_weight=True,gamma=gamma)

```


### 3. acknowledgement
|method|Title|Paper|Code|
|---|---|---|---|
|causal_dot_product|Fast Transformers with Clustered Attention|[arxiv](https://arxiv.org/abs/2007.04825) |[code](https://github.com/idiap/fast-transformers/tree/master/fast_transformers/causal_product)|
|Lighting Attention-2|Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models|[arxiv](https://arxiv.org/abs/2401.04658)|[code](https://github.com/OpenNLPLab/lightning-attention)