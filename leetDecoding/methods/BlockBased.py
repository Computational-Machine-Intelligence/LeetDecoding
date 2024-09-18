import torch
import math
from leetDecoding.methods.linear_attn import get_full_mask
from leetDecoding.methods.causal_dot_product import causal_dot_product


BLOCKM_BlockBased = 32 # Number of block rows
"""
python test/test_method.py --n 20 --type float32 --method blockbased --gpu 2  
"""

class BlockBased(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q,K,V,gamma=None):
        if gamma is not None:
            BLOCKM = BLOCKM_BlockBased
            b, h, seq, d = V.shape
            r = Q.shape[-1]
            num_block = math.floor(seq/BLOCKM)
            u = torch.zeros((b, h, r, d), device=V.device, dtype=V.dtype)
            ans = torch.zeros((b, h, seq, d), device=V.device, dtype=V.dtype)
            
            mask = get_full_mask(BLOCKM, gamma).to(V.device).to(V.dtype)
            cons = torch.exp(-gamma * BLOCKM)
            w1 = (torch.exp((-gamma).unsqueeze(1) * torch.arange(0, BLOCKM, device=V.device))).view(1,h,BLOCKM,1)
            w2 = (torch.exp((-gamma).unsqueeze(1) * torch.arange(BLOCKM, 0, step = -1, device=V.device))).view(1,h,BLOCKM,1)  

            for i in range(num_block):
                pbegin = i*BLOCKM
                pend = min(seq, (i+1)*BLOCKM)
                B_block = Q[:,:,pbegin:pend,:]
                C_block = K[:,:,pbegin:pend,:]
                V_block = V[:,:,pbegin:pend,:]
                
                tmp1 = torch.einsum('...mk,...dk->...md', B_block,  C_block) * mask
                l =  torch.einsum('...md,...dk->...mk', tmp1, V_block) 
                tmp2 = B_block * w1
                o = l + torch.einsum('...mk,...kd->...md',tmp2,u)
                
                tmp3 = C_block * w2
                # print('u.shape:',u.shape,'cons.shape:',cons.shape,'tmp3.shape:',tmp3.shape,'V_block.shape:',V_block.shape)
                u = torch.einsum('bhrd,h->bhrd', u, cons.squeeze(-1).squeeze(-1)) + torch.einsum('...mk,...md->...kd', tmp3, V_block)
                ans[:,:,pbegin:pend,:] = o
            if seq%BLOCKM:
                pbegin = num_block * BLOCKM
                B_block = Q[:,:,pbegin:seq,:]
                C_block = K[:,:,pbegin:seq,:]
                V_block = V[:,:,pbegin:seq,:]
                
                ble = seq-pbegin

                mask = get_full_mask(ble, gamma).to(V.device).to(V.dtype)
                w1 =  (torch.exp((-gamma).unsqueeze(1) * torch.arange(0, ble, device=V.device))).view(1,h,ble,1)
                w2 = (torch.exp((-gamma).unsqueeze(1) * torch.arange(ble, 0, step = -1, device=V.device))).view(1,h,ble,1)  

                tmp1 = torch.einsum('...mk,...dk->...md', B_block,  C_block) * mask
                l =  torch.einsum('...md,...dk->...mk', tmp1, V_block) 
                tmp2 = B_block * w1
                o = l + torch.einsum('...mk,...kd->...md',tmp2,u)
                
                tmp3 = C_block * w2
                u = torch.einsum('bhrd,h->bhrd', u, cons.squeeze(-1).squeeze(-1)) + torch.einsum('...mk,...md->...kd', tmp3, V_block)
            
                ans[:,:,pbegin:seq,:] = o
        else:
            BLOCKM = BLOCKM_BlockBased
            b, h, seq, d = V.shape
            r = Q.shape[-1]
            num_block = math.ceil(seq/BLOCKM)
            u = torch.zeros((b, h, r, d), device=V.device, dtype=V.dtype)
            ans = torch.zeros((b, h, seq, d), device=V.device, dtype=V.dtype)
            for i in range(num_block):
                pbegin = i*BLOCKM
                pend = min(seq, (i+1)*BLOCKM)
                B_block = Q[:,:,pbegin:pend,:]
                C_block = K[:,:,pbegin:pend,:]
                V_block = V[:,:,pbegin:pend,:]
                l = causal_dot_product(B_block,C_block,V_block)
                o = l + torch.einsum('...mk,...kd->...md',B_block,u)
                u += torch.einsum('...mk,...md->...kd',C_block,V_block)
                ans[:,:,pbegin:pend,:] = o
        return ans
    

blockBased = BlockBased.apply
