import torch
from leetDecoding.methods.linear_attn import get_full_mask,linear_attn
from leetDecoding.methods.causal_dot_product import causal_dot_product
import pdb

"""
python test/test_method.py --n 20 --type float32 --method recursion --gpu 2  
"""

# BLOCKM = 32 # Number of block rows


def recursive_infer_with_decay(B,C,V,left,right,gamma, min_seq_len=4):
    if (right-left)<=min_seq_len:
        mask = get_full_mask(right-left, gamma).to(V.device).to(V.dtype)
        out = torch.matmul(torch.einsum("...am,...bm->...ab",B[:,:,left:right,:],C[:,:,left:right,:])*mask, V[:,:,left:right,:])
        return out
    else:
        mid = (left+right)//2
        attn_left_up = recursive_infer_with_decay(B,C,V,left,mid,gamma) 
        attn_right_down = recursive_infer_with_decay(B,C,V,mid,right,gamma) 
        w1 = (torch.exp((-gamma).unsqueeze(1) * torch.arange(0, right-mid, device=V.device))).view(1,B.shape[1],right-mid,1)
        w2 = (torch.exp((-gamma).unsqueeze(1) * torch.arange(mid-left, 0, step = -1, device=V.device))).view(1,B.shape[1],mid-left,1)  
        attn_left_down = torch.matmul(B[:,:,mid:right,:] * w1, torch.einsum('...ma,...mb->...ab',C[:,:,left:mid,:]*w2, V[:,:,left:mid,:]))
        attn_left_sum = attn_right_down + attn_left_down
        result_matrix = torch.cat((attn_left_up, attn_left_sum), dim=2)
        return result_matrix
        

def recursive_infer(B,C,V,left,right,min_seq_len:32):
    if (right-left)<=min_seq_len:
        out = causal_dot_product(B[:,:,left:right,:], C[:,:,left:right,:], V[:,:,left:right,:])
        return out
    else:
        mid = (left+right)//2
        attn_left_up = recursive_infer(B,C,V,left,mid,min_seq_len) 
        attn_right_down = recursive_infer(B,C,V,mid,right,min_seq_len) 
        attn_left_down = torch.matmul(B[:,:,mid:right,:], torch.matmul(C[:,:,left:mid,:].transpose(2,3),V[:,:,left:mid,:]))
        attn_left_sum = attn_right_down + attn_left_down
        result_matrix = torch.cat((attn_left_up, attn_left_sum), dim=2)
        return result_matrix


class Recursion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q,K,V,gamma=None):
        b,h,n,r = Q.shape
        d = V.shape[-1]
        if gamma is not None:
            ans = recursive_infer_with_decay(Q,K,V,0,n,gamma,32)
        else:
            ans = recursive_infer(Q,K,V,0,n,32)
        return ans


recursion = Recursion.apply
