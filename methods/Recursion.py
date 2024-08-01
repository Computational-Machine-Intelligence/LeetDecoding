import torch
from methods.linear_attn import get_full_mask,linear_attn
from methods.FleetAttention import FleetAttention


"""
python test/test_method.py --n 20 --type float32 --method recursion --gpu 2  
"""

BLOCKM = 4 # Number of block rows


def recursive_infer_with_decay(B,C,V,left,right,gamma):
    if (right-left)<=BLOCKM:
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
        out = pytorch_FleetAttention(B[:,:,left:right,:], C[:,:,left:right,:], V[:,:,left:right,:])
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
            ans = recursive_infer_with_decay(Q,K,V,0,n,gamma)
        else:
            ans = recursive_infer(Q,K,V,0,n,BLOCKM)
        return ans


recursion = Recursion.apply
