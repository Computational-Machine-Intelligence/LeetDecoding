import torch


class CausalDotProductTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, gamma=None):
        bsz, head, seq, dim = V.shape
        rank = Q.shape[-1]
        u = torch.zeros((bsz, head, rank, dim), device=V.device, dtype=V.dtype)
        ans = torch.zeros((bsz, head, seq, dim), device=V.device, dtype=V.dtype)
        if gamma is not None:
            for i in range(seq):
                B_block = Q[:,:,i:i+1,:]
                C_block = K[:,:,i:i+1,:]
                V_block = V[:,:,i:i+1,:]
                u += torch.einsum('...mk,...md->...kd',C_block,V_block)
                o = torch.einsum('...mk,...kd->...md',B_block,u)
                u *= gamma.view(1,gamma.shape[0],1,1)
                ans[:,:,i:i+1,:] = o
        else:
            for i in range(seq):
                B_block = Q[:,:,i:i+1,:]
                C_block = K[:,:,i:i+1,:]
                V_block = V[:,:,i:i+1,:]
                u += torch.einsum('...mk,...md->...kd',C_block,V_block)
                o = torch.einsum('...mk,...kd->...md',B_block,u)
                ans[:,:,i:i+1,:] = o
        return ans

causal_dot_product_torch = CausalDotProductTorch.apply

