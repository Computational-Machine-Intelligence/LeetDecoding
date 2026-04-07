import torch


class CausalDotProductTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, gamma=None):
        bsz, head, seq, dim = V.shape
        rank = Q.shape[-1]
        orig_dtype = V.dtype
        # Keep u in float32 to avoid bf16 precision loss during long-range accumulation
        u = torch.zeros((bsz, head, rank, dim), device=V.device, dtype=torch.float32)
        ans = torch.zeros((bsz, head, seq, dim), device=V.device, dtype=torch.float32)
        if gamma is not None:
            gamma_f = gamma.float()
            for i in range(seq):
                B_block = Q[:,:,i:i+1,:].float()
                C_block = K[:,:,i:i+1,:].float()
                V_block = V[:,:,i:i+1,:].float()
                u *= gamma_f.view(1,gamma_f.shape[0],1,1)
                u += torch.einsum('...mk,...md->...kd',C_block,V_block)
                o = torch.einsum('...mk,...kd->...md',B_block,u)
                ans[:,:,i:i+1,:] = o
        else:
            for i in range(seq):
                B_block = Q[:,:,i:i+1,:].float()
                C_block = K[:,:,i:i+1,:].float()
                V_block = V[:,:,i:i+1,:].float()
                u += torch.einsum('...mk,...md->...kd',C_block,V_block)
                o = torch.einsum('...mk,...kd->...md',B_block,u)
                ans[:,:,i:i+1,:] = o
        return ans.to(orig_dtype)

causal_dot_product_torch = CausalDotProductTorch.apply

