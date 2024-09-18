import torch


def discounted_cumsum(x, s):
    b,h,n,d = x.shape
    s = s.view(h,1,1)
    p = torch.cumsum(torch.log(s).expand(-1,-1,n),dim=-1)
    return torch.exp(p.view(1,h,n,1) + torch.logcumsumexp(torch.log(x)-p.view(1,h,n,1),dim=-2))

def FleetAttention(Q,K,V,gamma=None):
    b,h,n,r = Q.shape
    d = V.shape[-1]
    type = Q.dtype
    device = Q.device
    ans = torch.zeros_like(V,device=device,dtype=type)
    if gamma is None:
        for i in range(r):
            ans += Q[:,:,:,i].unsqueeze(-1) * torch.cumsum(K[:,:,:,i].unsqueeze(-1)* V,dim=-2)
    else:
        for i in range(r):
            ans += Q[:,:,:,i].unsqueeze(-1) * discounted_cumsum(K[:,:,:,i].unsqueeze(-1)* V,gamma)
    return ans

    
   