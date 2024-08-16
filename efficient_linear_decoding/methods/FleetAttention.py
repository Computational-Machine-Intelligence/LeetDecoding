import torch


def ewma_cumsum(x, weight_decay):
    h = x.shape[1]
    result = torch.zeros_like(x)
    result[:,:,0,:] = x[:,:,0,:]
    for i in range(1, x.shape[-2]):
        result[:,:,i,:] = x[:,:,i,:] + result[:,:,i-1,:] * weight_decay.view(1,h,1)
    return result


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
            ans += Q[:,:,:,i].unsqueeze(-1) * ewma_cumsum(K[:,:,:,i].unsqueeze(-1)* V,gamma)
    return ans


    
   