import torch


def discounted_cumsum(x, s):
    """
    Compute discounted cumulative sum: result[t] = sum_{j=0}^{t} s^(t-j) * x[j]
    
    Uses iterative recursion instead of log/exp tricks to avoid
    numerical issues when x contains zeros or negative values.
    """
    b, h, n, d = x.shape
    s_f = s.view(1, h, 1).float()
    
    result = torch.empty_like(x, dtype=torch.float32)
    cv = torch.zeros(b, h, d, device=x.device, dtype=torch.float32)
    
    x_f = x.float()
    for t in range(n):
        cv = cv * s_f + x_f[:, :, t]
        result[:, :, t] = cv
    
    return result


def FleetAttention(Q, K, V, gamma=None):
    b, h, n, r = Q.shape
    d = V.shape[-1]
    orig_dtype = Q.dtype
    device = Q.device
    
    ans = torch.zeros(b, h, n, d, device=device, dtype=torch.float32)
    
    Q_f = Q.float()   # [b,h,n,r]
    K_f = K.float()   # [b,h,n,r]
    V_f = V.float()   # [b,h,n,d]
    
    if gamma is None:
        for i in range(r):
            ki = K_f[:, :, :, i]     # [b,h,n]
            qi = Q_f[:, :, :, i]     # [b,h,n]
            cv = torch.cumsum(ki.unsqueeze(-1) * V_f, dim=-2)  # [b,h,n,d]
            ans += qi.unsqueeze(-1) * cv
    else:
        gamma_f = gamma.float()
        for i in range(r):
            qi = Q_f[:, :, :, i]     # [b,h,n]
            ki = K_f[:, :, :, i]     # [b,h,n]
            kv = ki.unsqueeze(-1) * V_f  # [b,h,n,d]
            cv = discounted_cumsum(kv, gamma_f)  # [b,h,n,d]
            ans += qi.unsqueeze(-1) * cv
    
    return ans.to(orig_dtype)
