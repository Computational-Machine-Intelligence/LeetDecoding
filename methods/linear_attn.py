import torch
import math

# Construct weight decay vector
def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )
            
    slopes = torch.FloatTensor(get_slopes(n_attention_heads)).reshape(n_attention_heads, 1, 1)
    return slopes


# Construct a decay mask for a single head
def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    # -n, ..., -2, -1, 0
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, : i + 1] = -torch.flip(y, [0])

    return torch.exp(mask)


# Build decay mask for all heads
def get_full_mask(n, slopes):
    if slopes == None:
        mask = torch.tril(torch.ones((n, n)))
    else:
        arr = []
        for slope in slopes:
            arr.append(get_mask(n, slope.item()))
        mask = torch.stack(arr, dim=0)
    return mask


# The original algorithm is vanilla
def linear_attn(q, k, v, s=None):
    b, h, n, r = q.shape
    d = v.shape[-1]
    if s is not None:
        mask = get_full_mask(n, s).to(q.device).to(torch.float32)
        qk = torch.matmul(q, k.transpose(2, 3))
        qk = (qk.to(torch.float32) * mask).to(q.dtype)
        o = torch.matmul(qk, v)
    else:
        M = torch.tril(torch.ones(n,n,device=q.device,dtype=q.dtype))
        o = torch.matmul(torch.matmul(q, k.transpose(2, 3)) * M, v)
    return o


if __name__ == "__main__":
    tmp = _build_slope_tensor(8)
    tmp2 = get_full_mask(4, tmp)
    print(tmp)
    print(tmp2)