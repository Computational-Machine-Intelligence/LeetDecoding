import torch


BLOCK = 256


def lightningAttention2_torch(q,k,v,gamma=None):
    if gamma is not None: 
        gamma = gamma.view(-1,1,1).to(torch.float32)
        b, h, n, d = q.shape
        e = v.shape[-1]
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK
        # other
        array = torch.arange(BLOCK).to(q) + 1  ## !!!! important
        q_decay = torch.exp(-gamma * array.reshape(-1, 1))
        k_decay = torch.exp(-gamma * (BLOCK - array.reshape(-1, 1)))
        index = array[:, None] - array[None, :]
        s_index = gamma * index[
            None,
            None,
        ]
        s_index = torch.where(index >= 0, -s_index, float("-inf"))
        diag_decay = torch.exp(s_index)

        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        for i in range(NUM_BLOCK):
            si = i * BLOCK
            ei = min(si + BLOCK, n)
            m = ei - si

            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()
            qkv_none_diag = torch.matmul(qi * q_decay[:, :m],
                                            kv).to(torch.float32)

            # diag
            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(
                torch.float32) * diag_decay[:, :, :m, :m]
            qkv_diag = torch.matmul(qk, vi.to(torch.float32))
            block_decay = torch.exp(-gamma * m)
            output[:, :, si:ei] = qkv_none_diag + qkv_diag
            kv = block_decay * kv + torch.matmul(
                (ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)
    else:
        b, h, n, d = q.shape
        e = v.shape[-1]
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK

        array = torch.arange(BLOCK).to(q) + 1  ## !!!! important
        index = array[:, None] - array[None, :]
        diag_decay = torch.where(index >= 0, 1, 0).to(torch.float32)
        kv = torch.zeros(b, h, d, e).to(q.device)
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        for i in range(NUM_BLOCK):
            si = i * BLOCK
            ei = min(si + BLOCK, n)
            m = ei - si

            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()
            o_inter = torch.matmul(qi[:, :m].to(torch.float32), kv)

            # diag
            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(
                torch.float32) * diag_decay[:m,:m]
            o_intra = torch.matmul(qk, vi.to(torch.float32))
            # block_decay = torch.exp(-slope_rate * m)
            output[:, :, si:ei] = o_inter + o_intra
            kv = kv + torch.matmul(ki[:, -m:].transpose(-1, -2).to(vi.dtype), vi)
    return output