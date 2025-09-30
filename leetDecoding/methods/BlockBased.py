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
    def forward(ctx, Q, K, V, gamma=None):
        # 保存输入设备和类型
        device = Q.device
        dtype = Q.dtype  # 关键：以输入 dtype 为准

        if gamma is not None:
            BLOCKM = BLOCKM_BlockBased
            b, h, seq, d = V.shape
            r = Q.shape[-1]
            num_block = math.floor(seq / BLOCKM)
            
            # 初始化时指定 dtype 和 device
            u = torch.zeros((b, h, r, d), device=device, dtype=dtype)
            ans = torch.zeros((b, h, seq, d), device=device, dtype=dtype)
            
            # gamma 相关计算：确保使用正确的 dtype
            with torch.no_grad():
                # 如果 gamma 是张量，确保其 dtype 正确
                if isinstance(gamma, torch.Tensor):
                    gamma = gamma.to(dtype=dtype, device=device)
                else:
                    gamma = torch.tensor(gamma, dtype=dtype, device=device)
                
                # mask
                mask = get_full_mask(BLOCKM, gamma).to(device).to(dtype)
                
                # cons: exp(-gamma * BLOCKM)
                cons = torch.exp(-gamma * BLOCKM)  # shape: [h] or scalar
                
                # w1: exp(-gamma * [0, 1, ..., BLOCKM-1])
                w1 = torch.exp(-gamma.unsqueeze(1) * torch.arange(0, BLOCKM, device=device, dtype=dtype))
                w1 = w1.view(1, h, BLOCKM, 1)
                
                # w2: exp(-gamma * [BLOCKM, ..., 1])
                w2 = torch.exp(-gamma.unsqueeze(1) * torch.arange(BLOCKM, 0, step=-1, device=device, dtype=dtype))
                w2 = w2.view(1, h, BLOCKM, 1)
            for i in range(num_block):
                pbegin = i * BLOCKM
                pend = min(seq, (i + 1) * BLOCKM)
                B_block = Q[:, :, pbegin:pend, :]
                C_block = K[:, :, pbegin:pend, :]
                V_block = V[:, :, pbegin:pend, :]

                # 计算当前 block 的长度
                block_len = B_block.shape[2]

                tmp1 = torch.einsum('...mk,...dk->...md', B_block, C_block) * mask
                l = torch.einsum('...md,...dk->...mk', tmp1, V_block)

                # 修复：使用 block_len 切片 w1 和 w2，而不是引用未定义的 tmp2
                tmp2 = B_block * w1[:, :, :block_len, :]
                tmp2 = tmp2.to(dtype=l.dtype, device=l.device)
                u = u.to(dtype=l.dtype, device=l.device)
                o = l + torch.einsum('...mk,...kd->...md', tmp2, u)

                tmp3 = C_block * w2[:, :, :block_len, :]  # 同样修复 w2 切片
                term1 = torch.einsum('bhrd,h->bhrd', u, cons.squeeze(-1).squeeze(-1).to(dtype=u.dtype))
                term2 = torch.einsum('...mk,...md->...kd', tmp3, V_block.to(dtype=tmp3.dtype))
                u = term1 + term2

                ans[:, :, pbegin:pend, :] = o

            # 处理剩余部分（last block）
            if seq % BLOCKM != 0:
                pbegin = num_block * BLOCKM
                B_block = Q[:, :, pbegin:seq, :]
                C_block = K[:, :, pbegin:seq, :]
                V_block = V[:, :, pbegin:seq, :]
                ble = seq - pbegin

                with torch.no_grad():
                    mask = get_full_mask(ble, gamma).to(device).to(dtype)
                    w1_tail = torch.exp(-gamma.unsqueeze(1) * torch.arange(0, ble, device=device, dtype=dtype)).view(1, h, ble, 1)
                    w2_tail = torch.exp(-gamma.unsqueeze(1) * torch.arange(ble, 0, step=-1, device=device, dtype=dtype)).view(1, h, ble, 1)

                tmp1 = torch.einsum('...mk,...dk->...md', B_block, C_block) * mask
                l = torch.einsum('...md,...dk->...mk', tmp1, V_block)
                tmp2 = B_block * w1_tail
                tmp2 = tmp2.to(dtype=l.dtype, device=l.device)
                u = u.to(dtype=l.dtype, device=l.device)
                o = l + torch.einsum('...mk,...kd->...md', tmp2, u)

                tmp3 = C_block * w2_tail
                term1 = torch.einsum('bhrd,h->bhrd', u, cons.squeeze(-1).squeeze(-1).to(dtype=u.dtype))
                term2 = torch.einsum('...mk,...md->...kd', tmp3, V_block.to(dtype=tmp3.dtype))
                u = term1 + term2

                ans[:, :, pbegin:seq, :] = o

        else:
            # No gamma case
            BLOCKM = BLOCKM_BlockBased
            b, h, seq, d = V.shape
            r = Q.shape[-1]
            num_block = math.ceil(seq / BLOCKM)
            u = torch.zeros((b, h, r, d), device=device, dtype=dtype)
            ans = torch.zeros((b, h, seq, d), device=device, dtype=dtype)

            for i in range(num_block):
                pbegin = i * BLOCKM
                pend = min(seq, (i + 1) * BLOCKM)
                B_block = Q[:, :, pbegin:pend, :]
                C_block = K[:, :, pbegin:pend, :]
                V_block = V[:, :, pbegin:pend, :]

                l = causal_dot_product(B_block, C_block, V_block)
                o = l + torch.einsum('...mk,...kd->...md', B_block, u.to(dtype=B_block.dtype))
                u = u + torch.einsum('...mk,...md->...kd', C_block, V_block).to(dtype=u.dtype)
                ans[:, :, pbegin:pend, :] = o

        ctx.save_for_backward(Q, K, V, gamma)  # 如果需要反向传播
        return ans


# 使用
blockBased = BlockBased.apply