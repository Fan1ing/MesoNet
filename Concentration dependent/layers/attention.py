import torch.nn as nn
import torch
from torch_geometric.nn import  Set2Set
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class FeatureCrossAttention(nn.Module):
    def __init__(self, dim_in_q, dim_in_kv, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = 32

        self.q_map = nn.Linear(dim_in_q, 128)
        self.k_map = nn.Linear(dim_in_kv, 128)
        self.v_map = nn.Linear(dim_in_kv, 128)

        self.out_map = nn.Linear(128, model_dim)
        self.Qout = nn.Linear(128, model_dim)
        self.norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale   = 1 / math.sqrt(self.d_k)

    def forward(self, Q_in, KV_in, mask=None):
        """
        Q_in:  (B, L_q, dim_in_q)
        KV_in: (B, L_kv, dim_in_kv)
        """


        # ========== 1) 映射 ==========
        Qm = self.q_map(Q_in)  # (B, L_q, model_dim)
        Km = self.k_map(KV_in) # (B, L_kv, model_dim)
        Vm = self.v_map(KV_in) # (B, L_kv, model_dim)
        B, L_q, Dq = Qm.shape
        _, L_kv, Dk = Km.shape
        # ========== 2) 先拆成多头 ==========
        # 每个头负责一部分特征
        Qh = Qm.view(B, L_q, self.num_heads, self.d_k).permute(0, 2, 3, 1)  # (B, H, d_k, L_q)
        Kh = Km.view(B, L_kv, self.num_heads, self.d_k).permute(0, 2, 3, 1) # (B, H, d_k, L_kv)
        Vh = Vm.view(B, L_kv, self.num_heads, self.d_k).permute(0, 2, 3, 1) # (B, H, d_k, L_kv)

        # ========== 3) 现在交换特征维 & token维 ==========
        # 现在注意力是在“特征之间”计算
        # 这里 d_k 视为 sequence-like 维度，而 L_q/L_kv 是特征通道的上下文
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale  # (B, H, d_k, d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out_h = torch.matmul(attn, Vh)  # (B, H, d_k, L_q)

        out_h = out_h.permute(0, 3, 1, 2).contiguous().view(B, L_q, self.num_heads * self.d_k)  # (B, L_q, model_dim)

        out = self.out_map(out_h)
        Qm_ = self.Qout(Qm)
        out = self.norm(Qm_ + out)

        return out, attn


class CrossMolGroupInter(nn.Module):

    def __init__(self, group_dim: int, K: int, mol_emb_dim: int = 18,
                 num_heads: int = 4, use_set2set: bool = True, s2s_steps: int = 2):
        super().__init__()
        self.K = K
        self.group_dim = group_dim
        self.in_dim = group_dim + mol_emb_dim
        self.mol_emb = nn.Linear(K, mol_emb_dim, bias=False)

        self.mha = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)


        # === 2. 前馈网络 (FFN) ===
        self.ffn = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim),
            nn.ReLU(),
            nn.Linear(2 * self.in_dim, self.in_dim),
        )
        self.norm2 = nn.LayerNorm(self.in_dim)
        self.norm3 = nn.LayerNorm(self.in_dim)

        # === 3. 读出层 ===
        self.readout = nn.Sequential(
            nn.Linear(self.in_dim * 2, group_dim),
            nn.ReLU(),
            nn.Linear(group_dim, group_dim),
        )



        self.use_set2set = use_set2set
        if use_set2set:
            # 分子和混合物分别使用不同的 Set2Set 聚合
            self.mol_s2s = Set2Set(self.in_dim, processing_steps=s2s_steps)  # 分子层级聚合
            self.mix_s2s = Set2Set(self.in_dim, processing_steps=s2s_steps)  # 混合物层级聚合

    def forward(self, xg_list, gb_list, return_attn=False):
        """
        xg_list: [xg1, xg2, xg3], xg_i: [Gi, group_dim]
        gb_list: [gb1, gb2, gb3], gb_i: [Gi] in [0..B_sub-1]
        """
        device = xg_list[0].device
        K = self.K

        # 计算 B_sub（同一mini-batch内混合物个数）
        if any(gb.numel() > 0 for gb in gb_list):
            B_sub = int(max((int(gb.max()) if gb.numel() > 0 else -1) for gb in gb_list) + 1)
        else:
            B_sub = 1

        # ==== 1) 拼接所有 token（带分子ID嵌入） ====
        tokens_all, token_b, token_bi, token_mol = [], [], [], []
        for i in range(K):
            xg_i, gb_i = xg_list[i], gb_list[i]
            if xg_i.numel() == 0:
                continue

            # 🧩 构造 one-hot 表示分子ID
            one_hot = F.one_hot(torch.tensor(i, device=device), num_classes=K).float()  # [K]
            one_hot = one_hot.unsqueeze(0)  # [1, K]
            me = self.mol_emb(one_hot)      # [1, mol_emb_dim]
            me = me.expand(xg_i.size(0), -1)  # [Gi, mol_emb_dim]

            t = torch.cat([xg_i, me], dim=1)                                    # [Gi, H+Em]
            tokens_all.append(t)
            token_b.append(gb_i)
            token_bi.append(gb_i * K + i)                                       # [Gi]
            token_mol.append(torch.full((xg_i.size(0),), i, device=device, dtype=torch.long))

        if len(tokens_all) == 0:
            # 没有任何基团
            per_mol_out = [torch.zeros(B_sub, self.group_dim, device=device) for _ in range(K)]
            mix_feat = torch.zeros(B_sub, 2 * self.in_dim, device=device) if self.use_set2set else None
            return per_mol_out, mix_feat

        feats   = torch.cat(tokens_all, dim=0)         # [N_tok, H_in]
        b_idx   = torch.cat(token_b,   dim=0).long()   # [N_tok]  mixture id
        bi_idx  = torch.cat(token_bi,  dim=0).long()   # [N_tok]  global (b,i) id
        mol_id = torch.cat(token_mol, dim=0).long()  # [N_tok]

        # ==== 2) 构造按 mixture 分组的“批内序列” ====
        # 把 token 按 b 排序 -> 能按 b 一刀切地切分
        sort_order = torch.argsort(b_idx)              # [N_tok]
        feats_sorted  = feats.index_select(0, sort_order)
        b_sorted      = b_idx.index_select(0, sort_order)
        bi_sorted     = bi_idx.index_select(0, sort_order)
        mol_sorted = mol_id.index_select(0, sort_order)  # [N_tok]

        # 每个 b 有多少 token：
        counts = torch.bincount(b_sorted, minlength=B_sub)  # [B_sub]
        # 按 b 切成列表（Python层切一次，MHA 只调 1 次）
        chunks = torch.split(feats_sorted, counts.tolist())
        # pad 成同长度
        padded = pad_sequence(chunks, batch_first=True, padding_value=0.0)      # [B_sub, L_max, H_in]

        # key_padding_mask: True=要mask（pad位置）——每行后面的 pad 全是 0
        L_max = padded.size(1)
        # 有效长度 lens: [B_sub]
        lens = counts
        arange_L = torch.arange(L_max, device=device).unsqueeze(0)              # [1, L_max]
        key_pad_mask = arange_L >= lens.unsqueeze(1)                            # [B_sub, L_max], bool

        # ==== 3) 一次 MHA ====
        attn_out, attn_mat1 = self.mha(padded, padded, padded, key_padding_mask=key_pad_mask)  # [B_sub, L_max, H_in]

        # (b) FFN + 残差 + LayerNorm
        padded = self.norm2(padded + attn_out)
        attn_out, attn_mat2 = self.mha2(padded, padded, padded, key_padding_mask=key_pad_mask)
        x = self.norm3(padded + attn_out)

        # === 4) 去 pad ===
        valid_mask = (torch.arange(L_max, device=device)[None, :] < lens[:, None])
        x_flat = x.reshape(-1, x.size(-1))[valid_mask.view(-1)]

        N = feats.size(0)
        inv = torch.empty_like(sort_order)
        inv[sort_order] = torch.arange(N, device=device)
        attn_unsorted = x_flat[inv]  # [N_tok, H_in]                # [N_tok, H_in]             # [N_tok, H_in]

        # ==== 4) 读出（使用 Set2Set 聚合） ====
        # 4.1 per-molecule：通过 Set2Set 聚合每个分子的基团信息
        mol_id_per_token = (bi_idx % self.K)  # [N_tok]



        per_mol_out = []
        for i in range(self.K):
            mask_i = (mol_id_per_token == i)
            if mask_i.any():
                part_i = attn_unsorted[mask_i]  # [N_i, H_in]
                b_idx_i = b_idx[mask_i]  # [N_i]
                # 每个“混合物 b”在“第 i 个分子”上的 Set2Set 聚合
                s2s_i = self.mol_s2s(part_i, b_idx_i)  # [B_sub, 2*H_in]
            else:
                s2s_i = attn_unsorted.new_zeros(B_sub, 2 * self.in_dim)
            per_mol_out.append(self.readout(s2s_i))  # [B_sub, group_dim]

        if self.use_set2set:
            mix_feat = self.mix_s2s(attn_unsorted, b_idx)          # [B_sub, 2*H_in]
        else:
            mix_feat = None

        if return_attn:
            # 计算每个 mixture 的起始下标，方便画图分割
            b_offsets = torch.zeros(B_sub + 1, dtype=torch.long, device=device)
            b_offsets[1:] = torch.cumsum(counts, dim=0)

            return per_mol_out, mix_feat, {
                "attn1": attn_mat1,               # [B_sub, h, L_max, L_max]
                "attn2": attn_mat2,               # [B_sub, h, L_max, L_max]
                "lengths": lens,                  # [B_sub]
                "counts": counts,                 # [B_sub]
                "b_offsets": b_offsets,           # [B_sub+1]
                "mol_sorted": mol_sorted,         # [N_tok_sorted]
            }

        return per_mol_out, mix_feat

