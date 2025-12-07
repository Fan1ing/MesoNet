import torch
from torch_geometric.nn import  Set2Set,GeneralConv,GCNConv
import torch.nn as nn

def atoms_to_groups_local(x_atom, atom_idx, group_idx, G, reduce='mean'):

    if G == 0 or atom_idx.numel() == 0:
        return x_atom.new_zeros((G, x_atom.size(1)))
    Ha = x_atom.size(1)
    x_sel = x_atom.index_select(0, atom_idx)
    out = x_atom.new_zeros((G, Ha))
    out.scatter_add_(0, group_idx.view(-1,1).expand(-1,Ha), x_sel)
    if reduce == 'mean':
        cnt = x_atom.new_zeros((G,1))
        cnt.scatter_add_(0, group_idx.view(-1,1), x_atom.new_ones((atom_idx.numel(),1)))
        out = out / cnt.clamp_min_(1.0)
    return out

def groups_to_atoms_local(x_group, group_idx, atom_idx, N, reduce='mean'):

    if x_group.size(0) == 0 or atom_idx.numel() == 0:
        return x_group.new_zeros((N, x_group.size(1)))
    Hg = x_group.size(1)
    x_sel = x_group.index_select(0, group_idx)           # [N_inc, Hg]
    out = x_group.new_zeros((N, Hg))
    out.scatter_add_(0, atom_idx.view(-1,1).expand(-1,Hg), x_sel)
    if reduce == 'mean':
        cnt = x_group.new_zeros((N,1))
        cnt.scatter_add_(0, atom_idx.view(-1,1), x_group.new_ones((atom_idx.numel(),1)))
        out = out / cnt.clamp_min_(1.0)
    return out


def set2set_pool(features: torch.Tensor,
                 batch: torch.Tensor,
                 size: int,
                 s2s: Set2Set) -> torch.Tensor:

    if size == 0:
        D = features.size(1) if features.numel() > 0 else 0
        return features.new_zeros((0, 2 * D))

    if features.numel() == 0 or batch.numel() == 0:
        D = features.size(1) if features.numel() > 0 else 0
        return features.new_zeros((size, 2 * D))

    present = torch.unique(batch)                      # [P]
    P = int(present.numel())
    id_map = -torch.ones(size, dtype=torch.long, device=batch.device)
    id_map[present] = torch.arange(P, device=batch.device)
    compact_batch = id_map[batch]                      # [N_items] in [0..P-1]

    out_compact = s2s(features, compact_batch)         # [P, 2D]

    out = features.new_zeros((size, out_compact.size(1)))
    out[present] = out_compact
    return out



class AtomGroupBridgeFiLM(nn.Module):
    def __init__(self, atom_dim, group_dim, cond_dim, hidden=180,s2s_steps: int = 2):
        super().__init__()
        self.a2g_proj = nn.Linear(atom_dim, group_dim)
        self.g2a_proj = nn.Linear(group_dim, atom_dim)

        # FiLM 调制
        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, group_dim+42), nn.ReLU(),
            nn.Linear(group_dim+42, group_dim+39)
        )
        self.film_beta  = nn.Sequential(
            nn.Linear(cond_dim, group_dim+42), nn.ReLU(),
            nn.Linear(group_dim+42, group_dim+39)
        )
        self.a_proj_to_g = nn.Linear(atom_dim, group_dim-80)
        self.g_proj = nn.Linear(40, group_dim-80)
        self.s2s_a2g = Set2Set(80, processing_steps=s2s_steps)   # 输出 2*Dg
        self.merge_a2g = nn.Linear(group_dim+80, group_dim+39)            # 2*Dg -> Dg

        self.group_gcn1 =GeneralConv(group_dim, group_dim,attention=True)

        self.group_gcn2 =GCNConv(group_dim+42, group_dim+42)

        self.s2s_g2a = Set2Set(group_dim, processing_steps=s2s_steps)   # 输出 2*Dg
        self.g_proj_to_a = nn.Linear( group_dim, atom_dim)           # 2*Dg -> Ha

    def forward(self, x_atom, atom_idx, x_group, group_idx, edge_index_group, cond_atom,edge_attr_group=None):
        device = x_atom.device

        Na, Ha = x_atom.size(0), x_atom.size(1)

        Gm, Dg = x_group.size(0), x_group.size(1)
        x_group = x_group[:, 0:40]
        X_group = x_group
        if Gm == 0 or atom_idx.numel() == 0 or group_idx.numel() == 0:
            xg_empty = x_atom.new_zeros((0, Dg))
            return x_atom, xg_empty
        x_group = self.g_proj(x_group)
        xa_proj = self.a_proj_to_g(x_atom)  # [Na, Dg]
        xa_items = xa_proj.index_select(0, atom_idx)  # [N_inc, Dg]
        xg_a2g = set2set_pool(xa_items, group_idx, size=Gm, s2s=self.s2s_a2g)

        xg = torch.cat((x_group,xg_a2g),dim=1)

        xg = self.merge_a2g(xg)  # [Gm, Dg]# [Gm, 2*Dg]
        #xg_from_atom = self.merge_a2g(xg_a2g)  # [Gm, Dg]


        cond_g = atoms_to_groups_local(cond_atom, atom_idx, group_idx, Gm, reduce='mean')

        if Gm > 0:
            gamma = self.film_gamma(cond_g)                        # [Gm, Dg]
            beta  = self.film_beta(cond_g)                         # [Gm, Dg]
            xg    = gamma * xg  + beta   # [Gm, Dg]
        else:
            xg    = xg_from_atom  # [0, Dg] 安全路径
        '''if Gm > 0 and (edge_index_group is not None) and (edge_index_group.numel() > 0):
            xg = self.group_gcn2(xg, edge_index_group)'''
        xg = torch.cat((xg, cond_g), dim=1)
        if X_group.numel() > 0:
            type_ids_local = torch.argmax(X_group, dim=1).long()  # [Gm]
        else:
            type_ids_local = torch.empty(0, dtype=torch.long, device=X_group.device)

        #xg_items = xg.index_select(0, group_idx)  # [N_inc, Dg]
        '''xa_g2a = groups_to_atoms_local(xg, group_idx, atom_idx, Na, reduce='mean')  # [Na, 2*Dg]
        xa_from_group = self.g_proj_to_a(xa_g2a)'''
        xa_out = x_atom
        return xa_out, xg,type_ids_local