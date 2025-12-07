import torch.nn as nn
from ncps.torch import CfC
from torch_geometric.nn import NNConv, Set2Set,AttentiveFP,global_mean_pool,GeneralConv
from ncps.wirings import AutoNCP
from torch_geometric.utils import subgraph as pyg_subgraph

from layers.attention import *
from layers.atom_group_bridge import *
from model.utils import _groups_batch_from_a2g_local
hi_a_dim =6
hi_b_dim =41
G_dim = hi_g_dim = 40
C_dim= 1

feature_dim = 32



class MesoNet(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim,
                 d_group_in, d_group_hidden=128):

        super(MesoNet, self).__init__()

        self.K = 3
        self.mol_emb_dim = 18

        self.cross_group_attn = CrossMolGroupInter(
            group_dim=hidden_dim+42,
            K=3,
            mol_emb_dim=18,
            num_heads=4,
            use_set2set=True,
            s2s_steps=2
        )
        self.attn_atom_elem   = FeatureCrossAttention(dim_in_q=feature_dim, dim_in_kv=feature_dim, model_dim=feature_dim, num_heads=4)
        self.attn_group_atom  = FeatureCrossAttention(dim_in_q=feature_dim, dim_in_kv=feature_dim, model_dim=feature_dim, num_heads=4)
        self.attn_global_group= FeatureCrossAttention(dim_in_q=G_dim, dim_in_kv=feature_dim, model_dim=feature_dim, num_heads=4)
        self.inter            = FeatureCrossAttention(dim_in_q=G_dim+C_dim, dim_in_kv=G_dim*2+C_dim*2, model_dim=feature_dim, num_heads=4)

        edge_hidden_dim = feature_dim
        self.a11 = NNConv(hi_b_dim, feature_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, hi_b_dim * feature_dim)
        ), aggr="mean")

        self.NCP1 = CfC(feature_dim, AutoNCP(feature_dim*2+C_dim*3,feature_dim), batch_first=True)

        self.NCP2= CfC(hidden_dim+C_dim*3, AutoNCP(hidden_dim*2,hidden_dim), batch_first=True)
        self.x2 = nn.Linear(hi_a_dim,feature_dim)
        self.x22 = nn.Linear(feature_dim*3,feature_dim*3)
        self.relu = nn.ReLU()
        self.xm3 = nn.Linear(hidden_dim, hidden_dim)
        self.subgraph_conv1 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')
        self.subgraph_conv2 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')

        self.global_conv = NNConv(hidden_dim*2+3, hidden_dim, nn.Sequential(
            nn.Linear(4, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim , (hidden_dim*2+3*C_dim)*hidden_dim)
        ), aggr='mean')


        self.set2set  = Set2Set(hidden_dim, processing_steps=2)
        self.set2set2 = Set2Set(3*hidden_dim+3*C_dim , processing_steps=2)
        self.setgroup = Set2Set(237, processing_steps=2)

        self.fc = nn.Sequential(
            nn.Linear(2203+237,1024),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


        # FiLM
        self.c1_gamma = nn.Sequential(nn.Linear(3, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c1_beta  = nn.Sequential(nn.Linear(3, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c2_gamma = nn.Sequential(nn.Linear(3, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c2_beta  = nn.Sequential(nn.Linear(3, 160), nn.ReLU(), nn.Linear(160, 160))

        self.c3_gamma = nn.Sequential(nn.Linear(3, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c3_beta  = nn.Sequential(nn.Linear(3, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c4_gamma = nn.Sequential(nn.Linear(3, 234), nn.ReLU(), nn.Linear(234, 234))
        self.c4_beta  = nn.Sequential(nn.Linear(3, 234), nn.ReLU(), nn.Linear(234, 234))
        self.atom_group_bridge = AtomGroupBridgeFiLM(
            atom_dim=hidden_dim, group_dim=hidden_dim,cond_dim = 3, s2s_steps=2
        )

    @staticmethod
    def _slice_group_view(data, mol_id, atom_mask):
        """

        """
        device = data.x.device
        if data.group_mol_id.numel() == 0:
            return None

        gid_global = torch.nonzero(data.group_mol_id == mol_id, as_tuple=False).view(-1)
        if gid_global.numel() == 0:
            return None

        # group
        gid_map = torch.full((int(data.group_mol_id.numel()),), -1, device=device, dtype=torch.long)
        gid_map[gid_global] = torch.arange(gid_global.numel(), device=device)

        # atom
        aid_global = torch.nonzero(atom_mask, as_tuple=False).view(-1)
        aid_map = torch.full((data.x.size(0),), -1, device=device, dtype=torch.long)
        aid_map[aid_global] = torch.arange(aid_global.numel(), device=device)

        # a2g
        if data.atom2group_index.numel() > 0:
            g_idx_global = data.atom2group_index[0]
            a_idx_global = data.atom2group_index[1]
            keep = (gid_map[g_idx_global] >= 0) & (aid_map[a_idx_global] >= 0)
            g_idx_local = gid_map[g_idx_global[keep]]
            a_idx_local = aid_map[a_idx_global[keep]]
            a2g_local = torch.stack([g_idx_local, a_idx_local], dim=0)
        else:
            a2g_local = torch.empty(2, 0, dtype=torch.long, device=device)

        if data.edge_index_group.numel() > 0:
            u, v = data.edge_index_group
            keep_e = (gid_map[u] >= 0) & (gid_map[v] >= 0)
            eig_local = torch.stack([gid_map[u[keep_e]], gid_map[v[keep_e]]], dim=0)
        else:
            eig_local = torch.empty(2, 0, dtype=torch.long, device=device)

        xg_local = data.x_group.index_select(0, gid_global)
        return {"xg_local": xg_local, "a2g_local": a2g_local, "eig_local": eig_local}

    def process_subgraph(self, data, mask, mol_id):

        x = data.x
        #Feature reading
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        subgraph_x = x[mask]
        subgraph_edge_index, subgraph_edge_attr = pyg_subgraph(mask, edge_index, edge_attr, relabel_nodes=True)
        group_view = self._slice_group_view(data, mol_id, mask)
        xg_local  = group_view["xg_local"]
        a2g_local = group_view["a2g_local"]
        eig_local = group_view["eig_local"]
        x1 = subgraph_x[:, 0:hi_b_dim]
        x1 = self.a11(x1, subgraph_edge_index, subgraph_edge_attr)
        x1 = self.relu(x1)
        x2 = subgraph_x[:, hi_b_dim:hi_b_dim + hi_a_dim]
        x3 = subgraph_x[:, hi_b_dim + hi_a_dim:hi_b_dim + hi_a_dim + hi_g_dim]

        g = subgraph_x[:, hi_b_dim + hi_a_dim + hi_g_dim + 4:hi_b_dim + hi_a_dim + hi_g_dim + 4 + G_dim + C_dim]

        G_ = subgraph_x[:, hi_b_dim + hi_a_dim + hi_g_dim + 4 + G_dim + C_dim:]
        C = torch.cat((g[:, [hi_g_dim]], G_[:, [hi_g_dim]], G_[:, [hi_g_dim + C_dim]]), dim=1)
        global_G =C
        x2_output = self.x2(x2)
        x2_output = self.relu(x2_output)


        #cross-attention
        #Mixture - Molecular
        inter, _ = self.inter(g.unsqueeze(1), G_.unsqueeze(1))
        inter = inter.squeeze(1)

        #Molecular -  functional group attribution
        global_updated, _ = self.attn_global_group(x3.unsqueeze(1), inter.unsqueeze(1))
        global_updated = global_updated.squeeze(1)

        #Group Attribution - Atomic feature
        group_updated, _ = self.attn_group_atom(x1.unsqueeze(1), global_updated.unsqueeze(1))
        group_updated = group_updated.squeeze(1)

        #NCP solvent prior
        x2_input = x2_output.unsqueeze(1)
        predicted_steps, hidden_state = [], torch.cat((group_updated, global_updated, C), dim=1)
        for _ in range(3):
            output, hidden_state = self.NCP1(x2_input, hidden_state)
            x2_input = output
            predicted_steps.append(output.view(output.size(0), -1))
        x2_output = torch.cat(predicted_steps, dim=-1)
        x2_output = self.relu(self.x22(x2_output))
        xm = self.xm3(torch.cat((x2_output, x1, global_updated), dim=1))
        xm = self.relu(xm)

        gamma1 = self.c1_gamma(global_G); beta1 = self.c1_beta(global_G)
        xm_film = gamma1 * xm + beta1

        #Atomic group bridge, achieving characteristic aggregation from atoms to groups
        xm_film, xg_after, type_ids_local = self.atom_group_bridge(
            x_atom=xm_film,
            atom_idx=a2g_local[1],
            x_group=xg_local,
            group_idx=a2g_local[0],
            edge_index_group=eig_local,
            cond_atom=global_G,
            edge_attr_group=None
        )

        edge_attr_group =None

        #Concentration Aware module

        hidden = torch.cat((xm_film,xm_film),dim=1)
        xm_catC = torch.cat((xm_film, C), dim=1).unsqueeze(1)
        _, hidden = self.NCP2(xm_catC, hidden)

        subgraph_x = self.subgraph_conv1(xm_film, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)

        subgraph_x1 = torch.cat((subgraph_x, C), dim=1).unsqueeze(1)
        _, hidden = self.NCP2(subgraph_x1, hidden)

        gamma2 = self.c2_gamma(C); beta2 = self.c2_beta(C)
        x_film2 = gamma2 * subgraph_x + beta2

        subgraph_x = self.subgraph_conv2(x_film2, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)

        subgraph_x2 = torch.cat((subgraph_x, C), dim=1).unsqueeze(1)
        _, hidden = self.NCP2(subgraph_x2, hidden)

        x_film3 = gamma2 * subgraph_x + beta2

        subgraph_x = self.subgraph_conv2(x_film3, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)

        subgraph_x3 = torch.cat((subgraph_x, C), dim=1).unsqueeze(1)
        subgraph_x3,_ = self.NCP2(subgraph_x3, hidden)
        subgraph_x3 = subgraph_x3.squeeze(1)

        # readout
        subgraph_x = self.set2set(subgraph_x3, batch[mask])
        group = global_mean_pool(inter, batch[mask])
        group_batch = _groups_batch_from_a2g_local(xg_after, a2g_local, batch[mask])  # [Gm]
        s1 = subgraph_x3
        x2_outputs = group
        C_values = C[:, 0].detach().cpu().numpy()
        atom_types_sub = C
        return subgraph_x, xg_after, group_batch, group,atom_types_sub,s1,C_values, type_ids_local

    def forward(self, data):
        device = data.x.device
        K = 3
        global_edge_attrall = data.global_edge_attr.to(device)
        global_node_attr = data.global_node_attr.to(device)
        global_edge_attr = global_edge_attrall[:, 0:4]

        s1, xg_after1, group_batch1, grp1,_,_,_, type_ids1 = self.process_subgraph(data, data.mask1, mol_id=0)
        s2, xg_after2, group_batch2, grp2,_,_,_, type_ids2 = self.process_subgraph(data, data.mask2, mol_id=1)
        s3, xg_after3, group_batch3, grp3,_,_,_, type_ids3 = self.process_subgraph(data, data.mask3, mol_id=2)

        xg_list = [xg_after1, xg_after2,xg_after3]  # [Gi, H]
        gb_list = [group_batch1, group_batch2, group_batch3]  # [Gi]
        self._last_label_pack = {
            "types": [type_ids1, type_ids2, type_ids3],
            "batches": [group_batch1, group_batch2, group_batch3],
        }

        self._last_groups = [xg_after1, xg_after2, xg_after3]
        per_mol_cross, mix_feat, attn_info = self.cross_group_attn(xg_list, gb_list, return_attn=True)

        H = per_mol_cross[0].size(1)
        B_sub = per_mol_cross[0].size(0) if per_mol_cross[0].numel() > 0 else 1
        cross_stack = torch.empty((B_sub * self.K, H), device=device)
        cross_stack[0::self.K] = per_mol_cross[0]
        cross_stack[1::self.K] = per_mol_cross[1]
        cross_stack[2::self.K] = per_mol_cross[2]



        batch_size = s1.size(0); feat_dim = s1.size(1)
        s1 = self.relu(s1); s2 = self.relu(s2); s3 = self.relu(s3)

        expanded_x = torch.empty((batch_size * K, feat_dim), dtype=s1.dtype, device=device)
        expanded_x[0::K] = s1; expanded_x[1::K] = s2; expanded_x[2::K] = s3


        '''gamma3 = self.c3_gamma(global_node_attr); beta3 = self.c3_beta(global_node_attr)
        expanded_x = gamma3 * expanded_x + beta3'''



        expanded_x = torch.cat((expanded_x, global_node_attr), dim=1)

        group = torch.empty((batch_size * 3, 32), dtype=s1.dtype, device=device)
        group[0::K] = grp1; group[1::K] = grp2; group[2::K] = grp3



        group = torch.cat((group,cross_stack), dim=1)

        '''gamma4 = self.c4_gamma(global_node_attr); beta4 = self.c4_beta(global_node_attr)
        group = gamma4 * group + beta4'''

        group = torch.cat((group,global_node_attr), dim=1)

        def make_bidir_pairs_edge_index(K: int, batch_size: int, device):
            pairs = []
            for i in range(K - 1):
                for j in range(i + 1, K):
                    pairs.extend([(i, j), (j, i)])
            local_src = torch.tensor([u for (u, v) in pairs], dtype=torch.long, device=device)
            local_dst = torch.tensor([v for (u, v) in pairs], dtype=torch.long, device=device)
            per_graph_edges = local_src.numel()
            base = torch.arange(batch_size, device=device).repeat_interleave(per_graph_edges) * K
            src = base + local_src.repeat(batch_size)
            dst = base + local_dst.repeat(batch_size)
            return torch.stack([src, dst], dim=0)

        global_edge_index = make_bidir_pairs_edge_index(K, batch_size, device)

        combined_x = self.global_conv(expanded_x, global_edge_index, global_edge_attr.to(device))
        combined_x = self.relu(combined_x)
        combined_x = torch.cat((combined_x, expanded_x), dim=1)

        batch_vec = torch.repeat_interleave(torch.arange(batch_size, device=device), K)
        set2set_x = self.set2set2(combined_x, batch_vec)


        expand_group = self.setgroup(group, batch_vec)

        expand_group = torch.cat((mix_feat,expand_group),dim=1)

        set2set_x_shape = set2set_x.size()
        expanded_set2set_x = set2set_x.unsqueeze(1).expand(-1, 3, -1)
        expanded_set2set_x = expanded_set2set_x.contiguous().view(-1, set2set_x_shape[1])
        expand_group_shape = expand_group.size()
        expand_group = expand_group.unsqueeze(1).expand(-1, 3, -1)
        expand_group = expand_group.contiguous().view(-1, expand_group_shape[1])

        group_out = torch.cat((expanded_x,expanded_set2set_x, expand_group,group), dim=1)

        #final_x = torch.cat((group_out), dim=1)
        output = self.fc(group_out)
        return output, s1,attn_info