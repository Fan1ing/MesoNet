from ncps.torch import CfC,LTC
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.loader import DataLoader
from torch import nn
from ncps.wirings import AutoNCP
from tqdm import tqdm
import torch
from ncps.torch import CfC,LTC
from torch_geometric.nn import NNConv, Set2Set,AttentiveFP,global_mean_pool
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,mean_absolute_percentage_error
from rdkit.Chem import rdMolDescriptors,Crippen
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from sklearn.metrics import mean_absolute_error, r2_score
from torch.serialization import add_safe_globals
import torch_geometric.data.data
import math
import torch.nn.functional as F

#Change to local address
# The absorption wavelength and emission wavelength use the same code, only the dataset is different.

csv_path = '/MesoNet/data/aboso.csv'

df = pd.read_csv(csv_path)

smiles1,smiles2,ys=df['smiles1'],df['smiles2'],df['y']

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

    def isInRing(self, atom):
        return atom.IsInRing()

    def isaromatic(self, atom):
        return atom.GetIsAromatic()

    def formal_charge(self, atom):
        return atom.GetFormalCharge()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

    def aromatic(self, bond):
        return bond.GetIsAromatic()

    def ring(self, bond):
        return bond.IsInRing()


# Initialize atom and bond featurizers
atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Cl", "F", "Ge", "H", "I", "N", "Na", "O", "P", "S", "Se", "Si", "Te"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "formal_charge": {-1, -2, 1, 2, 0},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "isInRing": {True, False},
        "isaromatic": {True, False},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "aromatic": {True, False},
        "ring": {True, False},
    }
)



import torch
from torch_geometric.nn import NNConv, Set2Set
import torch.nn as nn
from torch_geometric.utils import subgraph

def process_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)


    mol = Chem.AddHs(mol)
    num_donors = rdMolDescriptors.CalcNumHBD(mol)
    num_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_featurizer.encode(atom))
    node_features = torch.tensor(node_features, dtype=torch.float32)

    rows, cols = node_features.shape
    zeros_tensor = torch.zeros(rows, 6)
    node_features = torch.cat((node_features, zeros_tensor), dim=1)



    for i in range(rows):
        # B
        if node_features[i, 0] == 1:
            node_features[i, -1] = 2.04
            node_features[i, -2] = 82
            node_features[i, -3] = 5
            node_features[i, -4] = 10.82
            node_features[i, -5] = 8.298
            node_features[i, -6] = 0.277

        # Br
        elif node_features[i, 1] == 1:
            node_features[i, -1] = 2.96
            node_features[i, -2] = 114
            node_features[i, -3] = 35
            node_features[i, -4] = 79.904
            node_features[i, -5] = 11.814
            node_features[i, -6] = 3.364

        # C
        elif node_features[i, 2] == 1:
            node_features[i, -1] = 2.55
            node_features[i, -2] = 77
            node_features[i, -3] = 6
            node_features[i, -4] = 12.011
            node_features[i, -5] = 11.261
            node_features[i, -6] = 1.595

        # Cl
        elif node_features[i, 3] == 1:
            node_features[i, -1] = 3.16
            node_features[i, -2] = 99
            node_features[i, -3] = 17
            node_features[i, -4] = 35.45
            node_features[i, -5] = 12.968
            node_features[i, -6] = 3.62

        # F
        elif node_features[i, 4] == 1:
            node_features[i, -1] = 3.98
            node_features[i, -2] = 71
            node_features[i, -3] = 9
            node_features[i, -4] = 18.998
            node_features[i, -5] = 17.422
            node_features[i, -6] = 3.40

        # Ge
        elif node_features[i, 5] == 1:
            node_features[i, -1] = 2.01
            node_features[i, -2] = 122
            node_features[i, -3] = 32
            node_features[i, -4] = 72.63
            node_features[i, -5] = 7.90
            node_features[i, -6] = 1.23

        # H
        elif node_features[i, 6] == 1:
            node_features[i, -1] = 2.20
            node_features[i, -2] = 37
            node_features[i, -3] = 1
            node_features[i, -4] = 1.008
            node_features[i, -5] = 13.598
            node_features[i, -6] = 0.755

        # I
        elif node_features[i, 7] == 1:
            node_features[i, -1] = 2.66
            node_features[i, -2] = 133
            node_features[i, -3] = 53
            node_features[i, -4] = 126.9
            node_features[i, -5] = 10.451
            node_features[i, -6] = 3.060

        # N
        elif node_features[i, 8] == 1:
            node_features[i, -1] = 3.04
            node_features[i, -2] = 75
            node_features[i, -3] = 7
            node_features[i, -4] = 14.007
            node_features[i, -5] = 14.534
            node_features[i, -6] = 0.07

        # Na
        elif node_features[i, 9] == 1:
            node_features[i, -1] = 0.93
            node_features[i, -2] = 154
            node_features[i, -3] = 11
            node_features[i, -4] = 22.99
            node_features[i, -5] = 5.139
            node_features[i, -6] = 0.547

        # O
        elif node_features[i, 10] == 1:
            node_features[i, -1] = 3.44
            node_features[i, -2] = 73
            node_features[i, -3] = 8
            node_features[i, -4] = 15.999
            node_features[i, -5] = 13.618
            node_features[i, -6] = 1.46

        # P
        elif node_features[i, 11] == 1:
            node_features[i, -1] = 2.19
            node_features[i, -2] = 106
            node_features[i, -3] = 15
            node_features[i, -4] = 30.974
            node_features[i, -5] = 10.487
            node_features[i, -6] = 0.75

        # S
        elif node_features[i, 12] == 1:
            node_features[i, -1] = 2.58
            node_features[i, -2] = 102
            node_features[i, -3] = 16
            node_features[i, -4] = 32.06
            node_features[i, -5] = 10.36
            node_features[i, -6] = 2.07

        # Se
        elif node_features[i, 13] == 1:
            node_features[i, -1] = 2.55
            node_features[i, -2] = 116
            node_features[i, -3] = 34
            node_features[i, -4] = 78.971
            node_features[i, -5] = 9.753
            node_features[i, -6] = 2.02

        # Si
        elif node_features[i, 14] == 1:
            node_features[i, -1] = 1.90
            node_features[i, -2] = 111
            node_features[i, -3] = 14
            node_features[i, -4] = 28.085
            node_features[i, -5] = 8.151
            node_features[i, -6] = 1.385

        #Te
        elif node_features[i,15] == 1:
            node_features[i,-1] = 2.1
            node_features[i,-2] = 135
            node_features[i,-3] = 52
            node_features[i,-4] = 127.6
            node_features[i,-5] = 9.010
            node_features[i,-6] = 1.971

    node_features = torch.tensor(node_features, dtype=torch.float32)
    functional_groups_smarts = {

        "hydroxyl": "[OX2H]",
        "carboxyl": "C(=O)O",
        "amine": "[NX3;H2,H1;!$(NC=O)]",
        "ester": "C(=O)O[C]",
        "phenyl": "c1ccccc1",
        "aldehyde": "C=O",
        "ketone": "C(=O)C",
        "methyl": "C",
        "amide": "C(=O)N",
        "nitrile": "C#N",
        "sulfhydryl": "[C-SH]",
        "sulfone": "S(=O)(=O)C",
        "phosphate": "P(=O)(O)O",
        "halide": "[F,Cl,Br,I]",
        "acetal": "C(O)C",
        "alkyne": "C#C",
        "nitro": "N(=O)=O",
        "ether": "C-O-C",
        "alkene": "C=C",
        "quaternary_amine": "[N+](C)(C)",
        "B-": "[B-]",

    }
    functional_groups_patterns = {
        name: Chem.MolFromSmarts(smarts)
        for name, smarts in functional_groups_smarts.items()
    }
    group_names = list(functional_groups_patterns.keys())
    num_groups = len(group_names)
    N = node_features.size(0)
    group_membership = torch.zeros((N, num_groups), dtype=torch.float32)

    for g_idx, name in enumerate(group_names):
        patt = functional_groups_patterns[name]
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            for atom_idx in match:
                group_membership[atom_idx, g_idx] = 1.0

    node_features = torch.cat((node_features,group_membership), dim=1)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_features.append(bond_featurizer.encode(bond))
        edge_features.append(bond_featurizer.encode(bond))

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)


    functional_groups_count = {key: 0 for key in functional_groups_smarts.keys()}

    for name, smarts in functional_groups_smarts.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS mode: {smarts}")
        matches = mol.GetSubstructMatches(patt)
        if matches:
            functional_groups_count[name] = len(matches)

    #  Add functional group counts to the feature vector
    global_features2 = torch.tensor(list(functional_groups_count.values()), dtype=torch.float32).unsqueeze(0)
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Combine all features (such as num_donors, num_acceptors, logp, tpsa) into one vector
    global_features1 = torch.tensor([num_donors, num_acceptors, logp, tpsa,num_aromatic_rings], dtype=torch.float32).unsqueeze(0)

    # Combine global features
    global_features = torch.cat((global_features1, global_features2), dim=1)
    num_nodes = node_features.size(0)

    global_features_repeated = global_features.repeat(num_nodes, 1)

    node_features = torch.cat([node_features, global_features_repeated], dim=1)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)

def combine_molecules(smiles1, smiles2):

    graph1 = process_molecule(smiles1)
    graph2 = process_molecule(smiles2)


    offset = graph1.x.size(0)
    graph2.edge_index += offset


    combined_x = torch.cat([graph1.x, graph2.x], dim=0)

    global_features1 = graph1.global_features
    global_features2 = graph2.global_features

    global_features1 = global_features1.flatten() if global_features1.dim() > 1 else global_features1
    global_features2 = global_features2.flatten() if global_features2.dim() > 1 else global_features2

    # 扩展 global_features1 和 global_features2 的行数，使其匹配 graph1.x 和 graph2.x 的行数
    global_features1_expanded = global_features2[None, 5:].expand(graph1.x.size(0), -1)
    global_features2_expanded = global_features1[None, 5:].expand(graph2.x.size(0), -1)
    global_features = torch.cat((global_features1_expanded, global_features2_expanded), dim=0)

    combined_x = torch.cat((combined_x, global_features), dim=1)




    combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index], dim=1)
    combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)


    mask1 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    mask2 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    mask1[:graph1.x.size(0)] = True
    mask2[graph1.x.size(0):] = True


    global_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    global_edge_attr = torch.cat(
        [graph1.global_features, graph2.global_features], dim=0
    )

    return Data(
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        global_edge_index=global_edge_index,
        global_edge_attr=global_edge_attr,
        mask1=mask1,
        mask2=mask2
    )

class FeatureCrossAttention(nn.Module):
    def __init__(self, dim_in_q, dim_in_kv, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = 1

        self.q_map = nn.Linear(dim_in_q, model_dim)
        self.k_map = nn.Linear(dim_in_kv, model_dim)
        self.v_map = nn.Linear(dim_in_kv, model_dim)

        self.out_map = nn.Linear(model_dim, model_dim)

        self.norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale   = 1 / math.sqrt(self.d_k)

    def forward(self, Q_in, KV_in, mask=None):
        """
        Q_in:   (B, L_q, dim_in_q)
        KV_in:  (B, L_kv, dim_in_kv)
        mask:   optional BoolTensor of shape (B, L_q, L_kv)
        """
        B, L_q, Dq = Q_in.shape
        _, L_kv, Dk = KV_in.shape

        # ========== 1) map to attention space ==========
        Qm = self.q_map(Q_in)      # (B, L_q, model_dim)
        Km = self.k_map(KV_in)     # (B, L_kv,model_dim)
        Vm = self.v_map(KV_in)     # (B, L_kv,model_dim)

        # ========== 2) 维度互换 ==========
        # Swap the sequence length <-> feature dimensions
        # Qm: (B, L_q, M) -> (B, M, L_q)
        Qs = Qm.transpose(1,2)
        # Km, Vm: (B, L_kv, M) -> (B, M, L_kv)
        Ks = Km.transpose(1,2)
        Vs = Vm.transpose(1,2)

        # ========== 3)  Attention ==========
        Qh = Qs.view(B, 32, self.num_heads, self.d_k).transpose(1,2)  # (B,H,M/H,L_q)
        Kh = Ks.view(B, 32, self.num_heads, self.d_k).transpose(1,2)  # (B,H,M/H,L_kv)
        Vh = Vs.view(B, 32, self.num_heads, self.d_k).transpose(1,2)  # (B,H,M/H,L_kv)

        #  scores
        scores = torch.matmul(Qh, Kh.transpose(-2,-1)) * self.scale         # (B,H,M/H,M/H?) depending dims
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out_h  = torch.matmul(attn, Vh)                                     # (B,H,M/H,L_q)
        out_h  = out_h.transpose(1,2).contiguous().view(B, 32, L_q)  # (B,M,L_q)

        out_s  = out_h.transpose(1,2)                                       # (B, L_q, model_dim)

        out = self.out_map(out_s)                                         # (B, L_q, model_dim)
        out = self.norm(Qm + out)

        return out, attn
class MesoNet(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(MesoNet, self).__init__()
        edge_hidden_dim = 32
        self.relu = nn.ReLU()
        self.x1 = nn.Linear(41, 32)
        self.x2 = nn.Linear(30, 32)

        self.x3 = nn.Linear(21, 32)
        self.x12 = nn.Linear(64, 64)
        self.x13 = nn.Linear(64, 64)
        self.a11 = NNConv(41, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 41 * 32)
        ), aggr="mean")
        self.xm2 = CfC(96, 96, batch_first=True)
        self.G = nn.Linear(21,32)
        self.transformer_layer1 = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=32, dropout=0.2)
        self.transformer_encoder1 = TransformerEncoder(self.transformer_layer1, num_layers=2)


        self.decoder_layer = TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=2)
        self.lstm_a2_1 = CfC(6, AutoNCP(12, 6), batch_first=True)
        self.x22 = nn.Linear(30, 32)  # 5-step * feature 6 -> 32
        self.a21 = NNConv(32, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 32 * 32)
        ), aggr="mean")

        self.g_conv = NNConv(21, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 21 * 32)
        ), aggr="mean")

        self.group_proj = nn.Linear(21, 128)
        self.g = nn.Linear(21, 32)
        self.group = nn.Linear(21, 128)
        self.attn_atom_elem = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=1)
        self.attn_group_atom = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=1)
        self.attn_global_group = FeatureCrossAttention(dim_in_q=21, dim_in_kv=32, model_dim=32, num_heads=1)
        self.inter = FeatureCrossAttention(dim_in_q=21, dim_in_kv=21, model_dim=32, num_heads=1)

        self.xm = nn.Linear(hidden_dim, hidden_dim)
        self.set2set2 = Set2Set(3*hidden_dim, processing_steps=2)
        self.set2set = Set2Set(hidden_dim, processing_steps=2)
        self.subgraph1_x = nn.Linear(hidden_dim*2,hidden_dim*2)
        self.subgraph2_x = nn.Linear(hidden_dim*2,hidden_dim*2)
        self.MIX = CfC(hidden_dim*2, hidden_dim*2, batch_first=True)
        self.xm3 = nn.Linear(hidden_dim,hidden_dim)
        self.setgroup = Set2Set(44, processing_steps=2)
        self.group = nn.Linear(44*2+22,128)
        self.combine = nn.Linear(hidden_dim*2,hidden_dim*2)
        self.FF = nn.Linear(6*hidden_dim,128)
        self.combine = CfC(hidden_dim*2,hidden_dim*2, batch_first=True)
        self.sub1 = NNConv(hidden_dim , hidden_dim, nn.Sequential(

            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')

        self.sub2 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')
        self.set2set = Set2Set(160, processing_steps=2)
        self.sub3 = AttentiveFP(in_channels=hidden_dim, hidden_channels= hidden_dim,out_channels=hidden_dim*2, edge_dim=edge_dim, num_layers=2 , num_timesteps=2,dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*4,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, output_dim)
        )
        self.global_conv = NNConv(hidden_dim*2,hidden_dim, nn.Sequential(
            nn.Linear(4, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(edge_hidden_dim , hidden_dim*2*hidden_dim)
        ), aggr='mean')

        self.global_conv2 = NNConv(32,32, nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4 , 32*32)
        ), aggr='mean')

        self.set2set2 = Set2Set(3*hidden_dim, processing_steps=2)
        self.set2set = Set2Set(hidden_dim, processing_steps=2)
        self.subgraph1_x = nn.Linear(hidden_dim*2,hidden_dim*2)
        self.subgraph2_x = nn.Linear(hidden_dim*2,hidden_dim*2)
        self.MIX = CfC(hidden_dim*2, hidden_dim*2, batch_first=True)
        self.xm3 = nn.Linear(hidden_dim,hidden_dim)
        self.setgroup = Set2Set(64, processing_steps=2)
        self.group = nn.Linear(64*2+32,hidden_dim)
        self.combine = nn.Linear(hidden_dim*2,hidden_dim*2)
        self.FF = nn.Linear(6*hidden_dim,hidden_dim)
        self.combine = CfC(hidden_dim*2,hidden_dim*2, batch_first=True)
        self.trans = nn.Linear(96,96)
    def process_subgraph(self, x, edge_index, edge_attr, batch, mask):

        subgraph_x = x[mask]
        subgraph_edge_index, subgraph_edge_attr= subgraph(mask, edge_index, edge_attr, relabel_nodes=True)
        #hi-b
        x1 = subgraph_x[:, :41]
        #hi-a
        x2 = subgraph_x[:, 41:47]
        #hi-c
        x3 = subgraph_x[:, 47:47+21]
        #Hgi
        g = subgraph_x[:, 47+21+5:47+21+5+21]
        #Hgj ：other molecule in mix system
        G =subgraph_x[:, 47+21+5+21:]



        #Elemental Feature Extraction Module(NCP)
        x2_input = x2.unsqueeze(1)
        predicted_steps = []
        hidden_state =torch.cat((x2,x2),dim=1)

        for _ in range(5):
            output, hidden_state = self.lstm_a2_1(x2_input,hidden_state)
            predicted_steps.append(output.view(output.size(0), -1))

        x2_output = torch.cat(predicted_steps, dim=-1)
        x2_output = self.x22(x2_output)
        x2_output = self.relu(x2_output)
        x2_output = self.a21(x2_output, subgraph_edge_index, subgraph_edge_attr)
        x2 = self.relu(x2_output)
        x1 = self.relu(self.a11(x1, subgraph_edge_index, subgraph_edge_attr))


        # Cross-Attention
        inter, _ = self.inter(
            g.unsqueeze(1),
            G.unsqueeze(1)
        )
        inter = inter.squeeze(1)
        #Mixture → molecule (group level)

        global_updated, _ = self.attn_global_group(
            x3.unsqueeze(1),
            inter.unsqueeze(1)
        )
        global_updated = global_updated.squeeze(1)
        #Group assignment → atomic environment

        group_updated, _ = self.attn_group_atom(
            x1.unsqueeze(1),
            global_updated.unsqueeze(1)
        )
        group_updated = group_updated.squeeze(1)
        #Environment → intrinsic physical features of atoms

        atom_updated, _  = self.attn_atom_elem(
            x2.unsqueeze(1),
            group_updated.unsqueeze(1)
        )
        atom_updated = atom_updated.squeeze(1)

        x = torch.cat((x1,x2),dim =1)
        xx =self.relu(self.trans(torch.cat(( global_updated,group_updated,atom_updated),dim =1)))
        x = torch.cat((x,xx),dim =1)
        x = self.relu(self.xm(x))


        # Intramolecular message transmission
        x = self.relu(self.sub1(x, subgraph_edge_index, subgraph_edge_attr))
        x = self.relu(self.sub2(x, subgraph_edge_index, subgraph_edge_attr))
        #x = self.sub2(x, edge_index, edge_attr)

        subgraph_x = self.sub3(x, subgraph_edge_index, subgraph_edge_attr,batch[mask])
        group = global_mean_pool(inter, batch[mask])
        return subgraph_x,x1,x2,group




    def forward(self, data):

        global_edge_attrall = data.global_edge_attr.to(data.x.device)
        global_edge_attr = global_edge_attrall[:, 0:4]
        group = global_edge_attrall[:, 4:]
        subgraph1_x,x2_outputs,x2out,group1 = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask1)
        subgraph2_x,_,_,group2 = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask2)
        subgraph1_x = self.relu(subgraph1_x)
        subgraph2_x = self.relu(subgraph2_x)
        batch_size = subgraph1_x.size(0)

        # global_edge_index
        new_global_edge_index = torch.empty((2, batch_size * 2), dtype=torch.long, device=subgraph1_x.device)
        for i in range(batch_size):
            new_global_edge_index[0][i * 2] = i * 2     # 0, 2, ...
            new_global_edge_index[1][i * 2] = i * 2 + 1 # 1, 3, ...
            new_global_edge_index[0][i * 2 + 1] = i * 2 + 1 # 1, 3, ...
            new_global_edge_index[1][i * 2 + 1] = i * 2 # 0, 2, ...

        global_edge_index = new_global_edge_index.view(2, batch_size * 2)

        expanded_x = torch.empty((batch_size * 2, subgraph1_x.size(1)), dtype=subgraph1_x.dtype, device=subgraph1_x.device)
        expanded_x[0::2] = subgraph1_x
        expanded_x[1::2] = subgraph2_x

        group = torch.empty((batch_size * 2, group1.size(1)), dtype=group1.dtype, device=group1.device)

        group[0::2] = group1
        group[1::2] = group2

        # Inter-molecular message transmission
        # Molecular Scale:
        combined_x = self.global_conv(expanded_x, global_edge_index, global_edge_attr.to(expanded_x.device))
        num_classes = batch_size
        repeats_per_class = 2
        combined_x = self.relu(combined_x)
        tensor = torch.cat([torch.full((repeats_per_class,), i, dtype=torch.long) for i in range(num_classes)])
        tensor = tensor.to(combined_x.device)
        combined_x = torch.cat((combined_x,expanded_x),dim=1)

        # Readout:
        set2set_x = self.set2set2(combined_x, tensor)
        set2set_x_shape = set2set_x.size()
        expanded_set2set_x = set2set_x.unsqueeze(1).expand(-1, 2, -1)

        expanded_set2set_x = expanded_set2set_x.contiguous().view(-1, set2set_x_shape[1])

        expanded_set2set_x = self.FF(expanded_set2set_x[0::2])
        expanded_set2set_x = self.relu(expanded_set2set_x)


        # Group Scale:
        expand_group = self.global_conv2(group, global_edge_index, global_edge_attr.to(expanded_x.device))
        expand_group = self.relu(expand_group)
        expand_group = self.setgroup(torch.cat((group,expand_group),dim=1), tensor)
        expand_group_shape = expand_group.size()

        expand_group = expand_group.unsqueeze(1).expand(-1, 2, -1)

        expand_group = expand_group.contiguous().view(-1, expand_group_shape[1])


        group = torch.cat((group,expand_group),dim =1)
        group = self.group(group)
        group = self.relu(group)
        expanded_x = expanded_x[0::2]
        m = expanded_x
        FAN = torch.cat((expanded_set2set_x,group[0::2]),dim=1)
        expanded_x = expanded_x.unsqueeze(1)
        expanded_x,_ = self.combine(expanded_x, FAN)
        expanded_x =expanded_x.squeeze(1)
        expanded_x =self.relu(expanded_x)
        final_x = torch.cat((m,expanded_x), dim=1)

        # final prediction

        output = self.fc(final_x)


        return output


from torch_geometric.data import InMemoryDataset, Data

class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, smiles1, smiles2, ys, transform=None, pre_transform=None):

        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.ys = ys
        super(MoleculesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):

        return ['aboso.csv']

    @property
    def processed_file_names(self):

        return ['aboso.pt']

    def download(self):

        pass

    def process(self):

        datas = []
        for smile1, smile2, y in zip(self.smiles1, self.smiles2, self.ys):

            data = combine_molecules(smile1, smile2)
            data.y = torch.tensor([y], dtype=torch.float32)
            datas.append(data)

        torch.save(self.collate(datas), self.processed_paths[0])
dataset = MoleculesDataset(root="aboso", smiles1=smiles1, smiles2=smiles2, ys=ys)
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - valid_size  # 注意 valid_size 改为 0.1

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(9)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  pin_memory=True, num_workers=10)
val_loader   = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, pin_memory=True, num_workers=10)

print(f"Train size: {len(train_dataset)},  Valid size: {len(valid_dataset)},  Test size: {len(test_dataset)}")


'''train_dataset = dataset[0:660]+dataset[661:1562]+dataset[1566:1786]+dataset[1787:]
valid_dataset = dataset[660:661]+dataset[1562:1563]+dataset[1786:1787]+dataset[1622:1623]'''
'''train_dataset = dataset[0:1622]+dataset[1623:]
valid_dataset = dataset[1622:1623]'''
'''train_dataset = dataset[0:692]+dataset[693:1620]+dataset[1621:1845]+dataset[1846:]
valid_dataset = dataset[692:693]+dataset[1620:1621]+dataset[1845:1846]'''

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=10)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)

print(len(train_dataset ))
print(len(valid_dataset))


input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 160
edge_hidden_dim = 32
output_dim = 1



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MesoNet(input_dim, edge_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


epochs = 400
early_stopping_counter = 0
patience = 10
mae = []
r = []


def mean_relative_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))

# Modify the existing training loop to track the best model
best_epoch = 0  # Initialize the epoch with the best result
best_model_state = None  # To store the best model state
best_val_loss = float('inf')
best_val_mae = float('inf')
best_val_mre = float('inf')
best_val_r2 = float('inf')

best_val_rmse    = float('inf')
best_model_state = None
best_epoch       = 0

for epoch in range(1, epochs+1):
    # ——— 1. 训练 ———
    model.train()
    train_loss = 0
    y_tr_true, y_tr_pred = [], []
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out   = model(batch).view(-1,1) * 1000
        tgt   = batch.y.unsqueeze(1).to(device) * 1000
        loss  = criterion(out, tgt)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs
        y_tr_true.extend(tgt.cpu().numpy().flatten())
        y_tr_pred.extend(out.detach().cpu().numpy().flatten())

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_mae  = mean_absolute_error(y_tr_true, y_tr_pred)
    train_mre  = mean_relative_error(np.array(y_tr_true), np.array(y_tr_pred))
    train_r2   = r2_score(y_tr_true, y_tr_pred)

    # ——— 2. 验证 ———
    model.eval()
    val_loss = 0
    y_val_true, y_val_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out   = model(batch).view(-1,1) * 1000
            tgt   = batch.y.unsqueeze(1).to(device) * 1000
            loss  = criterion(out, tgt)
            val_loss += loss.item() * batch.num_graphs
            y_val_true.extend(tgt.cpu().numpy().flatten())
            y_val_pred.extend(out.cpu().numpy().flatten())

    avg_val_mse = val_loss / len(val_loader.dataset)
    val_rmse    = np.sqrt(avg_val_mse)
    val_mae     = mean_absolute_error(y_val_true, y_val_pred)
    val_mre     = mean_relative_error(np.array(y_val_true), np.array(y_val_pred))
    val_r2      = r2_score(y_val_true, y_val_pred)

    # 保存最优模型


    # ——— 3. 测试 ———
    test_loss = 0
    y_test_true, y_test_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch).view(-1,1) * 1000
            tgt   = batch.y.unsqueeze(1).to(device) * 1000
            loss  = criterion(out, tgt)
            test_loss += loss.item() * batch.num_graphs
            y_test_true.extend(tgt.cpu().numpy().flatten())
            y_test_pred.extend(out.cpu().numpy().flatten())

    avg_test_mse = test_loss / len(test_loader.dataset)
    test_rmse    = np.sqrt(avg_test_mse)
    test_mae     = mean_absolute_error(y_test_true, y_test_pred)
    test_mre     = mean_relative_error(np.array(y_test_true), np.array(y_test_pred))
    test_r2      = r2_score(y_test_true, y_test_pred)

    if val_mae < best_val_rmse:
        best_val_rmse    = val_mae
        best_epoch       = test_mae

        best_model_state = model.state_dict()


    # ——— 打印本轮所有结果 ———
    print(f"Epoch {epoch}/{epochs}")
    print(f"  Train → Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, MRE: {train_mre:.4f}, R²: {train_r2:.4f}")
    print(f"  Valid → RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, MRE: {val_mre:.4f}, R²: {val_r2:.4f}")
    print(f"  Test  → RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MRE: {test_mre:.4f}, R²: {test_r2:.4f}")

# —— 训练结束，打印最优模型信息并保存 ——
print("\n=== Best Model Summary ===")
print(f"  Best Epoch: {best_epoch}, Valid RMSE: {best_val_rmse:.4f}")
torch.save(best_model_state, 'best_model.pth')

import matplotlib.pyplot as plt
import seaborn as sns

# 取 atom->elem 这个模块
mod = model.attn_global_group

Wq = mod.q_map.weight.detach().cpu()  # (M, Dq)
Wk = mod.k_map.weight.detach().cpu()  # (M, Dk)
Corr = Wq.t() @ Wk
# 3) 提取映射权重


# 4) 投影后的激活
import os
save_path = '/home/ubuntu/ffff.svg'
# 5) 计算原始特征空间的交互强度
#    channel_inter[i,j,d] = Qm[i,d] * Km[j,d]
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(
    Corr.numpy(),
    xticklabels=[f"K_feat{j}" for j in range(Corr.size(1))],
    yticklabels=[f"Q_feat{i}" for i in range(Corr.size(0))],
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Correlation"}
)
plt.title("Original‐Feature Correlation via Q/K Projection")
plt.xlabel("K 原始特征维度")
plt.ylabel("Q 原始特征维度")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Correlation heatmap saved to {save_path}")