import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set,AttentiveFP,global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen
from torch_geometric.utils import subgraph as pyg_subgraph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ncps.torch import CfC,LTC
from ncps.wirings import AutoNCP
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
import math
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool,GCNConv,GeneralConv

triple_csv_path = '/data/Solubility.csv'


from torch_geometric.data import Data

class MixData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def num_nodes(self):
        return self.x.size(0) if hasattr(self, "x") and self.x is not None else super().num_nodes

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes

        if key == 'edge_index_group':
            G = self.x_group.size(0) if hasattr(self, 'x_group') and self.x_group is not None else 0
            return torch.tensor([[G], [G]], dtype=torch.long)

        if key == 'atom2group_index':
            G = self.x_group.size(0) if hasattr(self, 'x_group') and self.x_group is not None else 0
            N = self.num_nodes
            return torch.tensor([[G], [N]], dtype=torch.long)

        if key == 'global_edge_index':
            return torch.tensor([[4], [4]], dtype=torch.long)

        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'edge_index_group', 'atom2group_index', 'global_edge_index']:
            return 1
        return 0

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




def process_molecule_hg(smiles):
    if not isinstance(smiles, str):
        raise ValueError(f"SMILES must be string, got {type(smiles)}: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # ------- 你的全局分子特征 -------
    mol = Chem.AddHs(mol)
    num_donors    = rdMolDescriptors.CalcNumHBD(mol)
    num_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    logp          = Crippen.MolLogP(mol)
    tpsa          = rdMolDescriptors.CalcTPSA(mol)

    # ------- 你的原子特征 -------
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_featurizer.encode(atom))
    node_features = torch.tensor(node_features, dtype=torch.float32)  # [Na, Da0]

    # 附加元素物性（你原逻辑原样保留）
    rows, _ = node_features.shape
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

        # Te
        elif node_features[i, 15] == 1:
            node_features[i, -1] = 2.1
            node_features[i, -2] = 135
            node_features[i, -3] = 52
            node_features[i, -4] = 127.6
            node_features[i, -5] = 9.010
            node_features[i, -6] = 1.971

    node_features = torch.tensor(node_features, dtype=torch.float32)

    # ------- 原子边 -------
    edges, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = bond_featurizer.encode(bond)
        edges.append([i, j]); edge_features.append(e)
        edges.append([j, i]); edge_features.append(e)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_features, dtype=torch.float32)

    # ------- 你的基团 SMARTS 定义（原样保留）-------
    functional_groups_smarts = {
        "hydroxyl": "[OX2H]",
        "carboxyl": "C(=O)O",
        "amine": "c(=O)",
        "phenyl": "c1ccccc1",
        "aldehyde": "C=O[!#8]",
        "amide": "C(=O)N",
        "nitrile": "C#N",
        "sulfhydryl": "C",
        "sulfone": "S(=O)(=O)",
        "heavy_atom_effect": "[P]",  # 增强自旋轨道耦合

        "alkyne": "C#C",
        "nitro": "O=[N+]([O-])",
        "alkene": "C=C",
        "8": "[OH2]",
        "杂原子–O–C（左非碳）": "[a]-O-[#6]",
        "杂原子–O–C（左非碳）2": "[a]-O-[a]",
        "ether": "C-O-C",
        "halide": "[F,Cl,Br,I]",
        "14": "N#N",
        "15": "[Si]",

        "16": "[N;X3;!+;!$([N]=*);!$([N]#*);!$([N]-C(=O));!$([N]-[S](=O)=O)]",
        "17": "[P;X3;!+;!$([P]=*);!$([P]#*)]",
        "18": "C=S",
        "phosphate": "[!#6]=O",
        "3": "[S;X2;!+;!$([S]=*);!$([S]#*)]",
        "磷酰（P=O，通用）": "[#15;X4;v5](=O)",
        "S=C=S（硫-碳-硫累积）": "[#16]=C=[#16]",
        "亚砜（S=O，非砜）": "[#16X3;!+](=O)([#6])[#6]",
        "N=N": "N=N",

        "aromatic_5_ring": "a1aaaa1",
        "aromatic_6_ring": "a1aaaaa1",

        "aromatic_hetero_in_ring": "[a;!#6;r]",
        "pyridine_like_N": "[nX2;r;H0]",  # 吡啶型
        "pyrrole_like_N": "[nH;r]",  # 吡咯型 N-H
        "aromatic_N_positive": "[n+;r]",  # 芳香环内 N+
        "O=C=S（氧-碳-硫累积）": "O=C=[#16]",
        "异硫氰酸基（–N=C=S）": "N=C=S",
        "烷基叠氮（R–N3）": "[#6]-N=[N+]=[N-]",
        "B": "[B]",
        "H": "[H]",

    }
    patt_dict = {name: Chem.MolFromSmarts(s) for name, s in functional_groups_smarts.items()}
    group_names = list(patt_dict.keys())
    N = node_features.size(0)

    # -------（保留你原先的原子×基团类型 one-hot）-------
    group_membership = torch.zeros((N, len(group_names)), dtype=torch.float32)
    for g_idx, name in enumerate(group_names):
        patt = patt_dict[name]
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            for atom_idx in match:
                group_membership[atom_idx, g_idx] = 1.0
    node_features = torch.cat((node_features, group_membership), dim=1)

    # ------- 基团“实例”节点（超图的关键）：每次匹配 = 一个基团节点 -------
    group_nodes = []               # [(name, [atom_ids]), ...]
    group_type_oh = []             # 每个实例的类型 one-hot
    for name, patt in patt_dict.items():
        if patt is None:
            continue
        t = torch.zeros(len(group_names))
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            group_nodes.append((name, list(match)))
            t = torch.zeros(len(group_names)); t[group_names.index(name)] = 1.0
            group_type_oh.append(t.unsqueeze(0))
    if len(group_nodes) == 0:
        print(f"Warning: SMILES {smiles} has no functional group matches. Adding a default group node.")
    if len(group_nodes) > 0:
        group_type_oh = torch.cat(group_type_oh, dim=0)    # [Gm, n_types]
    else:
        group_type_oh = torch.empty(0, len(group_names))

    # 基团节点的初始特征：类型 one-hot + 成员原子的原子特征均值
    if len(group_nodes) > 0:
        g_from_atoms = []
        for _, members in group_nodes:
            g_from_atoms.append(node_features[members, :].mean(dim=0, keepdim=True))
        g_from_atoms = torch.cat(g_from_atoms, dim=0)      # [Gm, D_atom_ext]
        x_group = torch.cat([group_type_oh, g_from_atoms], dim=1)  # [Gm, n_types + D_atom_ext]
    else:
        x_group = torch.empty(0, len(group_names) + node_features.size(1))

    # 原子–基团 二部边（超图关联）
    gi, ai = [], []
    for gid, (_n, members) in enumerate(group_nodes):
        for a in members:
            gi.append(gid); ai.append(a)
    atom2group_index = torch.tensor([gi, ai], dtype=torch.long) if gi else torch.empty(2,0, dtype=torch.long)

    # 基团–基团边：将原子键“收缩”到基团层
    # 映射 原子 -> 参与的基团实例列表
    atom2groups = defaultdict(list)
    for gid, (_n, members) in enumerate(group_nodes):
        for a in members:
            atom2groups[a].append(gid)
    # 原子键(i,j) → 基团对(gi,gj)
    gg_src, gg_dst = [], []
    erow, ecol = edge_index
    for i, j in zip(erow.tolist(), ecol.tolist()):
        if i == j:
            continue
        Gi, Gj = atom2groups.get(i, []), atom2groups.get(j, [])
        for g1 in Gi:
            for g2 in Gj:
                if g1 != g2:
                    gg_src.append(g1); gg_dst.append(g2)
    if gg_src:
        edge_index_group = torch.tensor([gg_src, gg_dst], dtype=torch.long)
        edge_index_group = coalesce(edge_index_group, num_nodes=len(group_nodes))
    else:
        edge_index_group = torch.empty(2,0, dtype=torch.long)

    # ------- 分子全局特征（保留你的做法）-------
    functional_groups_count = {key: 0 for key in functional_groups_smarts.keys()}
    for name, patt in patt_dict.items():
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            functional_groups_count[name] = len(matches)
    global_features2 = torch.tensor(list(functional_groups_count.values()), dtype=torch.float32).unsqueeze(0)
    global_features1 = torch.tensor([num_donors, num_acceptors, logp, tpsa], dtype=torch.float32).unsqueeze(0)
    global_features  = torch.cat((global_features1, global_features2), dim=1)

    # 原子拼接全局（保留你的做法）
    global_features_repeated = global_features.repeat(N, 1)
    node_features = torch.cat([node_features, global_features_repeated], dim=1)

    return Data(
        x=node_features, edge_index=edge_index, edge_attr=edge_attr,
        x_group=x_group, edge_index_group=edge_index_group,
        atom2group_index=atom2group_index,
        global_features=global_features
    )
def combine_molecules_hg(smiles1, smiles2, smiles3, smiles4, x1=None, x2=None, x3=None, x4=None, T=None):
    g1 = process_molecule_hg(smiles1)
    g2 = process_molecule_hg(smiles2)
    g3 = process_molecule_hg(smiles3)
    g4 = process_molecule_hg(smiles4)

    # ------- 浓度/温度拼接到原子（保留你的写法）-------
    x1 = torch.tensor([x1,0], dtype=torch.float32)
    x2 = torch.tensor([x2,0], dtype=torch.float32)
    x3 = torch.tensor([x3,0], dtype=torch.float32)
    x4 = torch.tensor([x4,1], dtype=torch.float32)
    T  = torch.tensor([T],     dtype=torch.float32)

    def add_conc(x_atom, c): return torch.cat([x_atom, c.expand(x_atom.size(0), -1)], dim=1)
    g1x, g2x, g3x, g4x = add_conc(g1.x, x1), add_conc(g2.x, x2), add_conc(g3.x, x3), add_conc(g4.x, x4)

    # ------- 原子层 offset & 拼接 -------
    off_a1 = 0
    off_a2 = off_a1 + g1x.size(0)
    off_a3 = off_a2 + g2x.size(0)
    off_a4 = off_a3 + g3x.size(0)

    combined_x = torch.cat([g1x, g2x, g3x, g4x], dim=0)
    combined_edge_index = torch.cat([
        g1.edge_index + off_a1,
        g2.edge_index + off_a2,
        g3.edge_index + off_a3,
        g4.edge_index + off_a4
    ], dim=1)
    combined_edge_attr = torch.cat([g1.edge_attr, g2.edge_attr, g3.edge_attr, g4.edge_attr], dim=0)

    # ------- 基团层 offset & 拼接（关键）-------
    G1, G2, G3, G4 = g1.x_group.size(0), g2.x_group.size(0), g3.x_group.size(0), g4.x_group.size(0)
    off_g1, off_g2, off_g3, off_g4 = 0, G1, G1+G2, G1+G2+G3

    def empty_idx():
        return torch.empty(2, 0, dtype=torch.long)

    if (G1+G2+G3+G4) > 0:
        x_group_all = torch.cat([g1.x_group, g2.x_group, g3.x_group, g4.x_group], dim=0)

        eig_all = torch.cat([
            (g1.edge_index_group + off_g1) if g1.edge_index_group.numel()>0 else empty_idx(),
            (g2.edge_index_group + off_g2) if g2.edge_index_group.numel()>0 else empty_idx(),
            (g3.edge_index_group + off_g3) if g3.edge_index_group.numel()>0 else empty_idx(),
            (g4.edge_index_group + off_g4) if g4.edge_index_group.numel()>0 else empty_idx(),
        ], dim=1)

        a2g_parts = []
        if G1>0 and g1.atom2group_index.numel()>0:
            a2g_parts.append(torch.stack([g1.atom2group_index[0] + off_g1, g1.atom2group_index[1] + off_a1], dim=0))
        if G2>0 and g2.atom2group_index.numel()>0:
            a2g_parts.append(torch.stack([g2.atom2group_index[0] + off_g2, g2.atom2group_index[1] + off_a2], dim=0))
        if G3>0 and g3.atom2group_index.numel()>0:
            a2g_parts.append(torch.stack([g3.atom2group_index[0] + off_g3, g3.atom2group_index[1] + off_a3], dim=0))
        if G4>0 and g4.atom2group_index.numel()>0:
            a2g_parts.append(torch.stack([g4.atom2group_index[0] + off_g4, g4.atom2group_index[1] + off_a4], dim=0))
        atom2group_index_all = torch.cat(a2g_parts, dim=1) if len(a2g_parts)>0 else empty_idx()

        group_mol_id = torch.tensor(
            ([0]*G1) + ([1]*G2) + ([2]*G3) + ([3]*G4), dtype=torch.long
        )
        gmask1 = torch.zeros(G1+G2+G3+G4, dtype=torch.bool); gmask1[:G1] = (G1>0)
        gmask2 = torch.zeros_like(gmask1); gmask2[G1:G1+G2] = (G2>0)
        gmask3 = torch.zeros_like(gmask1); gmask3[G1+G2:G1+G2+G3] = (G3>0)
        gmask4 = torch.zeros_like(gmask1); gmask4[G1+G2+G3:] = (G4>0)
    else:
        x_group_all = torch.empty(0, 1, dtype=torch.float32)
        eig_all = empty_idx()
        atom2group_index_all = empty_idx()
        group_mol_id = torch.empty(0, dtype=torch.long)
        gmask1 = gmask2 = gmask3 = gmask4 = torch.empty(0, dtype=torch.bool)

    # ------- 你的混合物“4分子图”保持不变（global_*）-------
    global_features1 = torch.cat([g1.global_features.flatten(), x1], dim=0)
    global_features2 = torch.cat([g2.global_features.flatten(), x2], dim=0)
    global_features3 = torch.cat([g3.global_features.flatten(), x3], dim=0)
    global_features4 = torch.cat([g4.global_features.flatten(), x4], dim=0)
    global_features1 = global_features1.flatten()
    global_features2 = global_features2.flatten()
    global_features3 = global_features3.flatten()
    global_features4 = global_features4.flatten()

    # 扩展除前4维外的 cross 全局特征（保留你的写法）
    g1exp = torch.cat((global_features2[None,4:].expand(g1.x.size(0), -1),
                       global_features3[None,4:].expand(g1.x.size(0), -1),
                       global_features4[None,4:].expand(g1.x.size(0), -1)), dim=1)
    g2exp = torch.cat((global_features1[None,4:].expand(g2.x.size(0), -1),
                       global_features3[None,4:].expand(g2.x.size(0), -1),
                       global_features4[None,4:].expand(g2.x.size(0), -1)), dim=1)
    g3exp = torch.cat((global_features1[None,4:].expand(g3.x.size(0), -1),
                       global_features2[None,4:].expand(g3.x.size(0), -1),
                       global_features4[None,4:].expand(g3.x.size(0), -1)), dim=1)
    g4exp = torch.cat((global_features1[None,4:].expand(g4.x.size(0), -1),
                       global_features2[None,4:].expand(g4.x.size(0), -1),
                       global_features3[None,4:].expand(g4.x.size(0), -1)), dim=1)
    global_features_nodes = torch.cat((g1exp, g2exp, g3exp, g4exp), dim=0)
    combined_x = torch.cat((combined_x, global_features_nodes), dim=1)

    global_edge_index = torch.tensor(
        [[0,1,0,2,0,3,1,2,1,3,2,3],
         [1,0,2,0,3,0,2,1,3,1,3,2]], dtype=torch.long
    )
    global_edge_attr = torch.cat(
        [g1.global_features, g2.global_features,
         g1.global_features, g3.global_features,
         g1.global_features, g4.global_features,
         g2.global_features, g3.global_features,
         g2.global_features, g4.global_features,
         g3.global_features, g4.global_features], dim=0
    )
    global_node_attr = torch.stack([
        torch.cat([x1, x2, x3, x4, T]),
        torch.cat([x2, x1, x3, x4, T]),
        torch.cat([x3, x1, x2, x4, T]),
        torch.cat([x4, x1, x2, x3, T]),
    ], dim=0)

    # ------- 原子层分子掩码（保留）-------
    offset1 = g1x.size(0); offset2 = offset1 + g2x.size(0); offset3 = offset2 + g3x.size(0)
    m1 = torch.zeros(combined_x.size(0), dtype=torch.bool); m1[:offset1] = True
    m2 = torch.zeros_like(m1); m2[offset1:offset2] = True
    m3 = torch.zeros_like(m1); m3[offset2:offset3] = True
    m4 = torch.zeros_like(m1); m4[offset3:] = True

    # ------- 基团层分子掩码（新增，可选）-------


    return MixData(
        # 原子层
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,

        # 基团层（新增）
        x_group=x_group_all,
        edge_index_group=eig_all,
        atom2group_index=atom2group_index_all,
        group_mol_id=group_mol_id,
        group_mask1=gmask1, group_mask2=gmask2, group_mask3=gmask3, group_mask4=gmask4,

        # 分子/混合物层（保留）
        global_edge_index=global_edge_index,
        global_edge_attr=global_edge_attr,
        global_node_attr=global_node_attr,

        # 原子层掩码（保留）
        mask1=m1, mask2=m2, mask3=m3, mask4=m4
    )

def load_data(triple_csv_path):
    triple_df = pd.read_csv(triple_csv_path)

    def clean_smiles_column(col):
        return [str(s).strip() if pd.notnull(s) else '' for s in triple_df[col]]

    smiles1 = clean_smiles_column('solv1_smiles')
    smiles2 = clean_smiles_column('solv2_smiles')
    smiles3 = clean_smiles_column('solv3_smiles')
    smiles4 = clean_smiles_column('solv4_smiles')

    targets = triple_df['y'].tolist()


    solv1_x_triple = triple_df['solv1_x'].tolist()
    solv2_x_triple = triple_df['solv2_x'].tolist()
    solv3_x_triple = triple_df['solv3_x'].tolist()
    solv4_x_triple = triple_df['solv4_x'].tolist()

    T = triple_df['T_K'].tolist()



    concentrations = list(zip(solv1_x_triple, solv2_x_triple, solv3_x_triple,solv4_x_triple,T))

    return smiles1, smiles2, smiles3,smiles4, targets, concentrations
class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, smiles1, smiles2, smiles3,smiles4, targets, concentrations, transform=None, pre_transform=None):
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.smiles3 = smiles3
        self.smiles4 = smiles4
        self.targets = targets
        self.concentrations = concentrations



        super(MoleculesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        return ['Mixsol.csv']

    @property
    def processed_file_names(self):
        return ['Mixsol.pt']

    def download(self):
        pass

    def process(self):
        datas = []
        for i in range(len(self.smiles1)):
            s1, s2, s3, s4 = self.smiles1[i], self.smiles2[i], self.smiles3[i], self.smiles4[i]
            target = self.targets[i]
            c1, c2, c3, c4, temp = self.concentrations[i]
            try:
                data = combine_molecules_hg(s1, s2, s3, s4, c1, c2, c3, c4, temp)
            except ValueError as e:
                print(f"[跳过样本] SMILES 错误: {e}")
                continue
            if data is not None:
                data.y = torch.tensor(target, dtype=torch.float32)
                datas.append(data)
            else:
                print(f"Sample {i} invalid, skipped.")
        torch.save(self.collate(datas), self.processed_paths[0])

smiles1, smiles2, smiles3,smiles4, targets, concentrations = load_data(triple_csv_path)

dataset = MoleculesDataset(root='Mixsol', smiles1=smiles1, smiles2=smiles2, smiles3=smiles3,smiles4=smiles4, targets=targets, concentrations=concentrations)

class FeatureCrossAttention(nn.Module):
    def __init__(self, dim_in_q, dim_in_kv, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = 32

        # 首先把原特征维度映射到 model_dim
        self.q_map = nn.Linear(dim_in_q, 128)
        self.k_map = nn.Linear(dim_in_kv, 128)
        self.v_map = nn.Linear(dim_in_kv, 128)

        # Attention 后再投回 model_dim
        self.out_map = nn.Linear(128, model_dim)
        self.Qout = nn.Linear(128, model_dim)
        # 输出再映射回原dim_in_q
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

        # ========== 4) 合并头 ==========
        out_h = out_h.permute(0, 3, 1, 2).contiguous().view(B, L_q, self.num_heads * self.d_k)  # (B, L_q, model_dim)

        # ========== 5) 输出映射 ==========
        out = self.out_map(out_h)
        Qm_ = self.Qout(Qm)
        out = self.norm(Qm_ + out)

        return out, attn

print(len(dataset))
from torch_geometric.nn import NNConv, GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Set2Set, global_mean_pool, NNConv
from torch_geometric.utils import subgraph as pyg_subgraph
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

# =============== 工具：局部 a2g/g2a scatter ===============
def atoms_to_groups_local(x_atom, atom_idx, group_idx, G, reduce='mean'):
    """
    x_atom:    [Na, Ha] 当前子图原子表示
    atom_idx:  [N_inc]  每条原子->基团“归属”使用的 原子局部索引
    group_idx: [N_inc]  每条“归属”的 基团局部索引
    G:         基团数
    """
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
    """
    x_group:   [G, Hg]
    group_idx: [N_inc]
    atom_idx:  [N_inc]
    N:         原子数
    """
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
    """
    对 batch 中存在的“包”做 Set2Set，然后把结果回填到固定大小 size 的输出中。
    缺失的包返回全零向量。
    features: [N_items, D]
    batch:    [N_items]，取值范围在 [0, size-1]（可能有缺失的 id）
    size:     目标包数量（固定输出行数）
    s2s:      Set2Set 模块（不带 size 参数）
    返回: [size, 2D]
    """

    if size == 0:
        # 没有任何包时，返回 [0, 2D]（保持维度语义）
        D = features.size(1) if features.numel() > 0 else 0
        return features.new_zeros((0, 2 * D))

    if features.numel() == 0 or batch.numel() == 0:
        # 有包但没有元素属于它们 -> 全零
        D = features.size(1) if features.numel() > 0 else 0
        return features.new_zeros((size, 2 * D))

    # 1) 只对实际出现的包做紧致映射：present_ids -> [0..P-1]
    present = torch.unique(batch)                      # [P]
    P = int(present.numel())
    # 建立 old_id -> new_id 映射表（长度=size，缺失为 -1）
    id_map = -torch.ones(size, dtype=torch.long, device=batch.device)
    id_map[present] = torch.arange(P, device=batch.device)
    compact_batch = id_map[batch]                      # [N_items] in [0..P-1]

    # 2) 在紧致批上跑 Set2Set
    out_compact = s2s(features, compact_batch)         # [P, 2D]

    # 3) 回填到固定大小 size 的输出
    out = features.new_zeros((size, out_compact.size(1)))
    out[present] = out_compact
    return out
# =============== Atom<->Group 桥（无边特征） ===============
class AtomGroupBridgeFiLM(nn.Module):
    def __init__(self, atom_dim, group_dim, cond_dim, hidden=160,s2s_steps: int = 2):
        super().__init__()
        self.a2g_proj = nn.Linear(atom_dim, group_dim)
        self.g2a_proj = nn.Linear(group_dim, atom_dim)
        self.g_proj = nn.Linear(40, group_dim-80)

        # FiLM 调制
        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, group_dim)
        )
        self.film_beta  = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, group_dim)
        )
        self.a_proj_to_g = nn.Linear(atom_dim, group_dim)
        # Set2Set 聚合（A->G, G->A 用两个实例，互不共享参数）
        self.s2s_a2g = Set2Set(group_dim, processing_steps=s2s_steps)   # 输出 2*Dg
        self.merge_a2g = nn.Linear(2 * group_dim, group_dim)            # 2*Dg -> Dg

        # （可选）基团级 GCN
        self.group_gcn = GeneralConv(group_dim, group_dim-80,attention=True)

        # G->A：Set2Set 聚合回原子，再映射回 Ha
        self.s2s_g2a = Set2Set(group_dim, processing_steps=s2s_steps)   # 输出 2*Dg
        self.g_proj_to_a = nn.Linear(2 * group_dim, atom_dim)           # 2*Dg -> Ha


    def forward(self, x_atom, atom_idx, x_group, group_idx, edge_index_group, cond_atom,edge_attr_group=None):
        device = x_atom.device
        x_group = x_group
        Na, Ha = x_atom.size(0), x_atom.size(1)

        Gm, Dg = x_group.size(0), x_group.size(1)
        x_group = x_group[:, 0:40]
        x_group = self.g_proj(x_group)

        if Gm == 0 or atom_idx.numel() == 0 or group_idx.numel() == 0:
            # 保证返回的 xg 形状是 [0, Dg]
            xg_empty = x_atom.new_zeros((0, Dg))
            return x_atom, xg_empty

        xa_proj = self.a_proj_to_g(x_atom)  # [Na, Dg]
        # 取归属边上的原子表示，形成“实例-包”的 items
        xa_items = xa_proj.index_select(0, atom_idx)  # [N_inc, Dg]
        # 用 group_idx 作为 batch，把每个基团的原子集合打包
        xg_a2g = set2set_pool(xa_items, group_idx, size=Gm, s2s=self.s2s_a2g)  # [Gm, 2*Dg]
        xg_from_atom = self.merge_a2g(xg_a2g)  # [Gm, Dg]




        # 2) 条件聚合
        cond_g = atoms_to_groups_local(cond_atom, atom_idx, group_idx, Gm, reduce='mean')

        # 3) FiLM 调制
        if Gm > 0:
            gamma = self.film_gamma(cond_g)                        # [Gm, Dg]
            beta  = self.film_beta(cond_g)                         # [Gm, Dg]
            xg    = gamma * xg_from_atom  + beta   # [Gm, Dg]
        else:
            xg    = xg_from_atom  # [0, Dg] 安全路径
        # 4) 基团图
        if Gm > 0 and (edge_index_group is not None) and (edge_index_group.numel() > 0):
            xg = F.relu(self.group_gcn(xg, edge_index_group))



        xg = torch.cat((x_group,xg),dim=1)

        xg_items = xg.index_select(0, group_idx)  # [N_inc, Dg]
        # 用 atom_idx 作为 batch，把每个原子的基团集合打包
        xa_g2a = set2set_pool(xg_items, atom_idx, size=Na, s2s=self.s2s_g2a)  # [Na, 2*Dg]
        xa_from_group = self.g_proj_to_a(xa_g2a)

        xa_out = x_atom+xa_from_group

        return xa_out, xg


from torch_geometric.nn import global_mean_pool
def pool_groups_per_graph(xg_local, a2g_local, batch_sub):
    """
    xg_local:   [Gm, Dg]          该子图（该分子）的所有基团向量（按 batch 拼成一个大块）
    a2g_local:  [2, N_inc]        (group_local_idx, atom_local_idx)
    batch_sub:  [Na]              该子图内每个“局部原子”的 batch 索引(0..B_sub-1)
    return:     [B_sub, Dg]       每个样本（在该mask下）一条基团全局向量（平均）
    """
    device = xg_local.device
    Dg = xg_local.size(1) if xg_local.numel() > 0 else 1
    if batch_sub.numel() == 0:
        return torch.zeros((0, Dg), device=device)
    B_sub = int(batch_sub.max().item()) + 1

    if xg_local.numel() == 0:
        return torch.zeros((B_sub, Dg), device=device)

    # 用 a2g 的“原子 batch”来给每个 group 打上它属于哪个 batch 的标签
    group_batch = torch.full((xg_local.size(0),), -1, dtype=torch.long, device=device)
    if a2g_local.numel() > 0:
        g_idx = a2g_local[0]                # [N_inc]
        a_idx = a2g_local[1]                # [N_inc]
        a_batch = batch_sub.index_select(0, a_idx)  # [N_inc]
        # 用第一次出现的 batch 作为该 group 的 batch（一个 group 不会跨样本）
        # 为了简单起见，使用“只写不回滚”的方式
        group_batch[g_idx] = torch.where(group_batch[g_idx] < 0, a_batch, group_batch[g_idx])

    # 对于没有成员的 group，默认丢到 batch 0；也可以改成忽略
    group_batch = torch.where(group_batch < 0, torch.zeros_like(group_batch), group_batch)

    # scatter 到 [B_sub, Dg]
    out = torch.zeros((B_sub, Dg), device=device)
    cnt = torch.zeros((B_sub, 1), device=device)
    out.index_add_(0, group_batch, xg_local)
    cnt.index_add_(0, group_batch, torch.ones((xg_local.size(0),1), device=device))
    out = out / cnt.clamp_min_(1.0)
    return out
#@torch.no_grad()
class GroupPoolSet2Set(torch.nn.Module):
    def __init__(self, in_dim: int, processing_steps: int = 2, num_layers: int = 1):
        super().__init__()
        # 注意：PyG 的 Set2Set 输出维度 = 2 * in_dim
        self.set2set = Set2Set(in_dim, processing_steps=processing_steps, num_layers=num_layers)

    @torch.no_grad()
    def _safe_mode(self, x: torch.Tensor) -> torch.Tensor:
        # torch.mode 在全部相同或空时也要安全返回
        if x.numel() == 0:
            return torch.tensor(0, device=x.device, dtype=x.dtype)
        return x.mode()[0]

    def forward(self,
                xg_local: torch.Tensor,    # [Gm, Dg]
                a2g_local: torch.Tensor,   # [2, N_inc] 行0=group_idx, 行1=atom_idx(局部)
                batch_sub: torch.Tensor    # [Na_sub]   子图原子 -> 子图内batch id
                ) -> torch.Tensor:
        """
        返回：group_pool [B_sub, 2*Dg]（严格与 batch_sub 的样本数对齐）
        - 若某个样本没有任何基团，返回该行全零；
        - 若有基团但没出现在 a2g_local，也会被分配到默认 batch（batch_sub 的众数）。
        """
        device = batch_sub.device
        Dg = int(xg_local.size(1)) if xg_local.numel() > 0 else 0

        # 子图里的样本数（行数目标）
        if batch_sub.numel() == 0:
            return xg_local.new_zeros((1, 2 * Dg))
        B_sub = int(batch_sub.max().item()) + 1

        # 没有任何基团：直接返回全零（B_sub 行）
        if xg_local.numel() == 0:
            return xg_local.new_zeros((B_sub, 2 * Dg))

        # ==== 构造 group -> batch_sub 的映射 ====
        Gm = xg_local.size(0)
        group_batch_local = torch.full((Gm,), -1, dtype=torch.long, device=device)

        if a2g_local.numel() > 0:
            g_idx = a2g_local[0]  # [N_inc]
            a_idx = a2g_local[1]  # [N_inc]（原子是子图局部索引）
            group_batch_local[g_idx] = batch_sub[a_idx]

        # 仍然存在未被映射的基团，给默认 batch（众数）；保证每个基团都有合法 batch id
        if (group_batch_local < 0).any():
            default_b = self._safe_mode(batch_sub.long())
            group_batch_local[group_batch_local < 0] = default_b

        # ==== 紧致化 present batch id，避免 Set2Set 产生“对不齐” ====
        present = torch.unique(group_batch_local)                 # [P]，真实出现的 batch id
        P = int(present.numel())
        id_map = -torch.ones(B_sub, dtype=torch.long, device=device)
        id_map[present] = torch.arange(P, device=device)          # 原始 id -> 紧致 id [0..P-1]
        compact_batch = id_map[group_batch_local]                 # [Gm] in [0..P-1]

        # ==== 在紧致批上跑 Set2Set ====
        # out_compact: [P, 2*Dg]
        out_compact = self.set2set(xg_local, compact_batch)

        # ==== 回填到固定大小 [B_sub, 2*Dg]，缺失 id 自动为零 ====
        out = xg_local.new_zeros((B_sub, out_compact.size(1)))
        out[present] = out_compact
        return out

class CrossMolGroupInter(nn.Module):
    """
    跨分子基团交互注意力（提速版）：
    - 一次性拼出所有 token（三个分子 * 全部样本），
      用 pad_sequence 构成 [B_sub, L_max, H] 的批，配合 key_padding_mask 调一次 MHA。
    - per-molecule / per-mixture 读出使用不同的 Set2Set 聚合。
    返回:
      per_mol_out: list 长度 K，每个 [B_sub, group_dim]
      mix_feat:     [B_sub, 2*in_dim] （Set2Set 聚合）
    """
    def __init__(self, group_dim: int, K: int, mol_emb_dim: int = 16,
                 num_heads: int = 4, use_set2set: bool = True, s2s_steps: int = 2):
        super().__init__()
        self.K = K
        self.group_dim = group_dim
        self.mol_emb = nn.Linear(K, mol_emb_dim)
        self.in_dim = group_dim + mol_emb_dim

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

        self.readout = nn.Sequential(
            nn.Linear(self.in_dim*2, group_dim),
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
            # 🔁 通过 Linear 层映射成 embedding
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
        from torch.nn.utils.rnn import pad_sequence
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
            # 可选线性读出到 group_dim（与你原逻辑一致）
            per_mol_out.append(self.readout(s2s_i))  # [B_sub, group_dim]

        # 4.2 per-mixture（Set2Set 聚合混合物）：
        # 使用 Set2Set 聚合整个混合物的特征
        if self.use_set2set:
            mix_feat = self.mix_s2s(attn_unsorted, b_idx)          # [B_sub, 2*H_in]
        else:
            mix_feat = None

        if return_attn:
            # 计算每个 mixture 的起始下标，方便画图分割
            b_offsets = torch.zeros(B_sub + 1, dtype=torch.long, device=device)
            b_offsets[1:] = torch.cumsum(counts, dim=0)  # [B_sub+1]，第 i 个 mixture 的范围是 [b_offsets[i], b_offsets[i+1])

            return per_mol_out, mix_feat, {
                "attn1": attn_mat1,               # [B_sub, h, L_max, L_max]
                "attn2": attn_mat2,               # [B_sub, h, L_max, L_max]
                "lengths": lens,                  # [B_sub]
                "counts": counts,                 # [B_sub]
                "b_offsets": b_offsets,           # [B_sub+1]
                "mol_sorted": mol_sorted,         # [N_tok_sorted]
            }

        return per_mol_out, mix_feat




# ====== 新增：把 a2g_local + batch_sub 映射成“本分子基团的紧致 batch id” ======
def _groups_batch_from_a2g_local(xg_local: torch.Tensor,
                                 a2g_local: torch.Tensor,
                                 batch_sub: torch.Tensor) -> torch.Tensor:
    """

    """
    device = xg_local.device
    Gm = xg_local.size(0)
    if Gm == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    group_batch_self = torch.full((Gm,), -1, dtype=torch.long, device=device)
    if a2g_local.numel() > 0:
        g = a2g_local[0]
        a = a2g_local[1]
        group_batch_self[g] = batch_sub[a]

    if (group_batch_self < 0).any():
        default_b = batch_sub.mode()[0] if batch_sub.numel() > 0 else torch.tensor(0, device=device)
        group_batch_self[group_batch_self < 0] = default_b

    # 紧致化到 0..B_sub-1
    present = torch.unique(batch_sub)
    # present 已经天然是 0..B_sub-1，如果你有非连续 id，这里再做一次 map 更保险
    id_map = -torch.ones(int(present.max().item()) + 1, dtype=torch.long, device=device)
    id_map[present] = torch.arange(present.numel(), device=device)
    compact = id_map[group_batch_self]
    return compact

# =============== 融合后的 MesoNet（不改你原有主干逻辑） ===============
class MesoNet(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim,
                 d_group_in, d_group_hidden=128):
        """
        d_group_in 必须传入 data.x_group.size(1)
        """
        super(MesoNet, self).__init__()

        # ======= 你原有的层（保持） =======
        self.K = 4
        self.mol_emb_dim = 16

        # 跨分子基团交互注意力（输入用基团维 hidden_dim）
        self.cross_group_attn = CrossMolGroupInter(
            group_dim=hidden_dim,  # 你的基团表示维度
            K=4,  # 三个分子
            mol_emb_dim=16,
            num_heads=4,
            use_set2set=True,
            s2s_steps=2
        )

        self.attn_atom_elem   = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.attn_group_atom  = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.attn_global_group= FeatureCrossAttention(dim_in_q=40, dim_in_kv=32, model_dim=32, num_heads=4)
        self.inter            = FeatureCrossAttention(dim_in_q=41, dim_in_kv=127, model_dim=32, num_heads=4)


        edge_hidden_dim = 32
        self.a11 = NNConv(41, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 41 * 32)
        ), aggr="mean")

        self.G = nn.Linear(21, 32)
        self.NCP1 = CfC(32, AutoNCP(66,32), batch_first=True)
        self.NCP2= CfC(162, AutoNCP(320,160), batch_first=True)
        self.x22 = nn.Linear(96,96)
        self.x2 = nn.Linear(6,32)
        self.a21 = NNConv(32, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 32 * 32)
        ), aggr="mean")

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

        self.global_conv = NNConv(hidden_dim*2+9, hidden_dim, nn.Sequential(
            nn.Linear(4, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim , 329*hidden_dim)
        ), aggr='mean')


        self.set2set  = Set2Set(hidden_dim, processing_steps=2)
        self.set2set2 = Set2Set(3*hidden_dim+9 , processing_steps=2)
        self.setgroup = Set2Set(233, processing_steps=2)

        self.group = nn.Linear(175,175)
        self.g = nn.Linear(21, 32)
        self.fc = nn.Sequential(
            nn.Linear(914, 1024),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 1)
        )
        self.lstm_a2_1 = CfC(32, AutoNCP(66,32), batch_first=True)
        self.NCP= CfC(162, AutoNCP(320,160), batch_first=True)

        # FiLM 参数
        self.c1_gamma = nn.Sequential(nn.Linear(168, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c1_beta  = nn.Sequential(nn.Linear(168, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c2_gamma = nn.Sequential(nn.Linear(168, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c2_beta  = nn.Sequential(nn.Linear(168, 160), nn.ReLU(), nn.Linear(160, 160))


        self.group2group = nn.Linear(hidden_dim*2,32)
        # ======= 新增：Atom<->Group 桥（无边特征） =======

        self.atom_group_bridge = AtomGroupBridgeFiLM(
            atom_dim=hidden_dim, group_dim=hidden_dim,cond_dim = 168, s2s_steps=2


        )
        self.group_pooler = GroupPoolSet2Set(in_dim=hidden_dim, processing_steps=2)

    @staticmethod
    def _slice_group_view(data, mol_id, atom_mask):
        """
        返回该分子的 group 局部视图：xg_local / a2g_local / eig_local
        a2g_local 的第二行（atom_idx）为该子图“原子局部索引”，可直接与子图张量对齐。
        """
        device = data.x.device
        if data.group_mol_id.numel() == 0:
            return None

        gid_global = torch.nonzero(data.group_mol_id == mol_id, as_tuple=False).view(-1)
        if gid_global.numel() == 0:
            return None

        # group 全局->局部
        gid_map = torch.full((int(data.group_mol_id.numel()),), -1, device=device, dtype=torch.long)
        gid_map[gid_global] = torch.arange(gid_global.numel(), device=device)

        # atom 全局->局部（子图）
        aid_global = torch.nonzero(atom_mask, as_tuple=False).view(-1)
        aid_map = torch.full((data.x.size(0),), -1, device=device, dtype=torch.long)
        aid_map[aid_global] = torch.arange(aid_global.numel(), device=device)

        # a2g 局部
        if data.atom2group_index.numel() > 0:
            g_idx_global = data.atom2group_index[0]
            a_idx_global = data.atom2group_index[1]
            keep = (gid_map[g_idx_global] >= 0) & (aid_map[a_idx_global] >= 0)
            g_idx_local = gid_map[g_idx_global[keep]]
            a_idx_local = aid_map[a_idx_global[keep]]
            a2g_local = torch.stack([g_idx_local, a_idx_local], dim=0)
        else:
            a2g_local = torch.empty(2, 0, dtype=torch.long, device=device)

        # eig 局部
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
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch

        # ===== 你的原流程：取该分子的原子子图 =====
        subgraph_x = x[mask]
        subgraph_edge_index, subgraph_edge_attr = pyg_subgraph(mask, edge_index, edge_attr, relabel_nodes=True)
        x1 = subgraph_x[:, 0:41]
        x1 = self.a11(x1, subgraph_edge_index, subgraph_edge_attr)
        x1 = self.relu(x1)
        x2 = subgraph_x[:, 41:47]
        x3 = subgraph_x[:, 47:47+40]

        g  = subgraph_x[:, 47+40+4:47+40+4+41]
        G_ = subgraph_x[:, 47+40+4+41:]
        C = g[:, -2:]

        x2_output = self.x2(x2)
        x2_output = self.relu(x2_output)

        # =======（保持）你的注意力交互 =======
        inter, _ = self.inter(g.unsqueeze(1), G_.unsqueeze(1))
        inter = inter.squeeze(1)
        global_G = torch.cat((g, G_), dim=1)

        global_updated, _ = self.attn_global_group(x3.unsqueeze(1), inter.unsqueeze(1))
        global_updated = global_updated.squeeze(1)

        group_updated, _ = self.attn_group_atom(x1.unsqueeze(1), global_updated.unsqueeze(1))
        group_updated = group_updated.squeeze(1)

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

        # ======= 插入：Atom<->Group 超图交互（无边特征）=======
        group_view = self._slice_group_view(data, mol_id, mask)
        xg_local  = group_view["xg_local"]
        a2g_local = group_view["a2g_local"]
        eig_local = group_view["eig_local"]
        xm_film, xg_after = self.atom_group_bridge(
            x_atom=xm_film,
            atom_idx=a2g_local[1],
            x_group=xg_local,
            group_idx=a2g_local[0],
            edge_index_group=eig_local,
            cond_atom=global_G,
            edge_attr_group=None
            # <--- 新增
        )

        # ======= 回到你的原子消息传递 + NCP =======
        hidden = torch.cat((xm_film, xm_film), dim=1)
        xm_catC = torch.cat((xm_film, C), dim=1).unsqueeze(1)
        _, hidden = self.NCP2(xm_catC, hidden)

        subgraph_x = self.subgraph_conv1(xm_film, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)

        subgraph_x1 = torch.cat((subgraph_x, C), dim=1).unsqueeze(1)
        _, hidden = self.NCP2(subgraph_x1, hidden)

        gamma2 = self.c2_gamma(global_G); beta2 = self.c2_beta(global_G)
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

        # readout（保持）
        subgraph_x = self.set2set(subgraph_x3, batch[mask])

        group = global_mean_pool(inter, batch[mask])
        group_pool = self.group_pooler(
            xg_local=xg_after,  # [Gm, Dg]
            a2g_local=a2g_local,  # [2, N_inc]
            batch_sub=batch[mask]  # [Na_sub]
        )
        group_pool = self.relu(self.group2group(group_pool))
        group_batch = _groups_batch_from_a2g_local(xg_after, a2g_local, batch[mask])  # [Gm]

        group = torch.cat((group, group_pool), dim=1)
        '''if group_view is not None and xg_after.numel() > 0:
            group_pool = pool_groups_per_graph(
                xg_local=xg_after,
                a2g_local=a2g_local,
                batch_sub=batch[mask]
            )  # [B_sub, Dg]
            group_pool = self.relu(self.group2group(group_pool))  # -> [B_sub, 32]
            group = torch.cat((group, group_pool), dim=1)'''
        x2_outputs = group
        return subgraph_x, xg_after, group_batch, group

    def forward(self, data):
        device = data.x.device
        K = 4
        global_edge_attrall = data.global_edge_attr.to(device)
        global_node_attr = data.global_node_attr.to(device)
        global_edge_attr = global_edge_attrall[:, 0:4]


        s1, xg_after1, group_batch1, grp1 = self.process_subgraph(data, data.mask1, mol_id=0)
        s2, xg_after2, group_batch2, grp2 = self.process_subgraph(data, data.mask2, mol_id=1)
        s3, xg_after3, group_batch3, grp3 = self.process_subgraph(data, data.mask3, mol_id=2)
        s4, xg_after4, group_batch4, grp4 = self.process_subgraph(data, data.mask4, mol_id=3)

        xg_list = [xg_after1, xg_after2,xg_after3,xg_after4]  # [Gi, H]
        gb_list = [group_batch1, group_batch2, group_batch3,group_batch4]  # [Gi]

        per_mol_cross, mix_feat, attn_info = self.cross_group_attn(xg_list, gb_list, return_attn=True)

        H = per_mol_cross[0].size(1)
        B_sub = per_mol_cross[0].size(0) if per_mol_cross[0].numel() > 0 else 1
        cross_stack = torch.empty((B_sub * self.K, H), device=device)
        cross_stack[0::self.K] = per_mol_cross[0]
        cross_stack[1::self.K] = per_mol_cross[1]
        cross_stack[2::self.K] = per_mol_cross[2]
        cross_stack[3::self.K] = per_mol_cross[3]

        batch_size = s1.size(0); feat_dim = s1.size(1)
        s1 = self.relu(s1); s2 = self.relu(s2); s3 = self.relu(s3); s4 = self.relu(s4)

        expanded_x = torch.empty((batch_size * K, feat_dim), dtype=s1.dtype, device=device)
        expanded_x[0::K] = s1; expanded_x[1::K] = s2; expanded_x[2::K] = s3; expanded_x[3::K] = s4
        expanded_x = torch.cat((expanded_x, global_node_attr), dim=1)

        group = torch.empty((batch_size * 4, 64), dtype=s1.dtype, device=device)
        group[0::4] = grp1; group[1::4] = grp2; group[2::4] = grp3; group[3::4] = grp4
        group = torch.cat((group,cross_stack), dim=1)



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

        final_x = torch.cat((expanded_x[3::4],group[3::4],mix_feat), dim=1)
        output = self.fc(final_x)
        return output, s1,attn_info






from sklearn.model_selection import KFold,StratifiedKFold
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import KFold, train_test_split

epochs = 350
k_folds = 5
batch_size =256
input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 160
output_dim = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_size = len(dataset)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
start_fold = 0
best_val_losses, best_val_maes, best_val_mses, best_val_r2s = [], [], [], []
test_rmse_list, test_mae_list, test_mse_list, test_r2_list = [], [], [], []

for fold, (train_idx, valtest_idx) in enumerate(kf.split(dataset)):
    if fold < start_fold:
        print(f"Skipping Fold {fold+1}")
        continue

    print(f"Start Fold {fold+1}/{k_folds}")

    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=0.5, random_state=42, shuffle=True
    )

    train_subset = [dataset[i] for i in train_idx]
    val_subset = [dataset[i] for i in val_idx]
    test_subset = [dataset[i] for i in test_idx]
    print(f"Fold {fold+1} ：")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")
    print(f"  总数:  {len(train_idx) + len(val_idx) + len(test_idx)}\n")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = MesoNet(input_dim, edge_dim, hidden_dim=160, output_dim=1,
                    d_group_in=160).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()

    best_val_rmse = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):



        model.train()
        y_train_true, y_train_pred = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output, _, attn_info  = model(batch)
            output = output.view(-1, 1)
            target = batch.y.unsqueeze(1).to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            y_train_true.extend(target.cpu().numpy().flatten())
            y_train_pred.extend(output.detach().cpu().numpy().flatten())

        train_mse = mean_squared_error(y_train_true, y_train_pred)
        train_rmse = math.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        train_r2 = r2_score(y_train_true, y_train_pred)

        # ---- validation ----
        model.eval()
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output, _ , _= model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)
                y_val_true.extend(target.cpu().numpy().flatten())
                y_val_pred.extend(output.cpu().numpy().flatten())

        val_mse = mean_squared_error(y_val_true, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)

        y_test_true, y_test_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output, _, _ = model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)
                y_test_true.extend(target.cpu().numpy().flatten())
                y_test_pred.extend(output.cpu().numpy().flatten())

        test_mse = mean_squared_error(y_test_true, y_test_pred)
        test_rmse = math.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            bsettest_mae, bsettest_rmse, bsettest_r2 = test_mae, test_rmse, test_r2

        # ---- 打印 ----
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"  Val   RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")




    test_rmse_list.append(bsettest_rmse)
    test_mae_list.append(bsettest_mae)
    test_r2_list.append(bsettest_r2)

    print(f"\nFold {fold+1} Best Epoch {best_epoch}")
    print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {bsettest_rmse:.4f}, Test MAE: {bsettest_mae:.4f}, Test R²: {bsettest_r2:.4f}")

print("\nAverage Results Across Folds:")
print(f"  Avg Test RMSE: {np.mean(test_rmse_list):.4f}, Avg Test MAE: {np.mean(test_mae_list):.4f}, Avg Test R²: {np.mean(test_r2_list):.4f}")

