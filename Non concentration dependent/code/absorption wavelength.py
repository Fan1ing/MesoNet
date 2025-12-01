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
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool,GCNConv,GeneralConv,AttentiveFP
import seaborn as sns
from adjustText import adjust_text
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import os

triple_csv_path = '/data/absorption wavelength.csv'
meta_df = pd.read_csv(triple_csv_path)



class MixData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def num_nodes(self):
        return self.x.size(0) if hasattr(self, "x") and self.x is not None else super().num_nodes

    def __inc__(self, key, value, *args, **kwargs):
        # PyG 会用它来决定 batch 拼接时每样本的“增量”
        if key == 'edge_index':
            # 原子图索引：按原子数递增（PyG 默认就是这样）
            return self.num_nodes

        if key == 'edge_index_group':
            # 基团-基团图：行/列都按该样本的 group 数递增
            G = self.x_group.size(0) if hasattr(self, 'x_group') and self.x_group is not None else 0
            return torch.tensor([[G], [G]], dtype=torch.long)

        if key == 'atom2group_index':
            # 二部图：(row=group, col=atom)
            G = self.x_group.size(0) if hasattr(self, 'x_group') and self.x_group is not None else 0
            N = self.num_nodes
            return torch.tensor([[G], [N]], dtype=torch.long)

        if key == 'global_edge_index':
            # 分子层 4 节点图：每样本 +4
            return torch.tensor([[4], [4]], dtype=torch.long)

        # 其他字段不需要增量（特征/掩码等）
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        # 索引形 [2, E] 的沿 dim=1 拼接；其它默认 dim=0
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

    functional_groups_smarts = {
        # ===== 疏水链 =====
        "hydroxyl": "[OX2H]",
        "carboxyl": "C(=O)O",
        "amine": "c(=O)",
        "phenyl": "c1ccccc1",
        "aldehyde": "C=O[!#8]",
        "amide": "C(=O)N",
        "nitrile": "C#N",
        "sulfhydryl": "C",
        "sulfone": "S(=O)(=O)",
        "acetal": "C(O)C",
        "alkyne": "C#C",
        "nitro": "O=[N+]([O-])",
        "alkene": "C=C",
        "8": "[OH2]",
        "杂原子–O–C": "[a]-O-[#6]",
        "杂原子–O–C2": "[a]-O-[a]",
        "ether": "C-O-C",
        "bodipy_BF2_core": "[B-](F)(F)",
        "9": "F",
        "10": "Cl",
        "11": "Br",
        "12": "I",
        "14": "C=N",
        "15": "[Si]",
        "16": "[N;X3;!+;!$([N]=*);!$([N]#*);!$([N]-C(=O));!$([N]-[S](=O)=O)]",
        "17": "[P;X3;!+;!$([P]=*);!$([P]#*)]",
        "18": "C=S",
        "phosphate": "[#15;X4;v5](=O)(-O)(-O)-O",
        "3": "[S;X2;!+;!$([S]=*);!$([S]#*)]",
        "磷酰（P=O，通用）": "[#15;X4;v5](=O)",
        "S=C=S（硫-碳-硫累积）": "[#16]=C=[#16]",
        "亚砜（S=O，非砜）": "[#16X3;!+](=O)([#6])[#6]",
        "硫酸根": "C(=O)[O-]",
        "N=N": "N=N",
        "trifluoromethyl": "C(F)(F)(F)",
        "heavy_atom_effect": "[Cl,Br,I]",
        "donor_amine": "N(C)(C)",
        "ar_bonded_O_anion": "a-[O-]",
        "aromatic_5_ring": "a1aaaa1",
        "aromatic_6_ring": "a1aaaaa1",
        "aromatic_hetero_in_ring": "[a;!#6;r]",
        "aromatic_N_in_ring": "[n;r]",
        "aromatic_O_in_ring": "[o;r]",
        "aromatic_S_in_ring": "[s;r]",
        "ar_bound_hetero": "a-[!#1;!#6]",
        "ar_bound_N": "a-[N;!$([N]=*);!$([N]#*)]",
        "ar_bound_S": "a-S",

        "pyridine_like_N": "[nX2;r;H0]",
        "pyrrole_like_N": "[nH;r]",
        "aromatic_N_positive": "[n+;r]",
        "aryl_quaternary_amine": "c-[N+](C)(C)C",
        "aryl_amide":"c-C(=O)N",
        "ar_vinylene": "c-C=C",
        "aryl_diarylamino":  "c[NX3;H0;!+](c)(c)",
    }

    patt_dict = {name: Chem.MolFromSmarts(s) for name, s in functional_groups_smarts.items()}
    group_names = list(patt_dict.keys())
    N = node_features.size(0)

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

    # ------- 基团“实例”节点（超图的关键）：
    group_nodes = []               # [(name, [atom_ids]), ...]
    group_type_oh = []             # one-hot
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

    # ------- 分子全局特征-------
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

    # 原子拼接全局
    global_features_repeated = global_features.repeat(N, 1)
    node_features = torch.cat([node_features, global_features_repeated], dim=1)

    return Data(
        x=node_features, edge_index=edge_index, edge_attr=edge_attr,
        x_group=x_group, edge_index_group=edge_index_group,
        atom2group_index=atom2group_index,
        global_features=global_features
    )
# ========= 2 组分 + 单标签 =========
def combine_molecules_hg_2(smiles1, smiles2, x1=None, x2=None,C=None):
    g1 = process_molecule_hg(smiles1)
    g2 = process_molecule_hg(smiles2)

    # --- 浓度拼到原子特征（每原子追加 1 维浓度标量）---
    x1 = torch.tensor([x1], dtype=torch.float32)
    x2 = torch.tensor([x2], dtype=torch.float32)


    def add_conc(x_atom, c):
        return torch.cat([x_atom, c.expand(x_atom.size(0), -1)], dim=1)

    g1x, g2x = add_conc(g1.x, x1), add_conc(g2.x, x2)

    # --- 原子层 offset & 拼接 ---
    off_a1 = 0
    off_a2 = off_a1 + g1x.size(0)
    combined_x = torch.cat([g1x, g2x], dim=0)
    combined_edge_index = torch.cat([
        g1.edge_index + off_a1,
        g2.edge_index + off_a2,
    ], dim=1)
    combined_edge_attr = torch.cat([g1.edge_attr, g2.edge_attr], dim=0)

    # --- 基团层 offset & 拼接 ---
    def empty_idx():
        return torch.empty(2, 0, dtype=torch.long)

    G1, G2 = g1.x_group.size(0), g2.x_group.size(0)
    off_g1, off_g2 = 0, G1

    if (G1 + G2) > 0:
        x_group_all = torch.cat([
            g1.x_group if g1.x_group.numel() > 0 else torch.empty(0, g2.x_group.size(1) if g2.x_group.numel()>0 else (g1.x_group.size(1) if g1.x_group.numel()>0 else 1), dtype=torch.float32),
            g2.x_group if g2.x_group.numel() > 0 else torch.empty(0, g1.x_group.size(1) if g1.x_group.numel()>0 else (g2.x_group.size(1) if g2.x_group.numel()>0 else 1), dtype=torch.float32),
        ], dim=0)

        eig_all = torch.cat([
            (g1.edge_index_group + off_g1) if g1.edge_index_group.numel() > 0 else empty_idx(),
            (g2.edge_index_group + off_g2) if g2.edge_index_group.numel() > 0 else empty_idx(),
        ], dim=1)

        a2g_parts = []
        if G1 > 0 and g1.atom2group_index.numel() > 0:
            a2g_parts.append(torch.stack([
                g1.atom2group_index[0] + off_g1,
                g1.atom2group_index[1] + off_a1
            ], dim=0))
        if G2 > 0 and g2.atom2group_index.numel() > 0:
            a2g_parts.append(torch.stack([
                g2.atom2group_index[0] + off_g2,
                g2.atom2group_index[1] + off_a2
            ], dim=0))
        atom2group_index_all = torch.cat(a2g_parts, dim=1) if len(a2g_parts) > 0 else empty_idx()

        group_mol_id = torch.tensor(([0] * G1) + ([1] * G2), dtype=torch.long)
        gmask1 = torch.zeros(G1 + G2, dtype=torch.bool); gmask1[:G1] = (G1 > 0)
        gmask2 = torch.zeros_like(gmask1);             gmask2[G1:] = (G2 > 0)
    else:
        x_group_all = torch.empty(0, 1, dtype=torch.float32)
        eig_all = empty_idx()
        atom2group_index_all = empty_idx()
        group_mol_id = torch.empty(0, dtype=torch.long)
        gmask1 = gmask2 = torch.empty(0, dtype=torch.bool)

    global_features1 = torch.cat([g1.global_features.flatten(), x1], dim=0).flatten()
    global_features2 = torch.cat([g2.global_features.flatten(), x2], dim=0).flatten()

    g1exp = global_features2[None, 4:].expand(g1.x.size(0), -1)
    g2exp = global_features1[None, 4:].expand(g2.x.size(0), -1)
    global_features_nodes = torch.cat((g1exp, g2exp), dim=0)
    combined_x = torch.cat((combined_x, global_features_nodes), dim=1)

    # 2 节点的有向完全图（
    global_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    global_edge_attr = torch.cat([g1.global_features, g2.global_features], dim=0)

    # 每个图节点的“节点属性”
    global_node_attr = torch.stack([
        torch.cat([x1, x2]),
        torch.cat([x2, x1]),
    ], dim=0)

    # --- 原子层分子掩码 ---
    offset1 = g1x.size(0)
    m1 = torch.zeros(combined_x.size(0), dtype=torch.bool); m1[:offset1] = True
    m2 = torch.zeros_like(m1); m2[offset1:] = True

    return MixData(
        # 原子层
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,

        # 基团层
        x_group=x_group_all,
        edge_index_group=eig_all,
        atom2group_index=atom2group_index_all,
        group_mol_id=group_mol_id,
        group_mask1=gmask1, group_mask2=gmask2,

        # 分子/混合物层
        global_edge_index=global_edge_index,
        global_edge_attr=global_edge_attr,
        global_node_attr=global_node_attr,

        # 原子层掩码
        mask1=m1, mask2=m2
    )


def load_data_two(csv_path):
    df = pd.read_csv(csv_path)

    smiles1 = [str(s).strip() if pd.notnull(s) else '' for s in df['solv1_smiles']]
    smiles2 = [str(s).strip() if pd.notnull(s) else '' for s in df['solv2_smiles']]

    y = df['y'].tolist()

    x1 = df['solv1_x'].tolist()
    x2 = df['solv2_x'].tolist()
    C  = df['C'].tolist()



    concentrations = list(zip(x1, x2, C))
    return smiles1, smiles2, y, concentrations


class MoleculesDatasetTwo(InMemoryDataset):
    def __init__(self, root, smiles1, smiles2, targets, concentrations,
                 transform=None, pre_transform=None):
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.targets = targets
        self.concentrations = concentrations
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['absorption wavelength.csv']

    @property
    def processed_file_names(self):
        return ['absorption wavelength.pt']

    def download(self):
        pass

    def process(self):
        datas = []
        for i in range(len(self.smiles1)):
            s1 = self.smiles1[i]
            s2 = self.smiles2[i]
            x1, x2, C = self.concentrations[i]
            try:
                data = combine_molecules_hg_2(s1, s2, x1, x2, C)
            except ValueError as e:
                print(f"[跳过样本#{i}] SMILES 错误: {e}")
                continue
            y = float(self.targets[i])
            y = y/1000
            data.y = torch.tensor(y, dtype=torch.float32)
            datas.append(data)

        torch.save(self.collate(datas), self.processed_paths[0])




smiles1, smiles2, targets, concentrations = load_data_two(triple_csv_path)
dataset = MoleculesDatasetTwo(root='absorption wavelength', smiles1=smiles1, smiles2=smiles2,
                              targets=targets, concentrations=concentrations)

print(len(dataset))

class FeatureCrossAttention(nn.Module):
    def __init__(self, dim_in_q, dim_in_kv, model_dim, num_heads, dropout=0):
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
    def __init__(self, atom_dim, group_dim, cond_dim, hidden=180,s2s_steps: int = 2):
        super().__init__()
        self.a2g_proj = nn.Linear(atom_dim, group_dim)
        self.g2a_proj = nn.Linear(group_dim, atom_dim)

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, group_dim+42), nn.ReLU(),
            nn.Linear(group_dim+42, group_dim+42)
        )
        self.film_beta  = nn.Sequential(
            nn.Linear(cond_dim, group_dim+42), nn.ReLU(),
            nn.Linear(group_dim+42, group_dim+42)
        )
        self.a_proj_to_g = nn.Linear(atom_dim, group_dim-80)
        self.g_proj = nn.Linear(54, group_dim-80)
        self.s2s_a2g = Set2Set(80, processing_steps=s2s_steps)
        self.merge_a2g = nn.Linear(group_dim, group_dim+42-80)

        self.group_gcn1 =GeneralConv(group_dim, group_dim,attention=True)

        self.group_gcn2 =GCNConv(group_dim+42, group_dim+42)

        self.s2s_g2a = Set2Set(group_dim, processing_steps=s2s_steps)
        self.g_proj_to_a = nn.Linear( group_dim, atom_dim)

    def forward(self, x_atom, atom_idx, x_group, group_idx, edge_index_group, cond_atom,edge_attr_group=None):
        device = x_atom.device

        Na, Ha = x_atom.size(0), x_atom.size(1)

        Gm, Dg = x_group.size(0), x_group.size(1)
        x_group = x_group[:, 0:54]
        X_group = x_group
        if Gm == 0 or atom_idx.numel() == 0 or group_idx.numel() == 0:
            xg_empty = x_atom.new_zeros((0, Dg))
            return x_atom, xg_empty
        x_group = self.g_proj(x_group)
        xa_proj = self.a_proj_to_g(x_atom)  # [Na, Dg]
        xa_items = xa_proj.index_select(0, atom_idx)  # [N_inc, Dg]
        xg_a2g = set2set_pool(xa_items, group_idx, size=Gm, s2s=self.s2s_a2g)


        xg = self.merge_a2g(xg_a2g)
        xg = torch.cat((x_group,xg),dim=1)


        cond_g = atoms_to_groups_local(cond_atom, atom_idx, group_idx, Gm, reduce='mean')
        '''cif Gm > 0:
            gamma = self.film_gamma(cond_g)                        # [Gm, Dg]
            beta  = self.film_beta(cond_g)                         # [Gm, Dg]
            xg    = gamma * xg  + beta   # [Gm, Dg]
        else:
            xg    = xg_from_atom  # [0, Dg] 安全路径'''
        '''if Gm > 0 and (edge_index_group is not None) and (edge_index_group.numel() > 0):
            xg = self.group_gcn2(xg, edge_index_group)'''

        xg = torch.cat((xg, cond_g), dim=1)


        '''xa_g2a = groups_to_atoms_local(xg, group_idx, atom_idx, Na, reduce='mean')  # [Na, 2*Dg]
        xa_from_group = self.g_proj_to_a(xa_g2a)'''

        xa_out = x_atom
        return xa_out, xg,X_group




from torch_geometric.nn import global_mean_pool

#@torch.no_grad()


def _groups_batch_from_a2g_local(xg_local: torch.Tensor,
                                 a2g_local: torch.Tensor,
                                 batch_sub: torch.Tensor) -> torch.Tensor:
    """
    返回: group_batch_self [Gm]，取值范围 0..B_sub-1（与 batch_sub 的样本数对齐）。
    若某些基团未出现在 a2g_local 中，则分配到 batch_sub 的众数。
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

    def forward(self, xg_list, gb_list):
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
        tokens_all, token_b, token_bi = [], [], []  # 每个 token 对应 mixture b，与 (b,i)
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

            # 拼接 token 特征
            t = torch.cat([xg_i, me], dim=1)  # [Gi, H + mol_emb_dim]
            tokens_all.append(t)
            token_b.append(gb_i)
            token_bi.append(gb_i * K + i)

        # === 空输入处理 ===
        if len(tokens_all) == 0:
            per_mol_out = [torch.zeros(B_sub, self.group_dim, device=device) for _ in range(K)]
            mix_feat = torch.zeros(B_sub, 2 * self.in_dim, device=device) if self.use_set2set else None
            return per_mol_out, mix_feat

        feats   = torch.cat(tokens_all, dim=0)         # [N_tok, H_in]
        b_idx   = torch.cat(token_b,   dim=0).long()   # [N_tok]  mixture id
        bi_idx  = torch.cat(token_bi,  dim=0).long()   # [N_tok]  global (b,i) id

        # ==== 2) 构造按 mixture 分组的“批内序列” ====
        # 把 token 按 b 排序 -> 能按 b 一刀切地切分
        sort_order = torch.argsort(b_idx)              # [N_tok]
        feats_sorted  = feats.index_select(0, sort_order)
        b_sorted      = b_idx.index_select(0, sort_order)
        bi_sorted     = bi_idx.index_select(0, sort_order)

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


        attn_out, attn_weights = self.mha(padded, padded, padded, key_padding_mask=key_pad_mask)  # [B_sub, L_max, H_in]

        padded = self.norm2(padded + attn_out)

        attn_out, attn_weights2 = self.mha2(padded, padded, padded, key_padding_mask=key_pad_mask)
        x = self.norm3(padded + attn_out)

        # delete pad
        valid_mask = (torch.arange(L_max, device=device)[None, :] < lens[:, None])
        x_flat = x.reshape(-1, x.size(-1))[valid_mask.view(-1)]

        N = feats.size(0)
        inv = torch.empty_like(sort_order)
        inv[sort_order] = torch.arange(N, device=device)
        attn_unsorted = x_flat[inv]


        mol_id_per_token = (bi_idx % self.K)  # [N_tok]

        per_mol_out = []
        for i in range(self.K):
            mask_i = (mol_id_per_token == i)
            if mask_i.any():
                part_i = attn_unsorted[mask_i]
                b_idx_i = b_idx[mask_i]
                s2s_i = self.mol_s2s(part_i, b_idx_i)
            else:
                s2s_i = attn_unsorted.new_zeros(B_sub, 2 * self.in_dim)
            per_mol_out.append(self.readout(s2s_i))


        if self.use_set2set:
            mix_feat = self.mix_s2s(attn_unsorted, b_idx)
        else:
            mix_feat = None

        return per_mol_out, mix_feat,attn_weights, attn_weights2,lens

# =============== 融合后的 MesoNet（不改你原有主干逻辑） ===============
class MesoNet(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim,
                 d_group_in, d_group_hidden=128):
        """
        d_group_in 必须传入 data.x_group.size(1)
        """
        super(MesoNet, self).__init__()

        # ======= 你原有的层（保持） =======
        self.K = 2
        self.mol_emb_dim = 18

        # 跨分子基团交互注意力（输入用基团维 hidden_dim）
        self.cross_group_attn = CrossMolGroupInter(
            group_dim=hidden_dim+44,  # 你的基团表示维度
            K=2,  # 三个分子
            mol_emb_dim=16,
            num_heads=4,
            use_set2set=True,
            s2s_steps=2
        )
        self.attn_atom_elem   = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.attn_group_atom  = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.attn_global_group= FeatureCrossAttention(dim_in_q=54, dim_in_kv=32, model_dim=32, num_heads=4)
        self.inter            = FeatureCrossAttention(dim_in_q=55, dim_in_kv=55, model_dim=32, num_heads=4)

        edge_hidden_dim = 32
        self.a11 = NNConv(41, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 41 * 32)
        ), aggr="mean")

        self.G = nn.Linear(21, 32)
        self.NCP1 = CfC(32, AutoNCP(64, 32), batch_first=True)
        self.x22 = nn.Linear(96,96)
        self.x2 = nn.Linear(6,32)
        self.a21 = NNConv(96, 96, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 96 * 96)
        ), aggr="mean")
        self.trans = nn.Linear(96,96)

        self.relu = nn.ReLU()
        self.xm = nn.Linear(hidden_dim, hidden_dim)

        self.subgraph_conv1 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')
        self.subgraph_conv2 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')

        self.global_conv = NNConv(hidden_dim*2+2, hidden_dim, nn.Sequential(
            nn.Linear(4, edge_hidden_dim), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim , 322*hidden_dim)
        ), aggr='mean')
        self.global_conv2 = NNConv(36,36, nn.Sequential(
            nn.Linear(4, 4), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(4 , 36*36)
        ), aggr='mean')

        self.set2set  = Set2Set(hidden_dim, processing_steps=2)
        self.set2set2 = Set2Set(3*hidden_dim+2 , processing_steps=2)
        self.setgroup = Set2Set(238, processing_steps=2)

        self.group = nn.Linear(175,175)
        self.g = nn.Linear(21, 32)
        self.fc = nn.Sequential(
            nn.Linear(1238, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 1)
        )

        self.sub3 = AttentiveFP(in_channels=hidden_dim, hidden_channels= hidden_dim,out_channels=hidden_dim*2, edge_dim=edge_dim, num_layers=2 , num_timesteps=2,dropout=0.0)

        # FiLM 参数
        self.c1_gamma = nn.Sequential(nn.Linear(110, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c1_beta  = nn.Sequential(nn.Linear(110, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c2_gamma = nn.Sequential(nn.Linear(110, 160), nn.ReLU(), nn.Linear(160, 160))
        self.c2_beta  = nn.Sequential(nn.Linear(110, 160), nn.ReLU(), nn.Linear(160, 160))

        self.c3_gamma = nn.Sequential(nn.Linear(2, 320), nn.ReLU(), nn.Linear(320, 320))
        self.c3_beta  = nn.Sequential(nn.Linear(2, 320), nn.ReLU(), nn.Linear(320, 320))

        self.c4_gamma = nn.Sequential(nn.Linear(2, 236), nn.ReLU(), nn.Linear(236, 236))
        self.c4_beta  = nn.Sequential(nn.Linear(2, 236), nn.ReLU(), nn.Linear(236, 236))
        self.group2group = nn.Linear(hidden_dim,32)
        # ======= 新增：Atom<->Group 桥（无边特征） =======
        self.atom_group_bridge = AtomGroupBridgeFiLM(
            atom_dim=hidden_dim, group_dim=hidden_dim,cond_dim = 2, s2s_steps=2
        )
        #self.group_pooler = GroupPoolSet2Set(in_dim=hidden_dim+20, processing_steps=2)

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
        group_view = self._slice_group_view(data, mol_id, mask)

        xg_local  = group_view["xg_local"]
        a2g_local = group_view["a2g_local"]
        eig_local = group_view["eig_local"]
        x1 = subgraph_x[:, 0:41]
        atom_onehot = subgraph_x[:, :54].detach().cpu().numpy()
        atom_types = atom_onehot.argmax(axis=1)


        x2 = subgraph_x[:, 41:47]
        x3 = subgraph_x[:, 47:47+54]

        g  = subgraph_x[:, 47+54+4:47+54+4+55]
        G = subgraph_x[:, 47+54+4+55:]

        C_raw = torch.cat((g[:, 54:55],G[:, 54:55]), dim=1)
        C = C_raw
        x1 = self.relu(self.a11(x1, subgraph_edge_index, subgraph_edge_attr))

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
        group_updated, _ = self.attn_group_atom(
            x1.unsqueeze(1),
            global_updated.unsqueeze(1)
        )
        group_updated = group_updated.squeeze(1)
        #Environment → intrinsic physical features of atoms

        '''atom_updated, _  = self.attn_atom_elem(
            x2.unsqueeze(1),
            global_updated.unsqueeze(1)
        )
        atom_updated = atom_updated.squeeze(1)'''
        #Elemental Feature Extraction Module(NCP)

        x2 = self.x2(x2)
        x2 = self.relu(x2)
        x2_input = x2.unsqueeze(1)
        predicted_steps = []
        hidden_state =torch.cat((global_updated,group_updated),dim=1)

        for _ in range(3):
            output, hidden_state = self.NCP1(x2_input,hidden_state)
            x2_input = output
            predicted_steps.append(output.view(output.size(0), -1))

        x2_output = torch.cat(predicted_steps, dim=-1)
        x2_output = self.x22(x2_output)
        x2_output = self.relu(x2_output)
        x2_output = self.a21(x2_output, subgraph_edge_index, subgraph_edge_attr)
        x2 = self.relu(x2_output)

        # Cross-Attention

        #Group assignment → atomic environment



        x = torch.cat((x1,x2,global_updated),dim =1)
        #xx =self.relu(self.trans(torch.cat(( global_updated,group_updated,atom_updated),dim =1)))
        #x = torch.cat((x,xx),dim=1)
        xm = self.relu(self.xm(x))
        '''gamma1 = self.c1_gamma(torch.cat((g,G),dim=1)); beta1 = self.c1_beta(torch.cat((g,G),dim=1))
        xm = gamma1 * xm + beta1'''


        # ======= 插入：Atom<->Group 超图交互（无边特征）=======

        edge_attr_group =None

        # ======= 回到你的原子消息传递 + NCP =======
        xm_film, xg_after,xg_type = self.atom_group_bridge(
            x_atom=xm,
            atom_idx=a2g_local[1],
            x_group=xg_local,
            group_idx=a2g_local[0],
            edge_index_group=eig_local,
            cond_atom=C_raw,
            edge_attr_group=None
            # <--- 新增
        )

        subgraph_x = self.subgraph_conv1(xm, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)




        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)


        subgraph_x = self.sub3(subgraph_x, subgraph_edge_index, subgraph_edge_attr,batch[mask])

        #subgraph_x = self.set2set(subgraph_x3, batch[mask])
        group = global_mean_pool(inter, batch[mask])

        group_batch = _groups_batch_from_a2g_local(xg_after, a2g_local, batch[mask])  # [Gm]
        atom_types_sub = data.x[mask][:, :16].argmax(dim=1).cpu().numpy()
        s1 = subgraph_x
        x2_outputs = group
        C_values = C[:, 0].detach().cpu().numpy()

        return subgraph_x, xg_after, group_batch, group,atom_types_sub,s1,C_values, xg_type

    def forward(self, data):
        device = data.x.device
        K = 2
        global_edge_attrall = data.global_edge_attr.to(device)
        global_node_attr = data.global_node_attr.to(device)
        global_edge_attr = global_edge_attrall[:, 0:4]

        # 四个分子分别跑
        s1, xg_after1, group_batch1, grp1,_,_,_,xg_type1 = self.process_subgraph(data, data.mask1, mol_id=0)
        s2, xg_after2, group_batch2, grp2,_,_ ,_,xg_type2= self.process_subgraph(data, data.mask2, mol_id=1)

        xg_list = [xg_after1, xg_after2]  # [Gi, H]
        gb_list = [group_batch1, group_batch2]  # [Gi]

        per_mol_cross,mix_feat, attn_w1, attn_w2,lens= self.cross_group_attn(xg_list, gb_list)

        H = per_mol_cross[0].size(1)
        B_sub = per_mol_cross[0].size(0) if per_mol_cross[0].numel() > 0 else 1
        cross_stack = torch.empty((B_sub * self.K, H), device=device)
        cross_stack[0::self.K] = per_mol_cross[0]
        cross_stack[1::self.K] = per_mol_cross[1]



        batch_size = s1.size(0); feat_dim = s1.size(1)
        s1 = self.relu(s1); s2 = self.relu(s2)

        expanded_x = torch.empty((batch_size * K, feat_dim), dtype=s1.dtype, device=device)
        expanded_x[0::K] = s1; expanded_x[1::K] = s2




        expanded_x = torch.cat((expanded_x, global_node_attr), dim=1)


        group = torch.empty((batch_size * K, 32), dtype=s1.dtype, device=device)
        group[0::K] = grp1; group[1::K] = grp2
        group = torch.cat((group,cross_stack), dim=1)



        group = torch.cat((group,global_node_attr), dim=1)


        # 全连接 4-节点图（每个样本内部构图，再 batch 偏移）
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

        group_out = torch.cat((expanded_x[0::K], expand_group), dim=1)

        #final_x = torch.cat((group_out), dim=1)

        output = self.fc(group_out)
        return output, (attn_w1, attn_w2, lens, xg_type1, xg_type2) # 第二返回保持你的接口
def idx_to_tag(idx):
    if isinstance(idx, (int, np.integer)) and 0 <= idx < len(meta_df):
        return meta_df.iloc[int(idx)]['Tag']
    return None

for i in range(len(dataset)):
    if not hasattr(dataset[i], 'idx'):
        dataset[i].idx = i

def collect_errors(model, loader, device, threshold, scale=1000.0, split_name='', fold_idx=0):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch)           # [num_graphs, 1]（按你当前实现）
            out = out.view(-1, 1)
            tgt = batch.y.unsqueeze(1).to(device)

            # 掩码 + 同步裁剪 idx
            mask = torch.abs(tgt) < threshold
            out = out[mask]
            tgt = tgt[mask]

            # 取回该 batch 中图级别的原始索引（我们前面加了 data.idx）
            if hasattr(batch, 'idx'):
                idx_tensor = batch.idx.view(-1, 1)
                idx_tensor = idx_tensor[mask]
                idx_np = idx_tensor.long().view(-1).cpu().numpy()
            else:
                # 理论上不会走到这里
                idx_np = np.array([-1] * out.shape[0])

            # 放大到你的千分位制
            y_true = (tgt * scale).cpu().numpy().flatten()
            y_pred = (out * scale).cpu().numpy().flatten()
            abs_err = np.abs(y_pred - y_true)

            for i_idx, yt, yp, ae in zip(idx_np, y_true, y_pred, abs_err):
                rows.append({
                    'fold': fold_idx + 1,
                    'split': split_name,      # 'val' 或 'test'
                    'idx': int(i_idx),
                    'y_true': float(yt),
                    'y_pred': float(yp),
                    'abs_err': float(ae),
                })
    return rows
# 用法：
# visualize_dispatch(dataset, smiles_list, sample_idx=0, mol_id=0, ax_img=axes[0,0], ax_bip=axes[0,1])
# visualize_dispatch(batch,   smiles_list, sample_idx=0, mol_id=1, ax_img=axes[1,0], ax_bip=axes[1,1])
from sklearn.model_selection import KFold,StratifiedKFold
import torch
from torch_geometric.loader import DataLoader  # Change to PyG DataLoader
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import KFold, train_test_split
epochs = 500
k_folds = 5
batch_size =128
input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 160
output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 400
dataset_size = len(dataset)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
start_fold = 2
best_val_losses, best_val_maes, best_val_mses, best_val_r2s = [], [], [], []
test_rmse_list, test_mae_list, test_mse_list, test_r2_list = [], [], [], []

for fold, (train_idx, valtest_idx) in enumerate(kf.split(dataset)):
    if fold < start_fold:
        print(f"Skipping Fold {fold+1}")
        continue

    print(f"Start Fold {fold+1}/{k_folds}")

    # 验证集 / 测试集 0.5:0.5 划分
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=0.5, random_state=0, shuffle=True
    )

    train_subset = [dataset[i] for i in train_idx]
    val_subset = [dataset[i] for i in val_idx]
    test_subset = [dataset[i] for i in test_idx]
    print(f"Fold {fold+1} 数据划分情况：")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")
    print(f"  总数:  {len(train_idx) + len(val_idx) + len(test_idx)}\n")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

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
            output,_ = model(batch)
            output = output.view(-1, 1)
            target = batch.y.unsqueeze(1).to(device)
            mask = torch.abs(target) < threshold

            output = output[mask]
            target = target[mask]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            target= target * 1000
            output = output * 1000
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
                output,_ = model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)
                mask = torch.abs(target) < threshold

                output = output[mask]
                target = target[mask]
                target = target * 1000
                output = output * 1000
                y_val_true.extend(target.cpu().numpy().flatten())
                y_val_pred.extend(output.cpu().numpy().flatten())

        val_mse = mean_squared_error(y_val_true, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)

        # 记录最优验证结果
        y_test_true, y_test_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output,_ = model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)
                mask = torch.abs(target) < threshold

                output = output[mask]
                target = target[mask]
                target = target * 1000
                output = output * 1000
                y_test_true.extend(target.cpu().numpy().flatten())
                y_test_pred.extend(output.cpu().numpy().flatten())

        test_mse = mean_squared_error(y_test_true, y_test_pred)
        test_rmse = math.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)

        # ---- 更新最佳模型 ----
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            bsettest_mae, bsettest_rmse, bsettest_r2 = test_mae, test_rmse, test_r2

        # ---- 打印 ----
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"  Val   RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        #print(f"  Test  RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")




    test_rmse_list.append(bsettest_rmse)
    test_mae_list.append(bsettest_mae)
    test_r2_list.append(bsettest_r2)

    print(f"\nFold {fold+1} Best Epoch {best_epoch}")
    print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {bsettest_rmse:.4f}, Test MAE: {bsettest_mae:.4f}, Test R²: {bsettest_r2:.4f}")

# ---- 最终平均结果 ----
print("\nAverage Results Across Folds:")
print(f"  Avg Test RMSE: {np.mean(test_rmse_list):.4f}, Avg Test MAE: {np.mean(test_mae_list):.4f}, Avg Test R²: {np.mean(test_r2_list):.4f}")

