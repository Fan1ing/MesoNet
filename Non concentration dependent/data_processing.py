from torch_geometric.data import Data
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen
import numpy as np
from collections import defaultdict
from torch_geometric.utils import coalesce
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


from pathlib import Path

triple_csv_path = 'data/absorption wavelength.csv'

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

    mol = Chem.AddHs(mol)
    num_donors    = rdMolDescriptors.CalcNumHBD(mol)
    num_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    logp          = Crippen.MolLogP(mol)
    tpsa          = rdMolDescriptors.CalcTPSA(mol)

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_featurizer.encode(atom))
    node_features = torch.tensor(node_features, dtype=torch.float32)  # [Na, Da0]

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

    edges, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = bond_featurizer.encode(bond)
        edges.append([i, j]); edge_features.append(e)
        edges.append([j, i]); edge_features.append(e)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_features, dtype=torch.float32)

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

    if len(group_nodes) > 0:
        g_from_atoms = []
        for _, members in group_nodes:
            g_from_atoms.append(node_features[members, :].mean(dim=0, keepdim=True))
        g_from_atoms = torch.cat(g_from_atoms, dim=0)      # [Gm, D_atom_ext]
        x_group = torch.cat([group_type_oh, g_from_atoms], dim=1)  # [Gm, n_types + D_atom_ext]
    else:
        x_group = torch.empty(0, len(group_names) + node_features.size(1))

    gi, ai = [], []
    for gid, (_n, members) in enumerate(group_nodes):
        for a in members:
            gi.append(gid); ai.append(a)
    atom2group_index = torch.tensor([gi, ai], dtype=torch.long) if gi else torch.empty(2,0, dtype=torch.long)


    atom2groups = defaultdict(list)
    for gid, (_n, members) in enumerate(group_nodes):
        for a in members:
            atom2groups[a].append(gid)
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

    global_features_repeated = global_features.repeat(N, 1)
    node_features = torch.cat([node_features, global_features_repeated], dim=1)

    return Data(
        x=node_features, edge_index=edge_index, edge_attr=edge_attr,
        x_group=x_group, edge_index_group=edge_index_group,
        atom2group_index=atom2group_index,
        global_features=global_features
    )
def combine_molecules_hg_2(smiles1, smiles2, x1=None, x2=None,C=None):
    g1 = process_molecule_hg(smiles1)
    g2 = process_molecule_hg(smiles2)

    x1 = torch.tensor([x1], dtype=torch.float32)
    x2 = torch.tensor([x2], dtype=torch.float32)


    def add_conc(x_atom, c):
        return torch.cat([x_atom, c.expand(x_atom.size(0), -1)], dim=1)

    g1x, g2x = add_conc(g1.x, x1), add_conc(g2.x, x2)

    off_a1 = 0
    off_a2 = off_a1 + g1x.size(0)
    combined_x = torch.cat([g1x, g2x], dim=0)
    combined_edge_index = torch.cat([
        g1.edge_index + off_a1,
        g2.edge_index + off_a2,
    ], dim=1)
    combined_edge_attr = torch.cat([g1.edge_attr, g2.edge_attr], dim=0)

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

    global_node_attr = torch.stack([
        torch.cat([x1, x2]),
        torch.cat([x2, x1]),
    ], dim=0)

    offset1 = g1x.size(0)
    m1 = torch.zeros(combined_x.size(0), dtype=torch.bool); m1[:offset1] = True
    m2 = torch.zeros_like(m1); m2[offset1:] = True

    return MixData(
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,

        x_group=x_group_all,
        edge_index_group=eig_all,
        atom2group_index=atom2group_index_all,
        group_mol_id=group_mol_id,
        group_mask1=gmask1, group_mask2=gmask2,

        global_edge_index=global_edge_index,
        global_edge_attr=global_edge_attr,
        global_node_attr=global_node_attr,

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
                print(f"[jump#{i}] SMILES : {e}")
                continue
            y = float(self.targets[i])
            y = y/1000
            data.y = torch.tensor(y, dtype=torch.float32)
            datas.append(data)

        torch.save(self.collate(datas), self.processed_paths[0])




smiles1, smiles2, targets, concentrations = load_data_two(triple_csv_path)
dataset = MoleculesDatasetTwo(root='absorption wavelength', smiles1=smiles1, smiles2=smiles2,
                              targets=targets, concentrations=concentrations)

