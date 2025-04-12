import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen
from torch_geometric.utils import subgraph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ncps.torch import CfC,LTC
from ncps.wirings import AutoNCP
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
triple_csv_path = '/MesoNet/data/Activity coefficient (three-component).csv'

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

        # Te
        elif node_features[i, 15] == 1:
            node_features[i, -1] = 2.1
            node_features[i, -2] = 135
            node_features[i, -3] = 52
            node_features[i, -4] = 127.6
            node_features[i, -5] = 9.010
            node_features[i, -6] = 1.971

    node_features = torch.tensor(node_features, dtype=torch.float32)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_features.append(bond_featurizer.encode(bond))
        edge_features.append(bond_featurizer.encode(bond))

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    functional_groups_smarts = {
        "hydroxyl": "[OX2H]",
        "carboxyl": "C(=O)O",
        "amine": "[NX3;H2,H1;!$(NC=O)]",
        "ester": "C(=O)O",
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
        "benzene": "c1ccccc1",
        "thiol": "[C-SH]",
        "alkyne": "C#C", 
        "nitro": "N(=O)=O",
        "ether": "C-O-C",
        "alkene": "C=C",
        "Na+_ion": "[Na+1]",       
        "K+_ion": "[K+1]",      
    }

    functional_groups_count = {key: 0 for key in functional_groups_smarts.keys()}

    for name, smarts in functional_groups_smarts.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")
        matches = mol.GetSubstructMatches(patt)
        if matches:
            functional_groups_count[name] = len(matches)

    # Adds the number of functional groups to the feature vector
    global_features2 = torch.tensor(list(functional_groups_count.values()), dtype=torch.float32).unsqueeze(0)

    # Combine all features (such as num_donors, num_acceptors, logp, tpsa) into a vector
    global_features1 = torch.tensor([num_donors, num_acceptors, logp, tpsa], dtype=torch.float32).unsqueeze(0)

    # Add global_features2 to global_features
    global_features = torch.cat((global_features1, global_features2), dim=1)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)



def combine_molecules(smiles1, smiles2, smiles3=None, x1=None, x2=None, x3=None):

    graph1 = process_molecule(smiles1)
    graph2 = process_molecule(smiles2)

    if smiles3:
        graph3 = process_molecule(smiles3)
    else:
        graph3 = None

    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index + graph1.x.size(0)], dim=1)
    combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)

    if graph3:
        combined_x = torch.cat([combined_x, graph3.x], dim=0)
        combined_edge_index = torch.cat([combined_edge_index, graph3.edge_index + graph1.x.size(0) + graph2.x.size(0)], dim=1)
        combined_edge_attr = torch.cat([combined_edge_attr, graph3.edge_attr], dim=0)


    global_edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2],
         [1, 0, 2, 0, 2, 1]],
        dtype=torch.long
    )
    global_edge_attr = torch.cat(
        [
            graph1.global_features, graph2.global_features,
            graph1.global_features, graph3.global_features if graph3 else graph2.global_features,
            graph2.global_features, graph3.global_features if graph3 else graph2.global_features,
        ],
        dim=0
    )
    global_node_attr = torch.tensor(
        [[x1, x2, x3],
        [x2, x1, x3],
        [x3, x1, x2]],dtype=torch.float32
    )

    # Generation mask
    mask1 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    mask1[:graph1.x.size(0)] = True

    mask2 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    mask2[graph1.x.size(0):graph1.x.size(0) + graph2.x.size(0)] = True

    mask3 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    if graph3:
        mask3[graph1.x.size(0) + graph2.x.size(0):] = True

    return Data(
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        global_edge_index=global_edge_index,
        global_edge_attr=global_edge_attr,
        global_node_attr=global_node_attr,
        mask1=mask1,
        mask2=mask2,
        mask3=mask3
    )



def load_data(triple_csv_path):
    triple_df = pd.read_csv(triple_csv_path)

    smiles1_triple = triple_df['solv1_smiles'].tolist()
    smiles2_triple = triple_df['solv2_smiles'].tolist()
    smiles3_triple = triple_df['solv3_smiles'].tolist()
    solv1_gamma_triple = triple_df['solv1_gamma'].tolist()
    solv2_gamma_triple = triple_df['solv2_gamma'].tolist()
    solv3_gamma_triple = triple_df['solv3_gamma'].tolist()

    solv1_x_triple = triple_df['solv1_x'].tolist()
    solv2_x_triple = triple_df['solv2_x'].tolist()
    solv3_x_triple = triple_df['solv3_x'].tolist()

    smiles1 = smiles1_triple
    smiles2 = smiles2_triple
    smiles3 = smiles3_triple

    targets_triple = list(zip(solv1_gamma_triple, solv2_gamma_triple, solv3_gamma_triple))

    concentrations = list(zip(solv1_x_triple, solv2_x_triple, solv3_x_triple))

    return smiles1, smiles2, smiles3, targets_triple, concentrations


class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, smiles1, smiles2, smiles3, targets, concentrations, transform=None, pre_transform=None):
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.smiles3 = smiles3
        self.targets = targets
        self.concentrations = concentrations
        super(MoleculesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['triple_molecule_data.csv']

    @property
    def processed_file_names(self):
        return ['datanewSS.pt']

    def download(self):
        pass

    def process(self):
        datas = []
        for i in range(len(self.smiles1)):
            smile1 = self.smiles1[i]
            smile2 = self.smiles2[i]
            smile3 = self.smiles3[i]
            target = self.targets[i]
            concentration = self.concentrations[i]

            if smile3 is not None:
                data = combine_molecules(smile1, smile2, smile3, concentration[0], concentration[1], concentration[2])
            else:
                data = combine_molecules(smile1, smile2, None, concentration[0], concentration[1])

            if data is not None:
                data.y = torch.tensor(target, dtype=torch.float32)
                datas.append(data)
            else:
                print(f"Sample {i} is invalid and has been skipped.")

        torch.save(self.collate(datas), self.processed_paths[0])


smiles1, smiles2, smiles3, targets, concentrations = load_data(triple_csv_path)

dataset = MoleculesDataset(root='datanewSS', smiles1=smiles1, smiles2=smiles2, smiles3=smiles3, targets=targets, concentrations=concentrations)
print(len(dataset))

class MesoNet(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(MesoNet, self).__init__()

        self.transformer_layer = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=2)
        self.decoder_layer = TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=2)
        edge_hidden_dim = 32
        self.a11 = NNConv(41, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 41 * 32)
        ), aggr="mean")

        self.lstm_a2_1 = CfC(6,AutoNCP(12,6), batch_first=True)
        self.x11 = nn.Linear(41,32)

        self.hidden = nn.Linear(6,12)


        self.x211 = nn.Linear(12,6)

        self.x22 = nn.Linear(30,32)
        self.a21 = NNConv(32, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 32 * 32)
        ), aggr="mean")

        self.last = nn.Linear(128,128)


        self.xm = nn.Linear(64, 64)

        self.relu = nn.ReLU()
        self.xm2 = CfC(64, 64, batch_first=True)
        self.xm3 = nn.Linear(128,128)
        self.subgraph_conv1 = NNConv(128, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(edge_hidden_dim, 128 * hidden_dim)
        ), aggr='mean')

        self.subgraph_conv2 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')
        self.global_conv = NNConv(259, 128, nn.Sequential(
            nn.Linear(4, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim , 259*128)
        ), aggr='mean')
        self.global_conv2 = NNConv(21,21, nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4 , 21*21)
        ), aggr='mean')
        self.set2set = Set2Set(hidden_dim, processing_steps=2)
        self.set2set2 = Set2Set(3*hidden_dim+3 , processing_steps=2)
        self.FF = nn.Linear(6*hidden_dim+6,768)

        self.setgroup = Set2Set(42, processing_steps=2)

        self.group = nn.Linear(84+21,128)
        self.combine = nn.Linear(259+768+128,259+768+128)
        self.fc = nn.Sequential(
            nn.Linear(259+768+128, 1024),
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


    def process_subgraph(self, x, edge_index, edge_attr, batch, mask):
        subgraph_x = x[mask]
        subgraph_edge_index, subgraph_edge_attr = subgraph(mask, edge_index, edge_attr, relabel_nodes=True)

        x1 = subgraph_x[:, 0:41]
        x1 = self.a11(x1, subgraph_edge_index, subgraph_edge_attr)
        x1 = self.relu(x1)


        x2 = subgraph_x[:, 41:]
        x2out = x2

        x2_input = x2.unsqueeze(1)
        predicted_steps = []
        hidden_state =torch.cat((x2,x2),dim=1)


        for _ in range(5):
            output, hidden_state = self.lstm_a2_1(x2_input,hidden_state)
            predicted_steps.append(output.view(output.size(0), -1))

        x2_output = torch.cat(predicted_steps, dim=-1)

        x2_output = self.x22(x2_output)
        x2_output = self.relu(x2_output)

        x2_output= self.a21(x2_output, subgraph_edge_index, subgraph_edge_attr)
        x2_output = self.relu(x2_output)
        x2_outputs = x2_output

        # Transformer Encoder & Decoder
        combined_output = torch.stack([x1, x2_output], dim=1)  # Shape: [batch_size, seq_len=2, feature_dim=32]
        combined_output = combined_output.transpose(0, 1)  # Shape: [seq_len, batch_size, feature_dim]
        encoded_memory = self.transformer_encoder(combined_output)

        # Transformer Decoder
        seq_len_tgt = 2  # Example: Set the target sequence length
        tgt = encoded_memory.mean(dim=0, keepdim=True).repeat(seq_len_tgt, 1, 1)  # Shape: [seq_len_tgt, batch_size, feature_dim]

        decoded_output = self.transformer_decoder(tgt, encoded_memory)  # Decoder output     decoded_output = self.transformer_decoder(tgt, encoded_memory)  # Decoder output
        # ReLU activation on decoded output


        transformer_output = self.relu(decoded_output.mean(dim=0))  # Shape: [batch_size, feature_dim=32]'''
        transformer_outputs =transformer_output


        # Merge processing feature


        xm = torch.cat((x1, x2_output), dim=1)
        xmm = xm
        xm = xm.unsqueeze(1)

        #xm = xm.squeeze(1)
        hidden_state = torch.cat((transformer_output,transformer_output),dim=1)
        #xm = torch.cat((transformer_output, xm), dim=1)
        #xm = xm.unsqueeze(1)
        xm, _ = self.xm2(xm,hidden_state)
        xm = xm.squeeze(1)
        xm = self.relu(xm)
        xm = torch.cat((xm,xmm),dim=1)
        xm = self.last(xm)
        xm = self.relu(xm)
        # Graph convolution processing
        subgraph_x = self.subgraph_conv1(xm, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.set2set(subgraph_x, batch[mask])
        return subgraph_x,x1,x2_outputs,transformer_outputs



    def forward(self, data):
        global_edge_attrall = data.global_edge_attr.to(data.x.device)
        global_node_attr = data.global_node_attr.to(data.x.device)
        global_edge_attr = global_edge_attrall[:, 0:4]
        group = global_edge_attrall[:, 4:]

        subgraph1_x,x2_outputs,x2out,transformer_outputs = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask1)
        subgraph2_x,_,_,_ = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask2)
        subgraph3_x,_,_,_ = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask3)
        subgraph1_x = self.relu(subgraph1_x)
        subgraph2_x = self.relu(subgraph2_x)
        subgraph3_x = self.relu(subgraph3_x)

        batch_size = subgraph1_x.size(0)

        group1 = group[0::6]
        group2 = group[1::6]
        group3 = group[3::6]
        group = torch.empty((batch_size * 3, 21), dtype=subgraph1_x.dtype, device=subgraph1_x.device)
        group[0::3] = group1
        group[1::3] = group2
        group[2::3] = group3



        new_global_edge_index = torch.empty((2, batch_size * 6), dtype=torch.long, device=subgraph1_x.device)
        for i in range(batch_size):
            new_global_edge_index[0][i * 6] = i * 2      # 0, 2, 4, ...
            new_global_edge_index[1][i * 6] = i * 2 + 1  # 1, 3, 5, ...

            new_global_edge_index[0][i * 6 + 1] = i * 2 + 1  # 1, 3, 5, ...
            new_global_edge_index[1][i * 6 + 1] = i * 2 + 2  # 2, 4, 6, ...

            new_global_edge_index[0][i * 6 + 2] = i * 2 + 2  # 2, 4, 6, ...
            new_global_edge_index[1][i * 6 + 2] = i * 2      # 0, 2, 4, ...

            new_global_edge_index[0][i * 6 + 3] = i * 2 + 1  # 1, 3, 5, ...
            new_global_edge_index[1][i * 6 + 3] = i * 2 + 2  # 2, 4, 6, ...

            new_global_edge_index[0][i * 6 + 4] = i * 2 + 2  # 2, 4, 6, ...
            new_global_edge_index[1][i * 6 + 4] = i * 2 + 1  # 1, 3, 5, ...

            new_global_edge_index[0][i * 6 + 5] = i * 2      # 0, 2, 4, ...
            new_global_edge_index[1][i * 6 + 5] = i * 2 + 2  # 2, 4, 6, ...

        global_edge_index = new_global_edge_index.view(2, batch_size * 6)
        expanded_x = torch.empty((batch_size * 3, subgraph1_x.size(1)), dtype=subgraph1_x.dtype, device=subgraph1_x.device)
        expanded_x[0::3] = subgraph1_x
        expanded_x[1::3] = subgraph2_x
        expanded_x[2::3] = subgraph3_x
        expanded_x =torch.cat((expanded_x, global_node_attr), dim=1)

        combined_x = self.global_conv(expanded_x, global_edge_index, global_edge_attr.to(expanded_x.device))  # Make sure the edge properties are on the same device
        combined_x =self.relu(combined_x)
        num_classes = batch_size
        repeats_per_class = 3

        tensor = torch.cat([torch.full((repeats_per_class,), i, dtype=torch.long) for i in range(num_classes)])
        tensor = tensor.to(combined_x.device)
        combined_x = torch.cat((combined_x,expanded_x),dim=1)
        set2set_x = self.set2set2(combined_x, tensor)
        set2set_x_shape = set2set_x.size()
        expanded_x_shape = expanded_x.size()
        expanded_set2set_x = set2set_x.unsqueeze(1).expand(-1, 3, -1)
        expanded_set2set_x = expanded_set2set_x.contiguous().view(-1, set2set_x_shape[1])

        expanded_set2set_x = self.FF(expanded_set2set_x)
        expanded_set2set_x = self.relu(expanded_set2set_x)

        expand_group = self.global_conv2(group, global_edge_index, global_edge_attr.to(expanded_x.device))
        expand_group = self.relu(expand_group)
        expand_group = self.setgroup(torch.cat((group,expand_group),dim=1), tensor)
        expand_group_shape = expand_group.size()

        expand_group = expand_group.unsqueeze(1).expand(-1, 3, -1)

        expand_group = expand_group.contiguous().view(-1, expand_group_shape[1])


        group = torch.cat((group,expand_group),dim =1)
        group = self.group(group)
        group = self.relu(group)

        final_x = torch.cat((expanded_x,expanded_set2set_x,group), dim=1)
        '''final_x = self.combine(final_x)
        final_x = self.relu(final_x)'''
        output = self.fc(final_x)

        return output


'''train_size = int(0.8 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size


train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(99)
)
'''
from sklearn.model_selection import KFold
import torch
from torch_geometric.loader import DataLoader  # Change to PyG DataLoader
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

epochs = 180
k_folds = 5
batch_size = 256
input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim

hidden_dim = 128

output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_size = len(dataset)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=2021)
best_val_losses=[]
best_val_maes=[]
best_val_mses=[]
best_val_r2s=[]



start_fold = 0
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    if fold < start_fold:
        print(f"Skipping Fold {fold+1}")
        continue

    print(f"Start Fold {fold+1}/{k_folds}")

    train_subset = [dataset[i] for i in train_idx]
    val_subset = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

    model = MesoNet(input_dim, edge_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity
    best_val_mae = float('inf')
    best_val_mse = float('inf')
    best_val_r2 = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        y_train_true = []
        y_train_pred = []

        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch)
            output = output.view(-1, 1)
            target = batch.y.unsqueeze(1).to(device)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

            y_train_true.extend(target.cpu().numpy().flatten())
            y_train_pred.extend(output.detach().cpu().numpy().flatten())

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        train_mse = mean_squared_error(y_train_true, y_train_pred)
        train_r2 = r2_score(y_train_true, y_train_pred)

        model.eval()
        val_loss = 0
        y_val_true = []
        y_val_pred = []
        absolute_errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)

                loss = criterion(output, target)
                val_loss += loss.item() * batch.num_graphs

                y_val_true.extend(target.cpu().numpy().flatten())
                y_val_pred.extend(output.cpu().numpy().flatten())
                absolute_errors.extend(np.abs(target.cpu().numpy().flatten() - output.cpu().numpy().flatten()))

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        val_mse = mean_squared_error(y_val_true, y_val_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)

        # Update the best model if current val loss is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # Update best validation loss
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")

    # Save absolute errors for this fold
    np.savetxt(f'absolute_errors_fold{fold+1}.csv', absolute_errors)

    print(f"\nBest Model Performance for Fold {fold+1}:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Best Validation MAE: {best_val_mae:.4f}")
    print(f"  Best Validation MSE: {best_val_mse:.4f}")
    print(f"  Best Validation R²: {best_val_r2:.4f}")

    # Save the best metrics for this fold
    best_val_losses.append(best_val_loss)
    best_val_maes.append(best_val_mae)
    best_val_mses.append(best_val_mse)
    best_val_r2s.append(best_val_r2)

    # Save the best model for the fold
    torch.save(best_model_state, f'best_model_fold{fold+1}.pth')

    del model
    torch.cuda.empty_cache()

# Calculate and print the average of the best results across all folds
avg_best_val_loss = np.mean(best_val_losses)
avg_best_val_mae = np.mean(best_val_maes)
avg_best_val_mse = np.mean(best_val_mses)
avg_best_val_r2 = np.mean(best_val_r2s)

print("\nAverage Best Results Across All Folds:")
print(f"  Avg Best Validation Loss: {avg_best_val_loss:.4f}")
print(f"  Avg Best Validation MAE: {avg_best_val_mae:.4f}")
print(f"  Avg Best Validation MSE: {avg_best_val_mse:.4f}")
print(f"  Avg Best Validation R²: {avg_best_val_r2:.4f}")

