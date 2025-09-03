from ncps.torch import CfC,LTC
import numpy as np
import pandas as pd
from past.translation import transform
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
import math
csv_path = '/MesoNet/data/Lipophilicity.csv'
#csv_path = '/home/ubuntu/Lipophilicity.csv'
#csv_path = "C:/Users/Ahan/Desktop/data/Lipophilicity.csv"

df = pd.read_csv(csv_path)
y1 = df['lipophilicity']
smiles = df['smiles']
ys = y1

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
        self.dim

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
        "symbol": {"B", "Br", "C", "Cl", "F","Ge", "H", "I", "N", "Na", "O", "P", "S","Se","Si","Te"},
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


# mol = Chem.MolFromSmiles('CO')
# mol = Chem.AddHs(mol)
# # for atom in mol.GetAtoms():
# #     # print(atom.GetSymbol())
# #     # print(atom_featurizer.encode(atom))
# for bond in mol.GetBonds():
#     print([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     print(bond_featurizer.encode(bond))


class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, transform = None):
        super(MoleculesDataset,self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        return 'qingzhidata.csv'

    @property
    def processed_file_names(self):
        return 'qingzhidata.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        datas = []
        for smile, y in zip(smiles,ys):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)

            embeddings = []
            for atom in mol.GetAtoms():
                embeddings.append(atom_featurizer.encode(atom))
            embeddings = torch.tensor(embeddings,dtype=torch.float32)

            #增加Nr
            rows, cols = embeddings.shape
            zeros_tensor = torch.zeros(rows, 7)
            embeddings = torch.cat((embeddings,zeros_tensor),dim = 1)
            for i in range(rows):
                #B
                if embeddings[i,0] == 1: # For Boron (B)
                    embeddings[i,-1] = 2.04 # Electronegativity
                    embeddings[i,-2] = 82  # Covalent radius
                    embeddings[i,-3] = 5   # Atomic number
                    embeddings[i,-4] = 10.82 # Atomic mass
                    embeddings[i,-5] = 8.298 # Ionization energy
                    embeddings[i,-6] = 0.277 # Electron affinity
                    embeddings[i,-7] = 0
                # Continue similarly for other elements like Br, C, Cl, etc

                #Br
                elif embeddings[i,1] == 1:
                    embeddings[i,-1] = 2.96
                    embeddings[i,-2] = 114
                    embeddings[i,-3] = 35
                    embeddings[i,-4] = 79.904
                    embeddings[i,-5] = 11.814
                    embeddings[i,-6] = 3.364
                    embeddings[i,-7] = 1

                #C
                elif embeddings[i,2] == 1:
                    embeddings[i,-1] = 2.55
                    embeddings[i,-2] = 77
                    embeddings[i,-3] = 6
                    embeddings[i,-4] = 12.011
                    embeddings[i,-5] = 11.261
                    embeddings[i,-6] = 1.595
                    embeddings[i,-7] = 0

                #Cl
                elif embeddings[i,3] == 1:
                    embeddings[i,-1] = 3.16
                    embeddings[i,-2] = 99
                    embeddings[i,-3] = 17
                    embeddings[i,-4] = 35.45
                    embeddings[i,-5] = 12.968
                    embeddings[i,-6] = 3.62
                    embeddings[i,-7] = 1
                #F
                elif embeddings[i,4] == 1:
                    embeddings[i,-1] = 3.98
                    embeddings[i,-2] = 71
                    embeddings[i,-3] = 9
                    embeddings[i,-4] = 18.998
                    embeddings[i,-5] = 17.422
                    embeddings[i,-6] = 3.40
                    embeddings[i,-7] = 1

            #Ge
                elif embeddings[i,5] == 1:
                    embeddings[i,-1] = 2.01
                    embeddings[i,-2] = 122
                    embeddings[i,-3] = 32
                    embeddings[i,-4] = 72.63
                    embeddings[i,-5] = 7.90
                    embeddings[i,-6] = 1.23
                    embeddings[i,-7] = 0


                #H
                elif embeddings[i,6] == 1:
                    embeddings[i,-1] = 2.20
                    embeddings[i,-2] = 37
                    embeddings[i,-3] = 1
                    embeddings[i,-4] = 1.008
                    embeddings[i,-5] = 13.598
                    embeddings[i,-6] = 0.755
                    embeddings[i,-7] = 0


                #I
                elif embeddings[i,7] == 1:
                    embeddings[i,-1] = 2.66
                    embeddings[i,-2] = 133
                    embeddings[i,-3] = 53
                    embeddings[i,-4] = 126.9
                    embeddings[i,-5] = 10.451
                    embeddings[i,-6] = 3.060
                    embeddings[i,-7] = 1

                 #N
                elif embeddings[i,8] == 1:
                    embeddings[i,-1] = 3.04
                    embeddings[i,-2] = 75
                    embeddings[i,-3] = 7
                    embeddings[i,-4] = 14.007
                    embeddings[i,-5] = 14.534
                    embeddings[i,-6] = 0.07
                    embeddings[i,-7] = 0

                 #Na
                elif embeddings[i,9] == 1:
                    embeddings[i,-1] = 0.93
                    embeddings[i,-2] = 154
                    embeddings[i,-3] = 11
                    embeddings[i,-4] = 22.99
                    embeddings[i,-5] = 5.139
                    embeddings[i,-6] = 0.547
                    embeddings[i,-7] = 0

                 #O
                elif embeddings[i,10] == 1:
                    embeddings[i,-1] = 3.44
                    embeddings[i,-2] = 73
                    embeddings[i,-3] = 8
                    embeddings[i,-4] = 15.999
                    embeddings[i,-5] = 13.618
                    embeddings[i,-6] = 1.46
                    embeddings[i,-7] = 0

                 #P
                elif embeddings[i,11] == 1:
                    embeddings[i,-1] = 2.19 
                    embeddings[i,-2] = 106  
                    embeddings[i,-3] = 15   
                    embeddings[i,-4] = 30.974 
                    embeddings[i,-5] = 10.487 
                    embeddings[i,-6] = 0.75 
                    embeddings[i,-7] = 0

                #S
                elif embeddings[i,12] == 1:
                    embeddings[i,-1] = 2.58 
                    embeddings[i,-2] = 102
                    embeddings[i,-3] = 16  
                    embeddings[i,-4] = 32.06 
                    embeddings[i,-5] = 10.36
                    embeddings[i,-6] = 2.07
                    embeddings[i,-7] = 0

                #Se
                elif embeddings[i,13] == 1:
                    embeddings[i,-1] = 2.55 
                    embeddings[i,-2] = 116  
                    embeddings[i,-3] = 34   
                    embeddings[i,-4] = 78.971 
                    embeddings[i,-5] = 9.753 
                    embeddings[i,-6] = 2.02 
                    embeddings[i,-7] = 0
                #Si
                elif embeddings[i,14] == 1:
                    embeddings[i,-1] = 1.90 
                    embeddings[i,-2] = 111 
                    embeddings[i,-3] = 14   
                    embeddings[i,-4] = 28.085 
                    embeddings[i,-5] = 8.151 
                    embeddings[i,-6] = 1.385
                    embeddings[i,-7] = 0


                #Te
                elif embeddings[i,15] == 1:
                    embeddings[i,-1] = 2.1
                    embeddings[i,-2] = 135
                    embeddings[i,-3] = 52
                    embeddings[i,-4] = 127.6
                    embeddings[i,-5] = 9.010
                    embeddings[i,-6] = 1.971
                    embeddings[i,-7] = 0

            embeddings = torch.tensor(embeddings,dtype=torch.float32)



            edges = []
            edge_attr = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

                edge_attr.append(bond_featurizer.encode(bond))
                edge_attr.append(bond_featurizer.encode(bond))

            edges = torch.tensor(edges).T
            edge_attr = torch.tensor(edge_attr,dtype=torch.float32)

            y = torch.tensor(y,dtype=torch.float32)
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
            }



            functional_groups_patterns = {
                name: Chem.MolFromSmarts(smarts)
                for name, smarts in functional_groups_smarts.items()
            }
            group_names = list(functional_groups_patterns.keys())
            num_groups = len(group_names)


            N = embeddings.size(0)
            group_membership = torch.zeros((N, num_groups), dtype=torch.float32)

            for g_idx, name in enumerate(group_names):
                patt = functional_groups_patterns[name]
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    for atom_idx in match:
                        group_membership[atom_idx, g_idx] = 1.0

            embeddings = torch.cat((embeddings,group_membership), dim=1)


            functional_groups_count = {key: 0 for key in functional_groups_smarts.keys()}

            for name, smarts in functional_groups_smarts.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    raise ValueError(f"Invalid SMARTS mode: {smarts}")
                matches = mol.GetSubstructMatches(patt)
                if matches:
                    functional_groups_count[name] = len(matches)

            #  Add functional group counts to the feature vector
            global_features = torch.tensor(list(functional_groups_count.values()), dtype=torch.float32).unsqueeze(0)
            num_nodes = embeddings.size(0)

            # 重复 global_features N 次得到 [N, F]
            global_features_repeated = global_features.repeat(num_nodes, 1)

            # 拼接到 embeddings 上，变成新的节点特征 [N, D+F]
            embeddings = torch.cat([embeddings, global_features_repeated], dim=1)

            data = Data(x=embeddings, y=y, edge_index=edges, edge_attr=edge_attr,global_features=global_features)
            datas.append(data)

        # self.data, self.slices = self.collate(datas)
        torch.save(self.collate(datas), self.processed_paths[0])

max_nodes = 128
dataset = MoleculesDataset(root= "qingzhidata")
import torch.nn.functional as F


num_heads = 8
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
        self.x1 = nn.Linear(42, 32)
        self.x2 = nn.Linear(30, 32)

        self.x3 = nn.Linear(19, 32)
        self.x12 = nn.Linear(64, 64)
        self.x13 = nn.Linear(64, 64)
        self.a11 = NNConv(41, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 41 * 32)
        ), aggr="mean")
        self.xm2 = CfC(96, 96, batch_first=True)

        self.transformer_layer1 = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=32, dropout=0.2)
        self.transformer_encoder1 = TransformerEncoder(self.transformer_layer1, num_layers=2)

        self.transformer_layer2 = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=32, dropout=0.2)
        self.transformer_encoder2 = TransformerEncoder(self.transformer_layer2, num_layers=2)

        self.transformer_layer3 = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=32, dropout=0.2)
        self.transformer_encoder3 = TransformerEncoder(self.transformer_layer3, num_layers=2)

        self.transformer_layer4 = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=32, dropout=0.2)
        self.transformer_encoder4 = TransformerEncoder(self.transformer_layer4, num_layers=2)

        self.decoder_layer = TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=2)
        self.lstm_a2_1 = CfC(6, AutoNCP(12, 6), batch_first=True)
        self.x22 = nn.Linear(6, 32)  # 5-step * feature 6 -> 32
        self.a21 = NNConv(32, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 32 * 32)
        ), aggr="mean")
        self.xm2 = CfC(96, 96, batch_first=True)

        self.g_conv = NNConv(19, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 19 * 32)
        ), aggr="mean")
        self.g_conv2 = NNConv(32, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0),
            nn.Linear(edge_hidden_dim, 32 * 32)
        ), aggr="mean")
        self.trans =nn.Linear(32*3,32*3)
        self.group_proj = nn.Linear(19, 128)
        self.g = nn.Linear(19, 32)
        self.group = nn.Linear(32, 128)
        self.attn_atom_elem = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.attn_group_atom = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.attn_global_group = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=4)
        self.xm = nn.Linear(hidden_dim, hidden_dim)
        self.attn_atom_elem = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=1)
        self.attn_group_atom = FeatureCrossAttention(dim_in_q=32, dim_in_kv=32, model_dim=32, num_heads=1)
        self.attn_global_group = FeatureCrossAttention(dim_in_q=19, dim_in_kv=32, model_dim=32, num_heads=1)
        self.inter = FeatureCrossAttention(dim_in_q=19, dim_in_kv=19, model_dim=32, num_heads=1)
        self.xm3 = nn.Linear(288, 288)  # fuse 3 parts
        self.sub1 = NNConv(32 , hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(edge_hidden_dim, 32 * hidden_dim)
        ), aggr='mean')
        self.sub2 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')
        self.set2set = Set2Set(160, processing_steps=2)
        self.sub3 = AttentiveFP(in_channels=hidden_dim, hidden_channels= hidden_dim,out_channels=hidden_dim*2, edge_dim=edge_dim, num_layers=2 , num_timesteps=2,dropout=0.0)
        self.FP = AttentiveFP(in_channels=41, hidden_channels= 128,out_channels=128, edge_dim=edge_dim, num_layers=2 , num_timesteps=2,dropout=0.0)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim*2+128, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch, global_g = (
            data.x, data.edge_index, data.edge_attr, data.batch, data.global_features
        )

        #hi-b
        x1 = x[:, :41]

        #hi-a
        x2 = x[:, 42:48]

        #hi-g
        x3 = x[:, 48:48+19]
        #Hgi

        g = x[:, 48+19:]

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
        x2_output = self.a21(x2_output, edge_index, edge_attr)
        x2 = self.relu(x2_output)

        x1 = self.relu(self.a11(x1, edge_index, edge_attr))
        x = torch.cat((x1,x2),dim =1)


        # Cross-Attention

        inter, _ = self.inter(
            g.unsqueeze(1),
            g.unsqueeze(1)
        )
        inter = inter.squeeze(1)
        #Self attention


        global_updated, _ = self.attn_global_group(
            x3.unsqueeze(1),
            inter.unsqueeze(1)
        )
        global_updated = global_updated.squeeze(1)
        #Global group → atomic group assignment


        group_updated, _ = self.attn_group_atom(
            x1.unsqueeze(1),
            global_updated.unsqueeze(1)
        )
        group_updated = group_updated.squeeze(1)
        #Group assignment → atomic environment


        atom_updated, _  = self.attn_atom_elem(
            x2.unsqueeze(1),
            group_updated.unsqueeze(1)
        )
        atom_updated = atom_updated.squeeze(1)
        #Environment → intrinsic physical features of atoms


        xx = self.relu(self.trans(torch.cat((global_updated, group_updated, atom_updated), dim=1)))
        x = torch.cat((x, xx), dim=1)

        x = self.relu(self.xm(x))

        # Intramolecular message transmission

        x = self.relu(self.sub1(x, edge_index, edge_attr))
        x = self.relu(self.sub2(x, edge_index, edge_attr))
        x = self.sub3(x, edge_index, edge_attr,batch)
        group = global_mean_pool(inter, batch)
        group = self.relu(self.group(group))
        x = torch.cat((x,group), dim=1)

        y = self.fc1(x)
        return y



input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 160
edge_hidden_dim = 32
output_dim = 1
batch_size=128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MesoNet(input_dim, edge_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(297)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=10)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)




epochs = 300
early_stopping_counter = 0
patience = 10
mae = []
r = []

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    model.train()
    train_loss = 0
    y_tr_true, y_tr_pred = [], []
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out   = model(batch).view(-1,1)
        tgt   = batch.y.unsqueeze(1).to(device)
        loss  = criterion(out, tgt)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs
        y_tr_true.extend(tgt.cpu().numpy().flatten())
        y_tr_pred.extend(out.detach().cpu().numpy().flatten())

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_mae  = mean_absolute_error(y_tr_true, y_tr_pred)
    train_r2   = r2_score(y_tr_true, y_tr_pred)

    model.eval()
    val_loss = 0
    y_val_true, y_val_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out   = model(batch).view(-1,1)
            tgt   = batch.y.unsqueeze(1).to(device)
            loss  = criterion(out, tgt)
            val_loss += loss.item() * batch.num_graphs
            y_val_true.extend(tgt.cpu().numpy().flatten())
            y_val_pred.extend(out.cpu().numpy().flatten())

    avg_val_mse = val_loss / len(val_loader.dataset)
    val_rmse    = np.sqrt(avg_val_mse)
    val_mae     = mean_absolute_error(y_val_true, y_val_pred)
    val_r2      = r2_score(y_val_true, y_val_pred)



    test_loss = 0
    y_test_true, y_test_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch).view(-1,1)
            tgt   = batch.y.unsqueeze(1).to(device)
            loss  = criterion(out, tgt)
            test_loss += loss.item() * batch.num_graphs
            y_test_true.extend(tgt.cpu().numpy().flatten())
            y_test_pred.extend(out.cpu().numpy().flatten())

    avg_test_mse = test_loss / len(test_loader.dataset)
    test_rmse    = np.sqrt(avg_test_mse)
    test_mae     = mean_absolute_error(y_test_true, y_test_pred)
    test_r2      = r2_score(y_test_true, y_test_pred)

    if val_rmse < best_val_rmse:
        best_val_rmse    = val_rmse
        test_maelast      = test_mae

        best_model_state = model.state_dict()


    print(f"Epoch {epoch}/{epochs}")
    print(f"  Train → Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f},  R²: {train_r2:.4f}")
    print(f"  Valid → RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f},  R²: {val_r2:.4f}")
    print(f"  Test  → RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f},  R²: {test_r2:.4f}")

print("\n=== Best Model Summary ===")
print(f"  TEST MAE : {test_maelast }, Valid RMSE: {best_val_rmse:.4f}")
torch.save(best_model_state, 'best_model.pth')

