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
from torch_geometric.nn import NNConv, Set2Set
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,mean_absolute_percentage_error
from rdkit.Chem import rdMolDescriptors,Crippen
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

csv_path = '/MesoNet/data/IE.csv'
df = pd.read_csv(csv_path)
y1 = df['IE']
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
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'IEdata.csv'

    @property
    def processed_file_names(self):
        return 'IEdata.pt'

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

            
            rows, cols = embeddings.shape
            zeros_tensor = torch.zeros(rows, 6)
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
                # Continue similarly for other elements like Br, C, Cl, etc.

                #Br
                elif embeddings[i,1] == 1:
                    embeddings[i,-1] = 2.96
                    embeddings[i,-2] = 114
                    embeddings[i,-3] = 35
                    embeddings[i,-4] = 79.904
                    embeddings[i,-5] = 11.814
                    embeddings[i,-6] = 3.364

                #C
                elif embeddings[i,2] == 1:
                    embeddings[i,-1] = 2.55
                    embeddings[i,-2] = 77
                    embeddings[i,-3] = 6
                    embeddings[i,-4] = 12.011
                    embeddings[i,-5] = 11.261
                    embeddings[i,-6] = 1.595

                #Cl
                elif embeddings[i,3] == 1:
                    embeddings[i,-1] = 3.16
                    embeddings[i,-2] = 99
                    embeddings[i,-3] = 17
                    embeddings[i,-4] = 35.45
                    embeddings[i,-5] = 12.968
                    embeddings[i,-6] = 3.62
                #F
                elif embeddings[i,4] == 1:
                    embeddings[i,-1] = 3.98
                    embeddings[i,-2] = 71
                    embeddings[i,-3] = 9
                    embeddings[i,-4] = 18.998
                    embeddings[i,-5] = 17.422
                    embeddings[i,-6] = 3.40

            #Ge
                elif embeddings[i,5] == 1:
                    embeddings[i,-1] = 2.01
                    embeddings[i,-2] = 122
                    embeddings[i,-3] = 32
                    embeddings[i,-4] = 72.63
                    embeddings[i,-5] = 7.90
                    embeddings[i,-6] = 1.23


                #H
                elif embeddings[i,6] == 1:
                    embeddings[i,-1] = 2.20
                    embeddings[i,-2] = 37
                    embeddings[i,-3] = 1
                    embeddings[i,-4] = 1.008
                    embeddings[i,-5] = 13.598
                    embeddings[i,-6] = 0.755


                #I
                elif embeddings[i,7] == 1:
                    embeddings[i,-1] = 2.66
                    embeddings[i,-2] = 133
                    embeddings[i,-3] = 53
                    embeddings[i,-4] = 126.9
                    embeddings[i,-5] = 10.451
                    embeddings[i,-6] = 3.060

                 #N
                elif embeddings[i,8] == 1:
                    embeddings[i,-1] = 3.04
                    embeddings[i,-2] = 75
                    embeddings[i,-3] = 7
                    embeddings[i,-4] = 14.007
                    embeddings[i,-5] = 14.534
                    embeddings[i,-6] = 0.07

                 #Na
                elif embeddings[i,9] == 1:
                    embeddings[i,-1] = 0.93
                    embeddings[i,-2] = 154
                    embeddings[i,-3] = 11
                    embeddings[i,-4] = 22.99
                    embeddings[i,-5] = 5.139
                    embeddings[i,-6] = 0.547

                 #O
                elif embeddings[i,10] == 1:
                    embeddings[i,-1] = 3.44
                    embeddings[i,-2] = 73
                    embeddings[i,-3] = 8
                    embeddings[i,-4] = 15.999
                    embeddings[i,-5] = 13.618
                    embeddings[i,-6] = 1.46

                 #P
                elif embeddings[i,11] == 1:
                    embeddings[i,-1] = 2.19
                    embeddings[i,-2] = 106
                    embeddings[i,-3] = 15
                    embeddings[i,-4] = 30.974
                    embeddings[i,-5] = 10.487
                    embeddings[i,-6] = 0.75

                #S
                elif embeddings[i,12] == 1:
                    embeddings[i,-1] = 2.58
                    embeddings[i,-2] = 102
                    embeddings[i,-3] = 16 
                    embeddings[i,-4] = 32.06
                    embeddings[i,-5] = 10.36 
                    embeddings[i,-6] = 2.07 

                #Se
                elif embeddings[i,13] == 1:
                    embeddings[i,-1] = 2.55 
                    embeddings[i,-2] = 116 
                    embeddings[i,-3] = 34 
                    embeddings[i,-4] = 78.971 
                    embeddings[i,-5] = 9.753
                    embeddings[i,-6] = 2.02
                #Si
                elif embeddings[i,14] == 1:
                    embeddings[i,-1] = 1.90 
                    embeddings[i,-2] = 111  
                    embeddings[i,-3] = 14   
                    embeddings[i,-4] = 28.085 
                    embeddings[i,-5] = 8.151 
                    embeddings[i,-6] = 1.385

                #Te
                elif embeddings[i,15] == 1:
                    embeddings[i,-1] = 2.1 
                    embeddings[i,-2] = 135
                    embeddings[i,-3] = 52  
                    embeddings[i,-4] = 127.6 
                    embeddings[i,-5] = 9.010 
                    embeddings[i,-6] = 1.971 

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


            data = Data(x=embeddings, y=y, edge_index=edges, edge_attr=edge_attr,global_features=global_features)
            datas.append(data)

        # self.data, self.slices = self.collate(datas)
        torch.save(self.collate(datas), self.processed_paths[0])

max_nodes = 128
dataset = MoleculesDataset(root= "IEdata")


# for i in train_loader:
#     print(i.edge_attr.shape)

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
        self.x11 = nn.Linear(42,32)

        self.hidden = nn.Linear(6,12)


        self.x211 = nn.Linear(12,6)

        self.x22 = nn.Linear(30,32)
        self.a21 = NNConv(32, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 32 * 32)
        ), aggr="mean")



        self.xm = nn.Linear(64, 64)

        self.relu = nn.ReLU()
        self.xm2 = CfC(64, 64, batch_first=True)
        self.xm3 = nn.Linear(128,128)
        self.subgraph_conv1 = NNConv(128, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(edge_hidden_dim, 128 * hidden_dim)
        ), aggr='mean')

        self.subgraph_conv2 = NNConv(hidden_dim, hidden_dim, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(edge_hidden_dim, hidden_dim * hidden_dim)
        ), aggr='mean')
        self.global_conv = NNConv(hidden_dim*2,128, nn.Sequential(
            nn.Linear(4, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(edge_hidden_dim , 2*hidden_dim*128)
        ), aggr='mean')

        self.set2set2 = Set2Set(3*hidden_dim, processing_steps=2)
        self.set2set = Set2Set(hidden_dim, processing_steps=2)
        self.FF = nn.Linear(6*hidden_dim,64)
        self.group = nn.Linear(19,128)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2+128,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )


    def forward(self,data):
        x, edge_index, edge_attr,batch,group = data.x, data.edge_index, data.edge_attr,data.batch,data.global_features


        x1 = x[:, 0:41]
        x1 = self.a11(x1, edge_index, edge_attr)
        x1 = self.relu(x1)

        x2 = x[:, 41:]
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
        x2_output = self.a21(x2_output, edge_index, edge_attr)
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

        subgraph_x = self.subgraph_conv1(xm, edge_index, edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, edge_index, edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, edge_index, edge_attr)

        subgraph_x = self.set2set(subgraph_x, batch)
        group = self.group(group)
        group = self.relu(group)

        subgraph_x = torch.cat((subgraph_x,group),dim = 1)
        output = self.fc(subgraph_x)

        return output,x1,x2_outputs,transformer_outputs



input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 128
edge_hidden_dim = 32
output_dim = 1
batch_size=128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MesoNet(input_dim, edge_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
train_size = int(0.8 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(991)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=10)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)


epochs = 1000
early_stopping_counter = 0
patience = 10
mae = []
r = []

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modify the existing training loop to track the best model
best_epoch = 0  # Initialize the epoch with the best result
best_model_state = None  # To store the best model state
best_val_loss = float('inf')  # Initialize the best validation loss to infinity
best_val_mae = float('inf')
best_val_mse = float('inf')
best_val_r2 = float('inf')

for epoch in range(epochs):

    model.train()
    total_loss = 0
    y_train_true = []
    y_train_pred = []

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        output,_,_,_ = model(batch)
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
    x1 = []
    x2 = []
    tran = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output,_,_,_ = model(batch)
            output = output.view(-1, 1)
            target = batch.y.unsqueeze(1).to(device)

            loss = criterion(output, target)
            val_loss += loss.item() * batch.num_graphs

            y_val_true.extend(target.cpu().numpy().flatten())
            y_val_pred.extend(output.cpu().numpy().flatten())
            _, x2_output,x2out,trans = model(batch)

            x1.append(x2_output.cpu().numpy())
            x2.append(x2out.cpu().numpy())
            tran.append(trans.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)

    # Calculate additional metrics
    val_mae = mean_absolute_error(y_val_true, y_val_pred)
    val_mse = mean_squared_error(y_val_true, y_val_pred)
    val_r2 = r2_score(y_val_true, y_val_pred)

    # Check if the validation loss has improved
    if val_mae < best_val_mae:
        best_val_loss = avg_val_loss  # Update best validation loss
        best_val_mae = val_mae
        best_val_mse = val_mse
        best_val_r2 = val_r2
        best_epoch = epoch + 1  # Save the current epoch number (1-based indexing)
        best_model_state = model.state_dict()  # Save the current best model state

    mae.append(val_mae)
    r.append(val_r2)

    # Print out the results for this epoch
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, MRE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, MRE: {val_mse:.4f}, R²: {val_r2:.4f}")

# After the training loop, print the best epoch and its performance
print(f"\nBest Model Performance:")
print(f"  Best Epoch: {best_epoch}")
print(f"  Best Validation Loss: {best_val_loss:.4f}")
print(f"  Best Validation MAE: {best_val_mae:.4f}")
print(f"  Best Validation MSE: {best_val_mse:.4f}")
print(f"  Best Validation R²: {best_val_r2:.4f}")

# Save the best model
torch.save(best_model_state, 'best_model.pth')

x1 = np.concatenate(x1, axis=0)
x2 = np.concatenate(x2, axis=0)
tran = np.concatenate(tran, axis=0)
import numpy as np

np.savetxt('x1.csv', x1, delimiter=',')
np.savetxt('x2.csv', x2, delimiter=',')
np.savetxt('tran.csv', tran, delimiter=',')
