import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader  # 更新导入路径
from torch_geometric.nn import NNConv, Set2Set
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen
from torch_geometric.utils import subgraph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ncps.torch import CfC,LTC
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from ncps.wirings import AutoNCP

# Featurizer 类定义保持不变
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

# 初始化 featurizer
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


    mol = Chem.AddHs(mol)  # 保证 mol 不是 None
    # 计算全局特征
    num_donors = rdMolDescriptors.CalcNumHBD(mol)
    num_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # 节点特征
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_featurizer.encode(atom))
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # 扩展特征维度
    rows, cols = node_features.shape
    zeros_tensor = torch.zeros(rows, 6)  # 扩展6个维度
    node_features = torch.cat((node_features, zeros_tensor), dim=1)

    # 为每个原子设置额外特性
    for i in range(rows):
        # B
        if node_features[i, 0] == 1:
            node_features[i, -1] = 2.04  # 电负性
            node_features[i, -2] = 82  # 共价半径
            node_features[i, -3] = 5   # 原子序数
            node_features[i, -4] = 10.82  # 原子质量
            node_features[i, -5] = 8.298  # 第一电离能
            node_features[i, -6] = 0.277  # 电子亲合能

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

    # 去掉前16列，保留最后的特征
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # 边特征
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_features.append(bond_featurizer.encode(bond))
        edge_features.append(bond_featurizer.encode(bond))

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    global_features = torch.tensor([num_donors, num_acceptors, logp, tpsa], dtype=torch.float32).unsqueeze(0)

    # 如果提供了浓度信息，将其添加到全局特征中
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)



def combine_molecules(smiles1, smiles2, x1, x2):

    graph1 = process_molecule(smiles1)
    graph2 = process_molecule(smiles2)

    # 偏移第二个图的节点索引
    offset = graph1.x.size(0)
    graph2.edge_index += offset

    # 合并节点特征和边信息
    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index], dim=1)
    combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)

    # 创建 mask1 和 mask2 掩码
    mask1 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    mask2 = torch.zeros(combined_x.size(0), dtype=torch.bool)
    mask1[:graph1.x.size(0)] = True
    mask2[graph1.x.size(0):] = True

    # 创建全局特征边和边特征
    global_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    global_edge_attr = torch.cat(
        [graph1.global_features, graph2.global_features], dim=0
    )
    global_node_attr = torch.tensor(
    [[x1, x2],
    [x2, x1]],dtype=torch.float32
    )

    return Data(
        x=combined_x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        global_edge_index=global_edge_index,
        global_edge_attr=global_edge_attr,
        global_node_attr = global_node_attr,
        mask1=mask1,
        mask2=mask2
    )


    # 创建掩码



def load_data(triple_csv_path):
    # 读取三分子数据
    triple_df = pd.read_csv(triple_csv_path)

    # 生成 SMILES 列表和目标值
    smiles1_triple = triple_df['solv1_smiles'].tolist()
    smiles2_triple = triple_df['solv2_smiles'].tolist()
    solv1_gamma_triple = triple_df['solv1_gamma'].tolist()
    solv2_gamma_triple = triple_df['solv2_gamma'].tolist()

    solv1_x_triple = triple_df['solv1_x'].tolist()  # 浓度
    solv2_x_triple = triple_df['solv2_x'].tolist()

    # 合并数据
    smiles1 = smiles1_triple
    smiles2 = smiles2_triple

    # 对于三分子数据，目标值有三个 (solv1_gamma, solv2_gamma, solv3_gamma)
    targets_triple = list(zip(solv1_gamma_triple, solv2_gamma_triple))

    # 浓度信息
    concentrations = list(zip(solv1_x_triple, solv2_x_triple))

    return smiles1, smiles2,  targets_triple, concentrations


class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, smiles1, smiles2,  targets, concentrations, transform=None, pre_transform=None):
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.targets = targets
        self.concentrations = concentrations
        super(MoleculesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['triple_molecule_data.csv']

    @property
    def processed_file_names(self):
        return ['dataLL.pt']

    def download(self):
        pass

    def process(self):
        datas = []
        for i in range(len(self.smiles1)):
            smile1 = self.smiles1[i]
            smile2 = self.smiles2[i]
            target = self.targets[i]
            concentration = self.concentrations[i]

            # 合并分子数据

            data = combine_molecules(smile1, smile2, concentration[0], concentration[1])

            if data is not None:
                data.y = torch.tensor(target, dtype=torch.float32)  # 目标值作为多维张量
                datas.append(data)
            else:
                print(f"Sample {i} is invalid and has been skipped.")

        # 保存处理好的数据
        torch.save(self.collate(datas), self.processed_paths[0])

# 加载数据
#dual_csv_path = '/home/ubuntu/二组分.csv'
triple_csv_path = '/home/ubuntu/二组分全.csv'

smiles1, smiles2,  targets, concentrations = load_data(triple_csv_path)

# 创建数据集对象
dataset = MoleculesDataset(root='dataLL', smiles1=smiles1, smiles2=smiles2, targets=targets, concentrations=concentrations)
print(len(dataset))

class MultiLevelGraphNetWithEdgeFeatures(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(MultiLevelGraphNetWithEdgeFeatures, self).__init__()

        self.transformer_layer = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=2)
        self.decoder_layer = TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=2)
        edge_hidden_dim = 32  # 确保定义 edge_hidden_dim
        self.a11 = NNConv(41, 32, nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(edge_hidden_dim, 41 * 32)
        ), aggr="mean")

        # 假设 CfC 是已定义的模块，保持不变
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



        self.xm = nn.Linear(64, 64)  # 示例，替换为 CfC

        self.relu = nn.ReLU()
        self.xm2 = CfC(64, 64, batch_first=True)  # 示例，替换为 CfC
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

        self.global_conv = NNConv(256+2, 128, nn.Sequential(
            nn.Linear(4, edge_hidden_dim),  # 4 是 global_edge_attr 的输入维度
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(edge_hidden_dim , 258*128)
        ), aggr='mean')

        self.set2set = Set2Set(hidden_dim, processing_steps=2)
        self.set2set2 = Set2Set(3*hidden_dim+2, processing_steps=2)
        self.FF = nn.Linear(6*hidden_dim+4,512)

        self.fc = nn.Sequential(
            nn.Linear(258+512 , 1024),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 1)  # 输出维度为2或3
        )

    def process_subgraph(self, x, edge_index, edge_attr, batch, mask):
        # 提取子图特征
        subgraph_x = x[mask]
        subgraph_edge_index, subgraph_edge_attr = subgraph(mask, edge_index, edge_attr, relabel_nodes=True)

        # 第一部分特征处理 (前 25 维)
        x1 = subgraph_x[:, 0:41]
        x1 = self.a11(x1, subgraph_edge_index, subgraph_edge_attr)
        x1 = self.relu(x1)

        # 第二部分特征处理 (25 维之后)
        x2 = subgraph_x[:, 41:]
        x2out = x2

        x2_input = x2.unsqueeze(1)
        predicted_steps = []
        hidden_state =torch.cat((x2,x2),dim=1)


        # 连续预测 3 个时间步
        for _ in range(5):
            output, hidden_state = self.lstm_a2_1(x2_input,hidden_state)
            predicted_steps.append(output.view(output.size(0), -1))
            # 展平每个时间步的输出
 # 展平每个时间步的输出

        # 将所有时间步结果拼接成一维向量

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
        # 合并处理特征
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

        # 图卷积处理
        subgraph_x = self.subgraph_conv1(xm, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        # 聚合子图特征
        subgraph_x = self.set2set(subgraph_x, batch[mask])
        return subgraph_x

    def forward(self, data):
        # 全局属性转移到设备
        global_edge_attr = data.global_edge_attr.to(data.x.device)
        global_node_attr = data.global_node_attr.to(data.x.device)

        # 处理每个子图
        subgraph1_x = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask1)
        subgraph2_x = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask2)
        subgraph1_x = self.relu(subgraph1_x)
        subgraph2_x = self.relu(subgraph2_x)


        # 确定当前批次大小
        batch_size = subgraph1_x.size(0)

        # 创建全局边索引
        new_global_edge_index = torch.empty((2, batch_size * 2), dtype=torch.long, device=subgraph1_x.device)
        for i in range(batch_size):
            new_global_edge_index[0][i * 2] = i * 2     # 0, 2, ...
            new_global_edge_index[1][i * 2] = i * 2 + 1 # 1, 3, ...
            new_global_edge_index[0][i * 2 + 1] = i * 2 + 1 # 1, 3, ...
            new_global_edge_index[1][i * 2 + 1] = i * 2 # 0, 2, ...

        global_edge_index = new_global_edge_index.view(2, batch_size * 2)

        # 创建扩展的节点特征
        expanded_x = torch.empty((batch_size * 2, subgraph1_x.size(1)), dtype=subgraph1_x.dtype, device=subgraph1_x.device)
        expanded_x[0::2] = subgraph1_x  # 在偶数索引处放置subgraph1_x
        expanded_x[1::2] = subgraph2_x
        expanded_x =torch.cat((expanded_x, global_node_attr), dim=1)

        # 通过全局消息传递


        combined_x = self.global_conv(expanded_x, global_edge_index, global_edge_attr.to(expanded_x.device))  # 确保边属性在同一设备上
        num_classes = batch_size  # 例如，从 0 到 4
        repeats_per_class = 2  # 每个数字重复的次数
        combined_x = self.relu(combined_x)
        # 生成张量
        tensor = torch.cat([torch.full((repeats_per_class,), i, dtype=torch.long) for i in range(num_classes)])
        tensor = tensor.to(combined_x.device)
        combined_x = torch.cat((combined_x,expanded_x),dim=1)
        set2set_x = self.set2set2(combined_x, tensor)
        set2set_x_shape = set2set_x.size()
        expanded_x_shape = expanded_x.size()
        expanded_set2set_x = set2set_x.unsqueeze(1).expand(-1, 2, -1)

        expanded_set2set_x = expanded_set2set_x.contiguous().view(-1, set2set_x_shape[1])

        expanded_set2set_x = self.FF(expanded_set2set_x)
        expanded_set2set_x = self.relu(expanded_set2set_x)

        final_x = torch.cat((expanded_x, expanded_set2set_x), dim=1)

        # 通过全连接层进行输出
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
from torch_geometric.loader import DataLoader  # 改成 PyG 的 DataLoader
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 设置训练参数
epochs = 250
k_folds = 5  # 五折交叉验证
batch_size = 256
input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim

hidden_dim = 128

# 根据目标值数量确定输出维度
output_dim = 1  # 2 或 3
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取数据集大小
dataset_size = len(dataset)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=2021)
start_fold = 4
# 交叉验证循环
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    if fold < start_fold:
        print(f"Skipping Fold {fold+1} (已经训练过)")
        continue

    print(f"开始训练 Fold {fold+1}/{k_folds}")

    # 创建训练集和验证集
    train_subset = [dataset[i] for i in train_idx]
    val_subset = [dataset[i] for i in val_idx]

    # 数据加载器
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

    # 初始化模型
    model = MultiLevelGraphNetWithEdgeFeatures(input_dim, edge_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()

    # 训练循环
    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        y_train_true = []
        y_train_pred = []

        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)  # 将数据转移到 GPU
            output = model(batch)  # 前向传播
            output = output.view(-1, 1)  # 确保输出形状
            target = batch.y.unsqueeze(1).to(device)  # 目标值

            # 计算损失
            loss = criterion(output, target)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item() * batch.num_graphs  # 累计损失

            # 记录真实值和预测值
            y_train_true.extend(target.cpu().numpy().flatten())
            y_train_pred.extend(output.detach().cpu().numpy().flatten())

        avg_train_loss = total_loss / len(train_loader.dataset)

        # 计算训练集 MAE、MSE 和 R²
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        train_mse = mean_squared_error(y_train_true, y_train_pred)
        train_r2 = r2_score(y_train_true, y_train_pred)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0
        y_val_true = []
        y_val_pred = []
        absolute_errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                output = output.view(-1, 1)  # 确保输出形状
                target = batch.y.unsqueeze(1).to(device)

                # 计算损失
                loss = criterion(output, target)
                val_loss += loss.item() * batch.num_graphs

                # 记录真实值和预测值
                y_val_true.extend(target.cpu().numpy().flatten())
                y_val_pred.extend(output.cpu().numpy().flatten())
                absolute_errors.extend(abs(target.cpu().numpy().flatten() - output.cpu().numpy().flatten()))

        avg_val_loss = val_loss / len(val_loader.dataset)

        # 计算验证集 MAE、MSE 和 R²
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        val_mse = mean_squared_error(y_val_true, y_val_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)

        # 输出当前 epoch 的结果
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")

    # 保存每折的误差
    np.savetxt(f'absolute_errors_fold{fold+1}.csv', absolute_errors)

    # 释放 GPU 内存
    del model
    torch.cuda.empty_cache()

