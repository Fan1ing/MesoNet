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
# Featurizers
csv_path = '/home/ubuntu/aboso.csv'
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


    mol = Chem.AddHs(mol)  # 保证 mol 不是 None
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

        #Sn
        elif embeddings[i,15] == 1:
            embeddings[i,-1] = 1.96 #电负性
            embeddings[i,-2] = 141  #共价半径
            embeddings[i,-3] = 50   #原子序数
            embeddings[i,-4] = 118.71 #原子质量
            embeddings[i,-5] = 7.344 #第一电离能
            embeddings[i,-6] = 1.112 #电子亲合能

        #Te
        elif embeddings[i,16] == 1:
            embeddings[i,-1] = 2.1 #电负性
            embeddings[i,-2] = 135  #共价半径
            embeddings[i,-3] = 52   #原子序数
            embeddings[i,-4] = 127.6 #原子质量
            embeddings[i,-5] = 9.010 #第一电离能
            embeddings[i,-6] = 1.971 #电子亲合能

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


    # 全局特征
    global_features = torch.tensor([num_donors, num_acceptors, logp, tpsa], dtype=torch.float32).unsqueeze(0)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)


def combine_molecules(smiles1, smiles2):
    """
    处理两个分子的 SMILES 表示并组合成一个图
    :param smiles1: 第一个分子的 SMILES
    :param smiles2: 第二个分子的 SMILES
    :return: PyTorch Geometric 的 Data 对象
    """
    # 分别处理两个分子的图结构
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
    # 这里我们定义两条边 (0->1, 1->0) 来连接两个分子的全局特征
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
        self.NCP = CfC(6,AutoNCP(12,6), batch_first=True)

        self.x11 = nn.Linear(41,32)

        self.hidden = nn.Linear(6,12)

        self.last = nn.Linear(128,128)
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
        # 修改 global_conv 的权重网络
        self.global_conv = NNConv(hidden_dim*2,128, nn.Sequential(
            nn.Linear(4, edge_hidden_dim),  # 4 是 global_edge_attr 的输入维度
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(edge_hidden_dim , 2*hidden_dim*128)  # hidden_dim * 2 是输出维度
        ), aggr='mean')

        self.set2set2 = Set2Set(3*hidden_dim, processing_steps=2)
        self.set2set = Set2Set(hidden_dim, processing_steps=2)
        self.FF = nn.Linear(6*hidden_dim,64)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2+64,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(64, output_dim)
        )


    def process_subgraph(self, x, edge_index, edge_attr, batch, mask):

        subgraph_x = x[mask]
        subgraph_edge_index, subgraph_edge_attr = subgraph(mask, edge_index, edge_attr, relabel_nodes=True)

        # 第一部分特征处理 (前 25 维)
        x1 = subgraph_x[:, 0:41]

        x1 = self.a11(x1, subgraph_edge_index, subgraph_edge_attr)
        x1 = self.relu(x1)

        # 第二部分特征处理 (25 维之后)
        x2 = subgraph_x[:, 41:]
        x11 = x2

        x2_input = x2.unsqueeze(1)
        predicted_steps = []
        hidden_state =torch.cat((x2,x2),dim=1)


        # 连续预测 3 个时间步
        for _ in range(5):
            output, hidden_state = self.NCP(x2_input,hidden_state)
            predicted_steps.append(output.view(output.size(0), -1))
            # 展平每个时间步的输出
            # 展平每个时间步的输出

        # 将所有时间步结果拼接成一维向量

        x2_output = torch.cat(predicted_steps, dim=-1)
        x2_outputs = x2_output

        x2_output = self.x22(x2_output)
        x2_output = self.relu(x2_output)

        x2_output = self.a21(x2_output, subgraph_edge_index, subgraph_edge_attr)
        x2_output = self.relu(x2_output)
        # Transformer Encoder & Decoder
        combined_output = torch.stack([x1, x2_output], dim=1)  # Shape: [batch_size, seq_len=2, feature_dim=32]
        combined_output = combined_output.transpose(0, 1)  # Shape: [seq_len, batch_size, feature_dim]
        encoded_memory = self.transformer_encoder(combined_output)

        # Transformer Decoder
        seq_len_tgt = 2  # Example: Set the target sequence length
        tgt = encoded_memory.mean(dim=0, keepdim=True).repeat(seq_len_tgt, 1, 1)  # Shape: [seq_len_tgt, batch_size, feature_dim]

        decoded_output = self.transformer_decoder(tgt, encoded_memory)  # Decoder output     decoded_output = self.transformer_decoder(tgt, encoded_memory)  # Decoder output

        # ReLU activation on decoded output
        transformer_output = self.relu(decoded_output.mean(dim=0))  # Shape: [batch_size, feature_dim=32]
        transformer_outputs =transformer_output
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
        xm = self.last(xm)
        xm = self.relu(xm)
        # 图卷积处理
        subgraph_x = self.subgraph_conv1(xm, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        subgraph_x = self.relu(subgraph_x)
        subgraph_x = self.subgraph_conv2(subgraph_x, subgraph_edge_index, subgraph_edge_attr)
        # 聚合子图特征
        subgraph_x = self.set2set(subgraph_x, batch[mask])
        return subgraph_x,x11,x2_outputs,transformer_outputs

    def forward(self, data):
        # 处理每个子图
        global_edge_attr = data.global_edge_attr.to(data.x.device)

        subgraph1_x,x2_outputs,x2out,transformer_outputs = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask1)
        subgraph2_x,_,_,_ = self.process_subgraph(data.x, data.edge_index, data.edge_attr, data.batch, data.mask2)
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

        expanded_set2set_x = self.FF(expanded_set2set_x[0::2])
        expanded_set2set_x = self.relu(expanded_set2set_x)

        final_x = torch.cat((expanded_x[0::2],expanded_set2set_x), dim=1)
        # 通过全连接层进行输出
        output = self.fc(final_x)

        return output,x2_outputs,x2out,transformer_outputs


from torch_geometric.data import InMemoryDataset, Data

class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, smiles1, smiles2, ys, transform=None, pre_transform=None):
        """
        初始化 MoleculesDataset 数据集
        :param root: 数据存储路径
        :param smiles1: 第一组分子的 SMILES 表示
        :param smiles2: 第二组分子的 SMILES 表示
        :param ys: 分子对的目标值 (标签)
        :param transform: 数据变换
        :param pre_transform: 预处理变换
        """
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.ys = ys
        super(MoleculesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 如果有原始数据文件，返回文件名
        return ['dataff.csv']

    @property
    def processed_file_names(self):
        # 定义处理后保存的文件名
        return ['dataff.pt']

    def download(self):
        # 如果需要从网络下载数据，请在这里实现
        pass

    def process(self):
        """
        数据处理逻辑，将原始数据转换为图数据并保存
        """
        datas = []
        for smile1, smile2, y in zip(self.smiles1, self.smiles2, self.ys):
            # 调用 combine_molecules 生成分子对的图表示
            data = combine_molecules(smile1, smile2)
            data.y = torch.tensor([y], dtype=torch.float32)  # 添加目标值
            datas.append(data)

        # 保存处理后的数据
        torch.save(self.collate(datas), self.processed_paths[0])
dataset = MoleculesDataset(root="dataff", smiles1=smiles1, smiles2=smiles2, ys=ys)
print(len(dataset))
train_size = int(0.8 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(99)
)

'''train_dataset = dataset[0:660]+dataset[661:1562]+dataset[1566:1786]+dataset[1787:]
valid_dataset = dataset[660:661]+dataset[1562:1563]+dataset[1786:1787]+dataset[1622:1623]'''
'''train_dataset = dataset[0:1622]+dataset[1623:]
valid_dataset = dataset[1622:1623]'''
'''train_dataset = dataset[0:692]+dataset[693:1620]+dataset[1621:1845]+dataset[1846:]
valid_dataset = dataset[692:693]+dataset[1620:1621]+dataset[1845:1846]'''

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=10)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=10)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=10)
print(len(train_dataset ))
print(len(valid_dataset))


# 模型参数
input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 128
edge_hidden_dim = 32
output_dim = 1



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiLevelGraphNetWithEdgeFeatures(input_dim, edge_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# 设置训练参数
epochs = 1000
best_val_loss = float('inf')
best_model_state = None
early_stopping_counter = 0
patience = 10  # 提前停止的容忍轮数
mae = []
r = []

from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def mean_relative_error(y_true, y_pred):
    # 计算均相对误差 (MRE)
    return np.mean(np.abs((y_true - y_pred) / y_true))

for epoch in range(epochs):
    # ---------- 训练 ----------
    model.train()
    total_loss = 0
    y_train_true = []
    y_train_pred = []

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)  # 将数据转移到 GPU
        output, _, _, _ = model(batch)  # 前向传播
        output = output * 1000  # 恢复原始预测值
        output = output.view(-1, 1)  # 确保输出形状为 [batch_size, 1]
        target = batch.y.unsqueeze(1).to(device)  # 目标值
        target = target * 1000  # 恢复原始目标值

        # 计算损失
        loss = criterion(output, target)
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        total_loss += loss.item() * batch.num_graphs  # 累计损失

        # 收集预测值和真实值用于误差计算
        y_train_true.extend(target.cpu().numpy().flatten())  # 将目标值展平成一维数组
        y_train_pred.extend(output.detach().cpu().numpy().flatten())  # 同样展平成一维数组

    avg_train_loss = total_loss / len(train_loader.dataset)

    # 计算训练集 MAE、MRE 和 R²
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_mre = mean_relative_error(np.array(y_train_true), np.array(y_train_pred))
    train_r2 = r2_score(y_train_true, y_train_pred)

    # ---------- 验证 ----------
    model.eval()
    val_loss = 0
    y_val_true = []
    y_val_pred = []
    absolute_errors = []
    x1 = []
    x2 = []
    tran = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output, _, _, _ = model(batch)
            output = output * 1000  # 恢复原始预测值
            output = output.view(-1, 1)  # 确保输出形状为 [batch_size, 1]
            target = batch.y.unsqueeze(1).to(device)
            target = target * 1000  # 恢复原始目标值

            # 计算损失
            loss = criterion(output, target)
            val_loss += loss.item() * batch.num_graphs

            # 收集预测值和真实值用于误差计算
            y_val_true.extend(target.cpu().numpy().flatten())  # 确保展平成一维数组
            y_val_pred.extend(output.cpu().numpy().flatten())  # 确保展平成一维数组
            _, x2_output, x2out, trans = model(batch)  # 提取 x2_output

            # 将 x2_output 转换为 numpy 数组，并保存到列表中
            x1.append(x2_output.cpu().numpy())
            x2.append(x2out.cpu().numpy())
            tran.append(trans.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)

    # 计算验证集 MAE、MRE 和 R²
    val_mae = mean_absolute_error(y_val_true, y_val_pred)
    val_mre = mean_relative_error(np.array(y_val_true), np.array(y_val_pred))
    val_r2 = r2_score(y_val_true, y_val_pred)

    mae.append(val_mae)
    r.append(val_r2)

    # 输出当前 epoch 的结果
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, MRE: {train_mre:.4f}, R²: {train_r2:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, MRE: {val_mre:.4f}, R²: {val_r2:.4f}")



# 保存最终模型

# ---------- 测试模型性能 ----------
import numpy as np

# 保存误差到文件
x1 = np.concatenate(x1, axis=0)
x2 = np.concatenate(x2, axis=0)
tran = np.concatenate(tran, axis=0)
import numpy as np

# 保存误差到文件
np.savetxt('fffx1.csv', x1, delimiter=',')
np.savetxt('fffx2.csv', x2, delimiter=',')
np.savetxt('tran.csv', tran, delimiter=',')
