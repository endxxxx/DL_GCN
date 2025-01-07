import random
import json
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.datasets import PPI


class CiteseerData:
    """
    Citeseer数据集预处理
    """
    def __init__(self, path_of_citeseer):
        self.path_of_citeseer = path_of_citeseer

        self.index_of_pg = dict()
        self.index_of_pg_label = dict()

        self.feature_of_pg = []
        self.label_of_pg = []
        self.edge_of_pg = []

        self.__dataset_loader()
        self.num_nodes = len(self.feature_of_pg)
        self.num_edges = len(self.edge_of_pg)
        self.num_of_class = len(self.index_of_pg_label)
        self.feature_dim = len(self.feature_of_pg[0])

    def __dataset_loader(self):
        # path = "./cora/cora"
        path_cites = self.path_of_citeseer + "/citeseer.cites"
        path_contents = self.path_of_citeseer + "/citeseer.content"

        with open(path_contents, 'r', encoding='utf-8') as file_content:
            for node in file_content.readlines():
                node_cont = node.split()
                self.index_of_pg[node_cont[0]] = len(self.index_of_pg)
                self.feature_of_pg.append([int(i) for i in node_cont[1:-1]])

                label = node_cont[-1]
                if label not in self.index_of_pg_label.keys():
                    self.index_of_pg_label[label] = len(self.index_of_pg_label)
                self.label_of_pg.append(self.index_of_pg_label[label])

        with open(path_cites, 'r', encoding='utf-8') as file_cite:
            for edge in file_cite.readlines():
                cited, citing = edge.split()

                if (cited not in self.index_of_pg.keys()) or (citing not in self.index_of_pg.keys()):
                    continue

                edge_fir = [self.index_of_pg[citing], self.index_of_pg[cited]]
                edge_sec = [self.index_of_pg[cited], self.index_of_pg[citing]]
                if edge_fir not in self.edge_of_pg:
                    self.edge_of_pg.append([self.index_of_pg[citing], self.index_of_pg[cited]])
                if edge_sec not in self.edge_of_pg:
                    self.edge_of_pg.append([self.index_of_pg[cited], self.index_of_pg[citing]])

    def get_adjacent(self):
        graph_w = np.ones(self.num_edges)
        np_edge = np.array(self.edge_of_pg)
        adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    def random_adjacent_sampler(self, drop_edge=0.1):
        new_edge_of_pg = []
        half_edge_num = int(len(self.edge_of_pg) / 2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(self.edge_of_pg[2 * i])
                new_edge_of_pg.append(self.edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # 划分数据分类训练，验证，测试集
    @staticmethod
    def data_partition_node(data_size=3312):
        mask = torch.randperm(data_size)
        train_mask = mask[:180]
        val_mask = mask[180:800]
        test_mask = mask[2312:3312]
        return train_mask, val_mask, test_mask


# 读取cora数据
class CoraData:
    """
    Cora数据集预处理
    """
    def __init__(self, path_of_cora):
        self.cora_path = path_of_cora

        self.index_of_pg = dict()
        self.index_of_pg_label = dict()

        self.feature_of_pg = []
        self.label_of_pg = []
        self.edge_of_pg = []

        self.__dataset_loader()
        self.num_nodes = len(self.feature_of_pg)
        self.num_edges = len(self.edge_of_pg)
        self.num_of_class = len(self.index_of_pg_label)
        self.feature_dim = len(self.feature_of_pg[0])

    def __dataset_loader(self):
        path_cites = self.cora_path + "/cora.cites"
        path_contents = self.cora_path + "/cora.content"

        with open(path_contents, 'r', encoding='utf-8') as file_content:
            for node in file_content.readlines():
                node_cont = node.split()
                self.index_of_pg[node_cont[0]] = len(self.index_of_pg)
                self.feature_of_pg.append([int(i) for i in node_cont[1:-1]])

                label = node_cont[-1]
                if label not in self.index_of_pg_label.keys():
                    self.index_of_pg_label[label] = len(self.index_of_pg_label)
                self.label_of_pg.append(self.index_of_pg_label[label])

        with open(path_cites, 'r', encoding='utf-8') as file_cite:
            for edge in file_cite.readlines():
                cited, citing = edge.split()
                edge_fir = [self.index_of_pg[citing], self.index_of_pg[cited]]
                edge_sec = [self.index_of_pg[cited], self.index_of_pg[citing]]
                if edge_fir not in self.edge_of_pg:
                    self.edge_of_pg.append(edge_fir)
                if edge_sec not in self.edge_of_pg:
                    self.edge_of_pg.append(edge_sec)

    def get_adjacent(self):
        graph_w = np.ones(self.num_edges)
        np_edge = np.array(self.edge_of_pg)
        adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    def random_adjacent_sampler(self, drop_edge=0.1):
        new_edge_of_pg = []
        half_edge_num = int(len(self.edge_of_pg)/2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(self.edge_of_pg[2 * i])
                new_edge_of_pg.append(self.edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # 根据需求划分训练，验证，测试集
    @staticmethod
    def data_partition_node(data_size=2708):
        mask = torch.randperm(data_size)
        train_mask = mask[:140]
        val_mask = mask[140:640]
        test_mask = mask[1708:2708]
        return train_mask, val_mask, test_mask

class PPISplitData:
    def __init__(self, split="train"):
        self.ppi_model = None
        self.num_nodes = 0
        self.num_edges = 0
        self.num_of_class = 0
        self.feature_dim = 0
        self.feature_of_pg = None
        self.edge_of_pg = None
        self.label_of_pg = None
        self.undirected = True
        self.split = split

        self.generate_ppi_model(split=split)

    # 产生PPI数据集
    def generate_ppi_model(self, split="train"):
        ppi_model = PPI(root="../ppi", split=split)
        data_of_ppi = ppi_model.data
        self.num_nodes = data_of_ppi.num_nodes
        self.num_edges = data_of_ppi.num_edges
        self.num_of_class = ppi_model.num_classes
        self.feature_dim = data_of_ppi.num_node_features
        self.feature_of_pg = data_of_ppi['x']
        self.edge_of_pg = data_of_ppi['edge_index']
        self.label_of_pg = data_of_ppi['y']
        self.undirected = data_of_ppi.is_undirected()


class PPIData:
    def __init__(self):
        self.split_train_set = PPISplitData(split="train")
        self.split_val_set = PPISplitData(split="val")
        self.split_test_set = PPISplitData(split="test")

        self.num_nodes = 0
        self.num_edges = 0
        self.num_of_class = 0
        self.feature_dim = 0
        self.feature_of_pg = None
        self.edge_of_pg = None
        self.label_of_pg = None
        self.undirected = True

        self.generate_whole_dataset()

    # 将三个数据集拼接在一起
    def generate_whole_dataset(self):
        self.edge_of_pg = (torch.cat([self.split_train_set.edge_of_pg, self.split_val_set.edge_of_pg,
                                      self.split_test_set.edge_of_pg], dim=-1).t()).numpy()
        self.num_nodes = self.split_train_set.num_nodes + self.split_val_set.num_nodes + self.split_test_set.num_nodes
        self.num_edges = self.split_train_set.num_edges + self.split_val_set.num_edges + self.split_test_set.num_edges
        self.num_of_class = self.split_train_set.num_of_class
        self.feature_dim = self.split_train_set.feature_dim
        self.feature_of_pg = (torch.cat([self.split_train_set.feature_of_pg, self.split_val_set.feature_of_pg,
                                        self.split_test_set.feature_of_pg], dim=0)).numpy()
        self.label_of_pg = (torch.cat([self.split_train_set.label_of_pg, self.split_val_set.label_of_pg,
                                      self.split_test_set.label_of_pg], dim=0)).numpy()
        return

    # 产生mask
    def data_partition_node(self):
        train_num = self.split_train_set.num_nodes
        val_num = self.split_val_set.num_nodes
        test_num = self.split_test_set.num_nodes
        train_mask = torch.arange(0, train_num)
        val_mask = torch.arange(train_num, train_num+val_num)
        test_mask = torch.arange(train_num+val_num, train_num+val_num+test_num)
        return train_mask, val_mask, test_mask


class PPIDataFromJson:
    def __init__(self, path_of_ppi):
        self.ppi_path = path_of_ppi
        self.feature_of_pg = None
        self.label_of_pg = None

        self.edge_of_pg = []
        self.train_mask = []
        self.test_mask = []
        self.val_mask = []
        self.num_nodes = 0
        self.num_edges = 0
        self.num_of_class = 0
        self.feature_dim = 0

        self.get_node_feature()
        self.get_edge_index()
        self.get_node_label()

    # 读取节点处的feature信息
    def get_node_feature(self):
        path_of_feature = self.ppi_path + "/ppi-feats.npy"
        self.feature_of_pg = np.load(path_of_feature)
        self.num_nodes = len(self.feature_of_pg)
        self.feature_dim = len(self.feature_of_pg[0])
        return

    # 读取节点连接信息以及划分数据集的函数
    def get_edge_index(self):
        graph = self.ppi_path + "/ppi-G.json"
        with open(graph, 'r', encoding='utf-8') as fp:
            json_format = json.load(fp)
            for nodes in json_format['nodes']:
                test_bool = nodes['test']
                node_id = int(nodes['id'])
                val_bool = nodes['val']
                if test_bool:
                    self.test_mask.append(node_id)
                elif val_bool:
                    self.val_mask.append(node_id)
                else:
                    self.train_mask.append(node_id)

            for edges in json_format['links']:
                source = edges['source']
                target = edges['target']
                if source != target:
                    self.edge_of_pg.append([source, target])
                    self.edge_of_pg.append([target, source])
        self.num_edges = len(self.edge_of_pg)
        return

    # 读取节点处标签
    def get_node_label(self):
        labels = self.ppi_path + "/ppi-class_map.json"
        with open(labels, 'r', encoding='utf-8') as fp:
            json_format = json.load(fp)
            self.num_of_class = len(json_format['0'])
            self.label_of_pg = np.ones([len(json_format), len(json_format['0'])], dtype=float)
            for label in json_format.keys():
                key = int(label)
                self.label_of_pg[key] = np.array(json_format[label])
        return

    def data_partition_node(self):
        train_mask_tensor = torch.tensor(self.train_mask, dtype=torch.long)
        val_mask_tensor = torch.tensor(self.val_mask, dtype=torch.long)
        test_mask_tensor = torch.tensor(self.test_mask, dtype=torch.long)
        return train_mask_tensor, val_mask_tensor, test_mask_tensor

def data_partition_edge(edge_of_pg, num_graph_node):
    edge_of_pos_pg = edge_of_pg[::2]
    edge_of_neg_pg = []

    neg_node_num = int(5 * np.sqrt(len(edge_of_pos_pg)))

    if num_graph_node <= neg_node_num:
        for i in range(num_graph_node):
            for j in range(i+1, num_graph_node):
                edge = [i, j]
                inverse_edge = [j, i]
                if (edge not in edge_of_pos_pg) and (inverse_edge not in edge_of_pos_pg):
                    edge_of_neg_pg.append(edge)
    else:
        sampler_row = random.sample(range(0, num_graph_node), neg_node_num)
        for row in sampler_row:
            for col in range(row+1, neg_node_num):
                edge = [row, col]
                inverse_edge = [col, row]
                if (edge not in edge_of_pos_pg) and (inverse_edge not in edge_of_pos_pg):
                    edge_of_neg_pg.append(edge)

    edge_of_pos_pg = np.array(edge_of_pos_pg)
    edge_of_neg_pg = np.array(edge_of_neg_pg)

    num_pos_edge = len(edge_of_pos_pg)
    perm = np.random.permutation(num_pos_edge)  # 随机排列
    num_of_train_pos_edge = int(num_pos_edge * 0.85)  # 取出比例条边
    train_pos = perm[:num_of_train_pos_edge]  # 对随机排列的边取出 前num_of_train_pos_edge条边
    train_pos_edge_index = edge_of_pos_pg[train_pos]
    num_of_val_pos_edge = int(num_pos_edge * 0.05)
    val_pos = perm[num_of_train_pos_edge:(num_of_train_pos_edge+num_of_val_pos_edge)]
    validate_pos_edge_index = edge_of_pos_pg[val_pos]
    test_pos = perm[(num_of_train_pos_edge+num_of_val_pos_edge):]
    test_pos_edge_index = edge_of_pos_pg[test_pos]

    num_neg_edge = len(edge_of_neg_pg)
    perm = np.random.permutation(num_neg_edge)  # 随机排列
    num_of_train_neg_edge = int(num_neg_edge * 0.85)  # 取出比例条边
    train_neg = perm[:num_of_train_neg_edge]  # 对随机排列的边取出 前num_of_train_pos_edge条边
    train_neg_edge_index = edge_of_neg_pg[train_neg]
    num_of_val_neg_edge = int(num_neg_edge * 0.05)
    val_neg = perm[num_of_train_neg_edge:(num_of_train_neg_edge+num_of_val_neg_edge)]
    validate_neg_edge_index = edge_of_neg_pg[val_neg]
    test_neg = perm[(num_of_train_neg_edge+num_of_val_neg_edge):]
    test_neg_edge_index = edge_of_neg_pg[test_neg]

    return train_pos_edge_index, validate_pos_edge_index, test_pos_edge_index, train_neg_edge_index, \
                                                        validate_neg_edge_index, test_neg_edge_index


def negative_edge_sampling(train_neg_edge_index, train_pos_edge_index):
    num_pos_edge = len(train_pos_edge_index)
    num_neg_edge = len(train_neg_edge_index)
    perm = np.random.permutation(num_neg_edge)  # 随机排列
    train_neg = perm[:num_pos_edge]
    sampler_train_neg_edge_index = train_neg_edge_index[train_neg]
    return sampler_train_neg_edge_index


def get_link_labels(pos_edge_index, neg_edge_index, device):
    num_of_edge = pos_edge_index.size(0) + neg_edge_index.size(0)
    link_labels = torch.zeros(num_of_edge, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(0)] = 1.
    return link_labels

def get_adjacent(edge_of_pg, num_graph_node, symmetric_of_edge=False):
    if not symmetric_of_edge:
        new_edge_of_pg = convert_symmetric(edge_of_pg)
    else:
        new_edge_of_pg = np.copy(edge_of_pg)
    num_edges = len(new_edge_of_pg)
    graph_w = np.ones(num_edges)
    np_edge = np.array(new_edge_of_pg)
    adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                        shape=[num_graph_node, num_graph_node])

    return adj

def convert_symmetric(edge_of_pg):
    new_edge_of_pg = []
    for edge_index in edge_of_pg:
        symmetric_edge_index = [edge_index[1], edge_index[0]]
        if symmetric_edge_index not in edge_of_pg:
            new_edge_of_pg.append(symmetric_edge_index)

    new_edge_of_pg.extend(edge_of_pg)
    return np.array(new_edge_of_pg)

def random_adjacent_sampler(edge_of_pg, num_graph_node, drop_edge=0.1, symmetric_of_edge=False):
    if not symmetric_of_edge:
        new_edge_of_pg = []
        edge_num = int(len(edge_of_pg))
        sampler = np.random.rand(edge_num)
        for i in range(int(edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[i])
        new_edge_of_pg = np.array(new_edge_of_pg)
        new_edge_of_pg = convert_symmetric(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    else:
        new_edge_of_pg = []
        half_edge_num = int(len(edge_of_pg) / 2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[2 * i])
                new_edge_of_pg.append(edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    return adj

def normalization(adj, self_loop=True):
    adj = sp.coo_matrix(adj)
    if self_loop:
        adj += sp.eye(adj.shape[0])  # 增加自连接
    row_sum = np.array(adj.sum(1))  # 对列求和，得到每一行的度
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_hat = sp.diags(d_inv_sqrt)
    return d_hat.dot(adj).dot(d_hat).tocoo()   # 返回coo_matrix形式