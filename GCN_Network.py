import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F

# 图卷积层
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, use_bias=True):
        super(GraphConvolutionLayer, self).__init__()

        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adj, in_feature):
        support = torch.mm(in_feature, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.use_bias:
            output += self.bias
        return output

# 利用自实现的图卷积层构建用于节点分类的图卷积网络
class GCNForClassification(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(GCNForClassification, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolutionLayer(input_feature_dim, hidden_layer_dim))
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GraphConvolutionLayer(hidden_layer_dim, hidden_layer_dim))
        self.layers.append(GraphConvolutionLayer(hidden_layer_dim, num_of_class))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        """
        参考：https://github.com/LingxiaoShawn/PairNorm/blob/master/layers.py
        mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
        """
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, adj):
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(adj, output)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        output = torch.sigmoid(output)
        return output


# 利用自实现的图卷积构建用于链路预测的图卷积网络
class GCNForLinkPrediction(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer, out_feature_dim,
                 input_feature_dim=1433, dropout=0.1, use_pair_norm=True):
        super(GCNForLinkPrediction, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolutionLayer(input_feature_dim, hidden_layer_dim))
        for i in range(num_of_hidden_layer - 1):
            self.layers.append(GraphConvolutionLayer(hidden_layer_dim, hidden_layer_dim))
        self.layers.append(GraphConvolutionLayer(hidden_layer_dim, out_feature_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm

    @staticmethod
    def PairNorm(x_feature):
        """
        参考：https://github.com/LingxiaoShawn/PairNorm/blob/master/layers.py
        mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
        """
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def encode(self, in_feature, adj):
        output = in_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)
            output = layer(adj, output)
            output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)
        return output

    @staticmethod
    def decode(out_feature, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=0)
        logits = (out_feature[edge_index[:, 0]] * out_feature[edge_index[:, 1]]).sum(dim=-1)
        logits = torch.sigmoid(logits)
        return logits
