from Dataset import *
from GCN_Network import GCNForClassification
import torch.nn as nn
from utils import set_random_seed, micro_f1_score, set_pic_save_path, plot_loss_with_acc

# 产生tensor_adjacency
def generate_tensor_adjacency_for_classify(edge_index, drop_edge=0.05, self_loop=True):
    if drop_edge >= 1.0:
        adj = get_adjacent(edge_of_pg=edge_index, num_graph_node=num_nodes, symmetric_of_edge=True)
    else:
        adj = random_adjacent_sampler(edge_of_pg=edge_index, num_graph_node=num_nodes, symmetric_of_edge=True,
                                      drop_edge=drop_edge)

    normalize_adj = normalization(adj, self_loop=self_loop)

    index_of_coo_matrix = torch.from_numpy(np.asarray([normalize_adj.row,
                                                       normalize_adj.col]).astype('int64')).long()

    values_of_index_in_matrix = torch.from_numpy(normalize_adj.data.astype(np.float32))

    tensor_adjacency = torch.sparse.FloatTensor(
        index_of_coo_matrix, values_of_index_in_matrix,
        torch.Size([num_nodes, num_nodes]))
    return tensor_adjacency

def train():
        loss_list = []
        val_loss_list = []
        val_acc_history = []
        model.train()

        train_y = tensor_y[train_mask]
        for epoch in range(epoch_num):
            tensor_adjacency = generate_tensor_adjacency_for_classify(edge_index=edge_index, drop_edge=drop_edge).to(device)
            logits = model(tensor_x, tensor_adjacency)
            train_mask_logits = logits[train_mask]

            loss = compute_loss(train_mask_logits, train_y if dataset_name == "ppi" else train_y.long())  # 计算损失值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc, _ = test(train_mask)  # 计算当前模型训练集上的准确率
            val_acc, val_loss = test(val_mask)  # 计算当前模型在验证集上的准确率和损失函数

            # 记录训练过程中训练集、验证集损失函数和验证集准确率的变化，用于后续可视化
            loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())
            val_acc_history.append(val_acc.item())
            print("Epoch {:03d}: TrainLoss {:.4f}, TrainAcc {:.4f}, ValLoss {:.4f}, ValAcc {:.4f}".format(
                epoch, loss.item(), train_acc.item(), val_loss.item(), val_acc.item()))

        return loss_list, val_loss_list, val_acc_history

def test(mask):
        model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

        with torch.no_grad():  # 测试过程不计算梯度
            tensor_adjacency = generate_tensor_adjacency_for_classify(edge_index=edge_index,
                                                                      drop_edge=drop_edge, self_loop=self_loop).to(device)
            logits = model(tensor_x, tensor_adjacency)
            test_mask_logits = logits[mask]
            test_y = tensor_y[mask]

            if dataset_name == "ppi":
                test_loss = compute_loss(test_mask_logits, test_y)
                accuracy = micro_f1_score(test_mask_logits.cpu(), tensor_y[mask].cpu())
            else:
                test_loss = compute_loss(test_mask_logits, test_y.long())
                predict_y = test_mask_logits.max(1)[1]  # 返回每一行的最大值中索引（作为预测类别）
                accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
        return accuracy, test_loss

if __name__ == '__main__':

    set_random_seed(123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 超参数设置
    learning_rate = 5e-3
    epoch_num = 100
    weight_decay = 5e-4
    hidden_layer_dim = 512
    layer_num = 2
    drop_edge = 1
    use_pair_norm = True
    self_loop = True
    activation_function = "relu"

    # 读取数据
    dataset_name = "citeseer"

    dataset = None
    if dataset_name == "cora":
        dataset = CoraData("./cora/cora")
    elif dataset_name == "citeseer":
        dataset = CiteseerData("./citeseer/citeseer")
    elif dataset_name == "ppi":
        dataset = PPIDataFromJson("./ppi/ppi")

    num_nodes = dataset.num_nodes
    edge_index = dataset.edge_of_pg
    train_mask, val_mask, test_mask = dataset.data_partition_node()
    num_of_class = dataset.num_of_class
    feature_dim = dataset.feature_dim

    tensor_x = torch.tensor(dataset.feature_of_pg, device=device, dtype=torch.float)
    tensor_y = torch.tensor(dataset.label_of_pg, device=device, dtype=torch.float)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # 模型定义
    model = GCNForClassification(hidden_layer_dim=hidden_layer_dim,
                                num_of_hidden_layer=layer_num, use_pair_norm=use_pair_norm,
                                num_of_class=num_of_class, input_feature_dim=feature_dim).to(device)

    if dataset_name == "ppi":
        # BCE Loss 更适合如ppi数据集一样的多标签分类任务
        compute_loss = nn.BCELoss().to(device)
    else:
        compute_loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss, val_loss, val_acc = train()
    test_acc, _ = test(test_mask)
    print("Test accuracy: ", test_acc.item())
    plot_path = set_pic_save_path(
        learning_rate=learning_rate,
        epoch_num=epoch_num,
        layer_num=layer_num,
        self_loop=self_loop,
        pair_norm=use_pair_norm,
        drop_edge=drop_edge,
        activation_function=activation_function,
        dataset_name=dataset_name,
        task_name="NodeClassification"
    )
    plot_loss_with_acc(loss, val_loss, val_acc, plot_path)
