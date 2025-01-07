# 设置随机数种子
import random
import torch
import numpy as np
from matplotlib import pyplot as plt

# 设置随机数种子
def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

# 计算分类任务f1值
def micro_f1_score(y_pred, y_true):
        # Convert y_pred and y_true into binary tensors
        y_pred = (y_pred > 0.5).float()
        y_true = (y_true > 0.5).float()

        tp = (y_pred * y_true).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + 1e-16)  # 命中率
        recall = tp / (tp + fn + 1e-16)  # 召回率

        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        f1 = f1.mean()

        return f1

# 设置损失曲线可视化模块图片保存路径
def set_pic_save_path(
        learning_rate,
        epoch_num,
        layer_num,
        self_loop,
        pair_norm,
        drop_edge,
        activation_function = "relu",
        dataset_name='cora',
        task_name="NodeClassification",
):

    path = dataset_name + task_name + '_rate' + str(learning_rate).replace(".", "-") + '_epoch' + str(epoch_num) \
            + '_layer' + str(layer_num) + '_self_loop' if self_loop else '' + '_pair_norm' if pair_norm else '' \
            + '_drop_edge' + drop_edge + '_activation' + activation_function + '.png'
    return path

# 将训练过程中损失函数、准确率进行可视化
def plot_loss_with_acc(loss_history, val_loss_history, val_acc_history, plot_path):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)  # 将整幅画布分为1行2列，第一张子图位于第1个位置（画布左侧），用于记录损失函数
        ax1.plot(range(len(loss_history)), loss_history,color="#FF57B3", label="train_loss")  # color为颜色, label为图例
        ax1.plot(range(len(val_loss_history)), val_loss_history, color="#FFB357", label="val_loss")
        plt.ylabel('Loss')
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Training loss & Validation loss")

        ax2 = fig.add_subplot(122)  # 第二张子图位于第2个位置（画布右侧），用于记录损失函数
        ax2.plot(range(len(val_acc_history)), val_acc_history, color="#57B3FF", label="val_acc")
        ax2.yaxis.tick_right()  # 将准确率的y轴放在右侧，否则会和左边的子图叠在一起不美观
        ax2.yaxis.set_label_position("right")
        plt.ylabel('ValAcc')

        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Validation Accuracy')
        plt.savefig(plot_path)