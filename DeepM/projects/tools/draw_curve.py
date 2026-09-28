import json
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):
    """加载JSON数据文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data)


def interpolate_data(data, total_epochs=100):
    """将数据插值到指定的迭代次数"""
    epochs = data[:, 1]
    values = data[:, 2]
    # 生成 2 次多项式系数
    poly_coeffs = np.polyfit(epochs, values, 3)

    # 用对应的多项式计算 y 的值
    y_poly = np.polyval(poly_coeffs, np.arange(total_epochs))
    return y_poly

def plot_metrics():
    """绘制训练和测试指标曲线图"""
    # 加载数据
    train_acc = load_data('logs/train_acc.json')
    train_f1 = load_data('logs/train_f1.json')
    train_loss = load_data('logs/train_loss.json')
    test_acc = load_data('logs/test_acc.json')
    test_f1 = load_data('logs/test_f1.json')
    test_loss = load_data('logs/test_loss.json')

    # 插值处理
    train_acc_interp = interpolate_data(train_acc)
    train_f1_interp = interpolate_data(train_f1)
    train_loss_interp = interpolate_data(train_loss)
    test_acc_interp = interpolate_data(test_acc)
    test_f1_interp = interpolate_data(test_f1)
    test_loss_interp = interpolate_data(test_loss)

    # 创建图表
    epochs = np.arange(100)
    plt.figure(figsize=(15, 10))

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # # 准确率子图
    # plt.subplot(2, 2, 1)
    # plt.plot(epochs, train_acc_interp, label='训练准确率', color='#3498db', linewidth=2)
    # plt.plot(epochs, test_acc_interp, label='测试准确率', color='#e74c3c', linewidth=2, marker='o', markersize=3)
    # plt.title('模型准确率变化', fontsize=16, fontweight='bold')
    # plt.xlabel('迭代次数', fontsize=12)
    # plt.ylabel('准确率', fontsize=12)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # # F1分数子图
    # plt.subplot(2, 2, 2)
    # plt.plot(epochs, train_f1_interp, label='训练F1分数', color='#2ecc71', linewidth=2)
    # plt.plot(epochs, test_f1_interp, label='测试F1分数', color='#f39c12', linewidth=2, marker='s', markersize=3)
    # plt.title('模型F1分数变化', fontsize=16, fontweight='bold')
    # plt.xlabel('迭代次数', fontsize=12)
    # plt.ylabel('F1分数', fontsize=12)
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # 损失函数子图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_interp, label='训练损失', color='#9b59b6', linewidth=2)
    plt.plot(epochs, test_loss_interp, label='测试损失', color='#1abc9c', linewidth=2, marker='^', markersize=3)
    plt.title('模型损失函数变化', fontsize=16, fontweight='bold')
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 综合对比图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_interp, label='训练准确率', color='#3498db', linewidth=2)
    plt.plot(epochs, test_acc_interp, label='测试准确率', color='#e74c3c', linewidth=2)
    plt.plot(epochs, train_f1_interp, label='训练F1分数', color='#2ecc71', linewidth=2)
    plt.plot(epochs, test_f1_interp, label='测试F1分数', color='#f39c12', linewidth=2)
    plt.title('指标综合对比', fontsize=16, fontweight='bold')
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('指标值', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.suptitle('模型训练过程指标变化曲线图', fontsize=18, fontweight='bold')

    # 显示图表
    plt.show()


if __name__ == "__main__":
    plot_metrics()
