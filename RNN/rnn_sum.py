import torch
import torch.nn as nn
import torch.optim as optim


# ==============================
# 1. 定义一个 RNN 模型
#    功能：输入一个数字序列，输出这个序列的“和”
# ==============================
class SumRNN(nn.Module):
    def __init__(self):
        super(SumRNN, self).__init__()

        # 定义 RNN 层
        # input_size=1:
        #   表示每个时间步只输入 1 个数字
        #   例如序列 [1,2,3,4,5] 会被看成 5 个时间步，
        #   每一步输入分别是 [1]、[2]、[3]、[4]、[5]
        #
        # hidden_size=8:
        #   表示隐藏状态 h 的维度是 8
        #   也就是 RNN 在内部用一个 8 维向量来“记忆”前面看过的信息
        #
        # num_layers=1:
        #   表示只用 1 层 RNN
        #   对初学者最容易理解
        #
        # batch_first=True:
        #   表示输入 x 的形状使用：
        #   (batch_size, seq_len, input_size)
        #   也就是：
        #   批大小、序列长度、每步输入维度
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=8,
            num_layers=1, # 纵向的 RNN，表示网络深度
            batch_first=True
        )

        # 定义全连接层
        # 作用：
        #   把 RNN 最后一个时间步输出的 8 维特征
        #   映射成 1 个数值（预测的和）
        #
        # 输入维度：8
        # 输出维度：1
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        """
        x 的形状: (batch_size, seq_len, 1)

        举例：
        如果 batch_size=2, seq_len=5
        那么 x.shape = (2, 5, 1)

        可能表示两个样本：
        样本1: [1,2,3,4,5]
        样本2: [4,1,0,2,3]

        但在张量里会写成：
        [
          [[1],[2],[3],[4],[5]],
          [[4],[1],[0],[2],[3]]
        ]
        """

        # 取当前 batch 的大小
        # x.size(0) 就是 batch_size
        batch_size = x.size(0)

        # 初始化隐藏状态 h0
        #
        # 对于 nn.RNN，初始隐藏状态的形状必须是：
        # (num_layers, batch_size, hidden_size)
        #
        # 这里：
        # num_layers = 1
        # batch_size = 当前输入批次大小
        # hidden_size = 8
        #
        # 所以 h0.shape = (1, batch_size, 8)
        #
        # 为什么初始化成全 0？
        # 因为在序列开始之前，模型还没有看到任何信息，
        # 所以让“初始记忆”从 0 开始。
        
        h0 = torch.zeros(1, batch_size, 8).to(x.device)

        # 把输入序列 x 和初始隐藏状态 h0 一起送入 RNN
        #
        # 返回两个结果：
        #
        # out:
        #   所有时间步的输出
        #   形状: (batch_size, seq_len, hidden_size)
        #
        # _:
        #   最后一个隐藏状态 hn
        #   因为这里我们后面没直接用它，所以用 _ 忽略掉
        #
        # 假设：
        # batch_size = 2
        # seq_len = 5
        # hidden_size = 8
        #
        # 那么 out.shape = (2, 5, 8)
        #
        # 意思是：
        # 2 个样本
        # 每个样本有 5 个时间步
        # 每个时间步输出一个 8 维向量
        out, _ = self.rnn(x, h0)

        # 只取最后一个时间步的输出
        #
        # out 原来的形状是：
        # (batch_size, seq_len, hidden_size)
        #
        # out[:, -1, :] 的含义：
        # :   -> 取所有样本
        # -1  -> 取每个样本的最后一个时间步
        # :   -> 取这个时间步上的全部 hidden_size 维特征
        #
        # 取完后形状变成：
        # (batch_size, hidden_size)
        #
        # 为什么只取最后一个时间步？
        # 因为最后一个时间步已经“看完了整个序列”，
        # 它的隐藏状态包含了前面所有时间步的信息，
        # 所以可以拿来代表整个序列
        out = out[:, -1, :]

        # 把最后一个时间步的 8 维输出送入全连接层
        #
        # 输入形状：
        # (batch_size, 8)
        #
        # 输出形状：
        # (batch_size, 1)
        #
        # 每个样本输出一个预测值，表示模型预测的“序列和”
        out = self.fc(out)

        # 返回预测结果
        return out


# ==============================
# 2. 生成训练数据
#    功能：随机造一批数字序列，并计算它们的真实和
# ==============================
def generate_data(batch_size=16, seq_len=5):
    """
    返回：
    x: 输入序列，形状 (batch_size, seq_len, 1)
    y: 真实标签，形状 (batch_size, 1)

    例如：
    x 可能是
    [
      [[1],[2],[3],[4],[5]],
      [[0],[7],[1],[2],[3]]
    ]

    对应 y 是
    [
      [15],
      [13]
    ]
    """

    # torch.randint(0, 10, ...)
    # 表示随机生成 [0, 10) 之间的整数，也就是 0~9
    #
    # shape = (batch_size, seq_len, 1)
    # 表示：
    #   batch_size 个样本
    #   每个样本 seq_len 个时间步
    #   每个时间步 1 个数字
    #
    # .float() 把整数转成浮点数
    # 因为神经网络一般使用 float 类型计算
    x = torch.randint(0, 10, (batch_size, seq_len, 1)).float()

    # 对每个样本，在时间维度上求和
    #
    # dim=1 表示沿着 seq_len 那一维求和
    #
    # 如果 x.shape = (16, 5, 1)
    # 那么 y.shape = (16, 1)
    #
    # 例如：
    # [[1],[2],[3],[4],[5]] -> [15]
    y = x.sum(dim=1)

    return x, y


# ==============================
# 3. 创建模型、损失函数、优化器
# ==============================

# 实例化模型
model = SumRNN()

# 定义损失函数：均方误差损失 MSELoss
#
# 为什么用 MSELoss？
# 因为这是一个回归任务：
# 输入一个序列，输出一个连续数值（和）
#
# 不是分类任务，所以不用交叉熵
criterion = nn.MSELoss()

# 定义优化器 Adam
#
# model.parameters():
#   取出模型中所有需要训练的参数
#
# lr=0.01:
#   学习率，表示每次参数更新的步长
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ==============================
# 4. 训练模型
# ==============================

# 训练 200 轮
for epoch in range(200):
    # 每一轮都随机生成一批训练数据
    x, y = generate_data()

    # 前向传播
    #
    # pred.shape = (batch_size, 1)
    # 表示模型对每个样本预测一个“和”
    pred = model(x)

    # 计算损失
    #
    # 比较 pred 和真实标签 y 的差距
    # 差距越小，loss 越小，说明模型学得越好
    loss = criterion(pred, y)

    # 梯度清零
    #
    # PyTorch 默认会把梯度累加，
    # 所以每次反向传播前都要先清零
    optimizer.zero_grad()

    # 反向传播
    #
    # 根据 loss 计算每个参数的梯度
    loss.backward()

    # 更新参数
    #
    # 优化器根据梯度调整模型参数，
    # 让下次预测更接近真实值
    optimizer.step()

    # 每训练 20 轮打印一次损失
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ==============================
# 5. 测试模型
# ==============================

# 手动构造一个测试样本
#
# 这里是 1 个样本，序列长度为 5，每个时间步 1 个数字
# 所以形状是 (1, 5, 1)
#
# 序列内容：[1, 2, 3, 4, 5]
test_x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

# 用模型进行预测
pred = model(test_x)

# 打印输入序列
#
# squeeze() 用来去掉多余的维度
# 原来 test_x.shape = (1, 5, 1)
# squeeze() 后更方便显示成 [1,2,3,4,5]
print("输入:", test_x.squeeze().tolist())

# 打印模型预测结果
print("预测和:", pred.item())

# 打印真实和
print("真实和:", test_x.sum().item())