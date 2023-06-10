import pandas as pd
import numpy as np

# 1.初始化参数
def init_parameters(n_x, n_h1,n_h2, n_y):
    np.random.seed(3)

    # 权重和偏置矩阵
    w1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros(shape=(n_h1, 1))
    w2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros(shape=(n_h2, 1))
    w3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros(shape=(n_y, 1))
    # 用字典存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,'w3': w3, 'b3': b3}

    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']

    # 通过前向传播来计算a2
    z1 = np.dot(w1, X) + b1     
    a1 = np.tanh(z1)            # 使用tanh作为第一层的激活函数
    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)            # 使用tanh作为第二层的激活函数
    z3 = np.dot(w3, a2) + b3
    a3 = 1 / (1 + np.exp(-z3))  # 使用sigmoid作为第三层的激活函数

    # 通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2,'z3': z3, 'a3': a3}

    return a3, cache


# 3.计算代价函数
def compute_cost(a3, Y, parameters):
    m = Y.shape[1]      # Y的列数即为总的样本数

    # 采用交叉熵（cross-entropy）作为代价函数
    logprobs = np.multiply(np.log(a3), Y) + np.multiply((1 - Y), np.log(1 - a3))
    cost = - np.sum(logprobs) / m

    return cost


# 4.反向传播
def back_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']
    w3 = parameters['w3']
    a1 = cache['a1']
    a2 = cache['a2']
    a3 = cache['a3']

    dz3 = a3 - Y
    dw3 = (1 / m) * np.dot(dz3, a2.T)
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

    dz2 = np.multiply(np.dot(w3.T, dz3), 1 - np.power(a2, 2))
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2,'dw3': dw3, 'db3': db3}

    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate=0.4):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']
    dw3 = grads['dw3']
    db3 = grads['db3']

    # 更新参数
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate
    w3 = w3 - dw3 * learning_rate
    b3 = b3 - db3 * learning_rate
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,'w3': w3, 'b3': b3}

    return parameters


# 6.模型评估
def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = 1 / (1 + np.exp(-z3))

    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a3[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    print('预测结果：')
    print(output)
    print('真实结果：')
    print(y_test)

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k]:
            count = count + 1
        else:
            print('第 %d 列与真实结果不同' % k)

    acc = count / int(y_test.shape[1]) * 100
    print('准确率：%.2f%%' % acc)
    return output


# 建立神经网络
def nn_model(X, Y, n_h1, n_h2, n_input, n_output, num_iterations=1000, print_cost=False):
    np.random.seed(3)

    n_x = n_input           # 输入层节点数
    n_y = n_output          # 输出层节点数

    # 1.初始化参数
    parameters = init_parameters(n_x, n_h1, n_h2, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2.前向传播
        a3, cache = forward_propagation(X, parameters)
        # 3.计算代价函数
        cost = compute_cost(a3, Y, parameters)
        # 4.反向传播
        grads = back_propagation(parameters, cache, X, Y)
        # 5.更新参数
        parameters = update_parameters(parameters, grads)

        # 每1000次迭代，输出一次代价函数
        if print_cost and i % 100 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))

    return parameters


if __name__ == "__main__":
    # 读取数据
    data_set = pd.read_csv('heart_training.csv', header=None)#相对路径

    X = data_set.iloc[:, 0:13].values.T          # 前十三列是特征，T表示转置
    Y = data_set.iloc[:, 13:].values.T           # 后两列是标签
    Y = Y.astype('uint8')
    
    # 开始训练
    parameters = nn_model(X, Y, n_h1=10, n_h2=5, n_input=13, n_output=2, num_iterations=1000, print_cost=True)

    # 模型测试
    data_test = pd.read_csv('heart_test.csv', header=None)
    x_test = data_test.iloc[:, 0:13].values.T
    y_test = data_test.iloc[:, 13:].values.T
    y_test = y_test.astype('uint8')

    result = predict(parameters, x_test, y_test)