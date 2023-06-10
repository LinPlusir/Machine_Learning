import csv
import numpy as np
import matplotlib.pyplot as plt

# 读入数据
dates = []
data = []
with open('Lab3\Tesla.csv') as f:
    reader = csv.reader(f)
    # 解析表头
    header = next(reader)
    # 解析数据
    for row in reader:
        dates.append(row[0])
        data.append(row[1:])

# 转换数据格式
data = np.array(data).astype(float)

# 分割训练集和测试集
n_samples = len(data)
n_train = int(0.8*n_samples)
train_data = data[:n_train, :]
test_data = data[n_train:, :]

# ------------ CART回归树算法 -------------

# 定义节点类
class Node:
    def __init__(self, split_feature, split_value, left_child, right_child, prediction):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.prediction = prediction

# 计算均方误差
def mse(y):
    return np.mean((y - np.mean(y))**2)

# 递归构建CART树
def build_CART_tree(X, y, min_samples_leaf=2, depth=0):
    # 如果样本数达到叶子节点要求的最小值，则直接返回叶子节点的平均值
    if len(y) <= min_samples_leaf:
        return Node(None, None, None, None, np.mean(y))
    
    # 如果树的深度达到最大值，则直接返回叶子节点的平均值
    if depth >= max_depth:
        return Node(None, None, None, None, np.mean(y))
    
    # 寻找最优的分裂特征和分裂阈值
    best_feature, best_value, best_mse = None, None, np.inf
    for i in range(X.shape[1]):
        for value in np.unique(X[:,i]):
            left_indices = X[:,i] <= value
            right_indices = X[:,i] > value
            left_mse = mse(y[left_indices])
            right_mse = mse(y[right_indices])
            mse_sum = left_mse*(sum(left_indices)/len(X)) + right_mse*(sum(right_indices)/len(X))
            if mse_sum < best_mse:
                best_feature, best_value, best_mse = i, value, mse_sum
    
    # 如果没有找到满足分裂条件的节点，则直接返回叶子节点的平均值
    if best_feature is None:
        return Node(None, None, None, None, np.mean(y))
    
    # 根据最优的分裂特征和分裂阈值将数据集分为左右两个子树
    left_indices = X[:,best_feature] <= best_value
    right_indices = X[:,best_feature] > best_value
    left_X, left_y = X[left_indices], y[left_indices]
    right_X, right_y = X[right_indices], y[right_indices]
    
    # 递归构建左右两个子树
    left_child = build_CART_tree(left_X, left_y, min_samples_leaf, depth+1)
    right_child = build_CART_tree(right_X, right_y, min_samples_leaf, depth+1)
    
    # 返回当前节点
    return Node(best_feature, best_value, left_child, right_child, np.mean(y))

# 对新数据进行预测
def predict(node, x):
    if node.split_feature is None:
        return node.prediction
    elif x[node.split_feature] <= node.split_value:
        return predict(node.left_child, x)
    else:
        return predict(node.right_child, x)

# ------------ 执行程序 -------------

# 构建CART树
max_depth = 10
min_samples_leaf = 2
X_train, y_train = train_data[:,:-1], train_data[:,-1]
tree_root = build_CART_tree(X_train, y_train, min_samples_leaf)

# 在测试集上进行预测，并计算均方误差
X_test, y_test = test_data[:,:-1], test_data[:,-1]
y_pred = np.zeros(len(y_test))
for i in range(len(y_test)):
    y_pred[i] = predict(tree_root, X_test[i])
mse_test = mse(y_test - y_pred)
print("均方误差: {}".format(mse_test))

# 在测试集上进行预测
X_test, y_test = test_data[:,:-1], test_data[:,-1]
y_pred = np.zeros(len(y_test))
for i in range(len(y_test)):
    y_pred[i] = predict(tree_root, X_test[i])

# 绘制真实值和预测值比较图
plt.figure(figsize=(8, 8))
plt.scatter(range(len(y_test)), y_test, marker="o", label="True values")
plt.scatter(range(len(y_test)), y_pred, marker="x", label="Predicted values")
plt.xlabel("Test sample")
plt.ylabel("Value")
plt.title("True vs Predicted values")
plt.legend()
plt.show()