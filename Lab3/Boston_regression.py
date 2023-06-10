import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
min_samples_leaf = 7
# 从CSV文件读取数据
df = pd.read_csv('Lab3/housing.csv')
df = df.dropna()  # 删除空值所在的行
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, random_state=42)
tree_root = build_CART_tree(X_train, y_train, min_samples_leaf)

# 在测试集上进行预测，并计算均方误差
y_pred = np.zeros(len(y_test))
for i in range(len(y_test)):
    y_pred[i] = predict(tree_root, X_test[i])
mse_test = mse(y_test - y_pred)
print("均方误差: {}".format(mse_test))

# 使用模型进行预测
y_pred_train = np.zeros(len(y_train))
y_pred_test = np.zeros(len(y_test))
for i in range(len(y_train)):
    y_pred_train[i] = predict(tree_root, X_train[i])  
for i in range(len(y_test)):
    y_pred_test[i] = predict(tree_root, X_test[i])

# 绘制训练集和测试集的真实值和预测值对比图
plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred_train, label='Training set')
plt.scatter(y_test, y_pred_test, label='Testing set')
plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], 'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted values")
plt.legend(loc="upper left")
plt.show()