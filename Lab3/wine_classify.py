import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义节点类
class Node:
    def __init__(self, y_values=None, feature_ind=None, feature_val=None, left=None, right=None, is_leaf=True, class_label=None):
        self.y_values = y_values  # 节点包含的样本标签
        self.feature_ind = feature_ind  # 节点划分所用特征的索引
        self.feature_val = feature_val  # 节点划分所用特征的取值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.is_leaf = is_leaf  # 是否为叶节点
        self.class_label = class_label  # 叶节点的类别标签

# 定义决策树分类器类
class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth  # 树的最大深度

    # 训练决策树模型
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))  # 类别数
        self.n_features = X.shape[1]  # 特征数
        self.tree_ = self.build_tree(X, y)  # 构建决策树

    # 决策树预测函数
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.empty((n_samples,), dtype=int)  # 预测结果

        for i, x in enumerate(X):
            node = self.tree_
            while not node.is_leaf:
                if x[node.feature_ind] <= node.feature_val:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.class_label

        return y_pred

    # 计算基尼指数
    def get_gini(self, y):
        # 计算数据集的基尼指数
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        gini = 1 - np.sum(p ** 2)
        
        return gini

    # 计算加权基尼指数
    def get_weighted_gini(self, X_left, X_right, y_left, y_right):
        # 计算划分后的样本基尼指数，并加权平均
        p_left = len(y_left) / (len(y_left) + len(y_right))
        p_right = len(y_right) / (len(y_left) + len(y_right))
        g_left = self.get_gini(y_left)
        g_right = self.get_gini(y_right)
        wgini = p_left * g_left + p_right * g_right
        
        return wgini

    # 对数据进行划分并构建决策树
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # 判断是否达到叶子节点
        if (depth == self.max_depth) or (self.n_classes == 1):
            label = np.argmax(np.bincount(y))  # 返回频数最大的类别
            return Node(y_values=y, is_leaf=True, class_label=label)

        # 在所有特征中找到最佳划分特征和划分值
        best_feature = None
        best_value = None
        best_score = np.inf
        for feature in range(n_features):
            X_feature = X[:, feature]
            vals = np.unique(X_feature)

            # 生成划分方案
            for split_val in vals:
                # 根据划分方案分离出左右两侧数据
                y_left = y[X_feature <= split_val]
                y_right = y[X_feature > split_val]
                if (len(y_left) == 0) or (len(y_right) == 0):
                    continue

                # 计算划分方案的加权基尼指数
                gini_score = self.get_weighted_gini(X[X_feature <= split_val], X[X_feature > split_val], y_left, y_right)

                # 根据加权基尼指数找到最佳划分规则
                if gini_score < best_score:
                    best_feature = feature
                    best_value = split_val
                    best_score = gini_score

        # 判断是否达到叶子节点
        if best_feature is None:
            label = np.argmax(np.bincount(y))  # 返回频数最大的类别
            return Node(y_values=y, is_leaf=True, class_label=label)

        # 根据最佳划分规则分离出左右两侧数据和标签
        X_left = X[X[:, best_feature] <= best_value]
        y_left = y[X[:, best_feature] <= best_value]
        X_right = X[X[:, best_feature] > best_value]
        y_right = y[X[:, best_feature] > best_value]

        # 生成左右子树
        node = Node(y_values=y, feature_ind=best_feature, feature_val=best_value, is_leaf=False)
        node.left = self.build_tree(X_left, y_left, depth+1)
        node.right = self.build_tree(X_right, y_right, depth+1)

        return node


# 从csv文件中读取鸢尾花数据集
data = pd.read_csv('Lab3\wine.csv')

# 将特征和标签拆分开
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=42)

# 使用CART算法进行训练和预测
tree = DecisionTreeClassifier(max_depth=13)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# 计算模型准确率
acc = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", acc)