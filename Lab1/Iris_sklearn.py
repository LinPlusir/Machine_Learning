import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 加载数据集
iris = load_iris()
X = iris.data[:, :2]  # 取前两个特征以便可视化
y = iris.target

# 模型训练（逻辑回归模型）
clf_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf_lr.fit(X, y)

# 绘制决策边界
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

# 绘制逻辑回归模型的决策边界
plot_decision_boundary(clf_lr, X, y)
plt.title('Logistic Regression Model')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()