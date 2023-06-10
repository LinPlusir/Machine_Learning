import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def load_data():
    iris = load_iris()
    X = iris.data[:,:2]# 只选取前两个特征维度以方便可视化
    y = iris.target
    return X, y

# 对数据集按照类别进行切分
X,y=load_data()#加载数据集
X_setosa = X[y == 0]
X_versicolor = X[y == 1]
X_virginica = X[y == 2]

#绘制数据
def plot_data(X_setosa, X_versicolor, X_virginica):
    plt.scatter(X_setosa[:, 0], X_setosa[:, 1], marker='o', label='setosa')
    plt.scatter(X_versicolor[:, 0], X_versicolor[:, 1], marker='x', label='versicolor')
    plt.scatter(X_virginica[:, 0], X_virginica[:, 1], marker='*', label='virginica')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Logistic_Regression')
    plt.legend()
    plt.show()

plot_data(X_setosa[:, :2], X_versicolor[:, :2], X_virginica[:, :2])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegression:
    def __init__(self, alpha=0.001, num_iter=100000):
        self.alpha = alpha    # 学习率
        self.num_iter = num_iter  # 迭代次数

    def fit(self, X, y):
        # 在特征向量中添加一列1，以便进行截距的训练
        X = np.hstack([np.ones([X.shape[0], 1]), X])   # 在鸢尾花数据集中为[1, X1, X2, X3, X4]
        self.coef_ = np.zeros(X.shape[1])   # 初始化权重

        # 梯度下降法
        for i in range(self.num_iter):
            z = np.dot(X, self.coef_)
            h = sigmoid(z)
            grad = np.dot(X.T, (h - y)) / y.size
            self.coef_ -= self.alpha * grad

    def predict(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        z = np.dot(X, self.coef_)
        h = sigmoid(z)
        return np.round(h)

classifier = LogisticRegression()
X_train = np.vstack([X_setosa[:25], X_versicolor[:25], X_virginica[:25]])
y_train = np.array([0]*25 + [1]*25 + [2]*25)
classifier.fit(X_train[:, :2], (y_train == 2).astype(int))


# 输出模型在训练集上的准确率
y_pred = classifier.predict(X_train[:, :2])
accuracy = np.mean(y_train == 2 * y_pred)
print("Accuracy:", accuracy)