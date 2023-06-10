import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
df = pd.read_csv('watermelon3.0a.csv')

# 将标签转换为0和1
df['好瓜'] = df['好瓜'].map({'是': 1, '否': 0})

# 将数据集中的特征和标签进行分离
X = df.values[:,1:-1]
y = df.values[:, -1]

# 定义逻辑回归模型
class LogisticRegression:
    def __init__(self):
        self.w = None
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def train(self, X, y, learning_rate=0.1, num_iterations=1000):
        m, n = X.shape
        self.w = np.zeros((n, 1))
        
        for i in range(num_iterations):
            Z = X.dot(self.w)
            A = self.sigmoid(Z)
            dz = A - y.reshape(-1, 1)
            dw = X.T.dot(dz) / m
            self.w -= learning_rate * dw
        
        return self.w
    
    def predict(self, X, threshold=0.5):
        Z = X.dot(self.w)
        y_pred = np.int32(self.sigmoid(Z) > threshold)
        return y_pred.flatten()
        
# 将数据集划分为训练集和测试集
X_train = np.vstack((X[0:6], X[9:]))
y_train = np.hstack((y[0:6], y[9:]))
X_test = X[6:9]
y_test = y[6:9]

# 训练和测试模型
model = LogisticRegression()
w = model.train(X_train, y_train)
y_pred = model.predict(X_test)

# 可视化分界线和样本点
plt.scatter(X[:, 0], X[:, 1], c=y)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
X_grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X_grid)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k')
plt.xlabel('density')
plt.ylabel('sugar_content')
plt.title('Logistic_Regression')
plt.show()
accuracy = np.mean(np.int32(y_pred == y_test))
print("Test accuracy: ", accuracy)
