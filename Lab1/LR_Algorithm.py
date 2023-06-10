import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, print_cost=False):
        self.learning_rate = learning_rate      #学习率
        self.num_iterations = num_iterations    #迭代次数
        self.print_cost = print_cost
        self.theta = None
    
    #sigmoid函数
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    
    #代价函数和梯度下降算法
    def cost_function(self, X, y, theta):
        m = X.shape[0]
        h = self.sigmoid(X.dot(theta))
        cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))#代价函数
        grad = (1/m) * X.T.dot(h - y)
        return cost, grad
    
    #训练模型，X是训练集的特征矩阵，y是训练集的标签向量
    def fit(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        self.theta = np.zeros((n, 1))
        for i in range(self.num_iterations):
            cost, grad = self.cost_function(X, y, self.theta)
            self.theta = self.theta - self.learning_rate * grad#使用梯度下降算法更新theta
            if self.print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
    
    #对新样本进行分类预测，X是新样本的特征矩阵
    def predict(self, X):
        return np.round(self.sigmoid(X.dot(self.theta)))#返回的结果为预测的标签向量