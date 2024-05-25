import numpy as np
import matplotlib.pyplot as plt

# 定义XOR函数
def xor(x1, x2):
    return (x1 and not x2) or (not x1 and x2)

# 定义前馈神经网络
class FeedforwardNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.num_hidden, self.num_inputs) * 0.1
        self.b1 = np.zeros((self.num_hidden, 1))
        self.W2 = np.random.randn(self.num_outputs, self.num_hidden) * 0.1
        self.b2 = np.zeros((self.num_outputs, 1))
        
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU激活函数
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def train(self, X, y, epochs, learning_rate):
        # 训练网络
        for epoch in range(epochs):
            # 前向传播
            a2 = self.forward(X)
            
            # 计算误差
            error = y - a2
            
            # 反向传播和参数更新
            delta2 = error * self.sigmoid_derivative(a2)
            delta1 = np.dot(self.W2.T, delta2) * self.relu_derivative(self.a1)
            
            self.W2 += learning_rate * np.dot(delta2, self.a1.T)
            self.b2 += learning_rate * np.sum(delta2, axis=1, keepdims=True)
            self.W1 += learning_rate * np.dot(delta1, X.T)
            self.b1 += learning_rate * np.sum(delta1, axis=1, keepdims=True)
            
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

# 测试
network = FeedforwardNetwork(2, 2, 1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[xor(x1, x2)] for x1, x2 in X])
network.train(X.T, y.T, epochs=10000, learning_rate=0.1)

# 预测
print("Predictions:")
for x1, x2 in X:
    print(f"({x1}, {x2}) = {network.forward([[x1], [x2]])[0,0]:.2f}")