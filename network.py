import numpy as np

class Network:
    def __init__(self):
        self.num_layers = 3
        self.sizes = [2, 4, 1]
        self.biases = [
            np.array([-0.5, 0.5, -0.5, 0.5]),
            np.array([0.2, 0.8])
        ]
        self.weights = [
            # T, B, R, L
            np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]),
            np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
        ]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = ReLU(np.dot(w, a) + b)
        return a

    def eval(self, a):
        "Return 1 if close 0 if far"
        p = softmax(self.feedforward(a))
        return np.argmax(p)

def ReLU(a): return (abs(a) + a) / 2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

