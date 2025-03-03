import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def random_weight():
    return random.uniform(-1, 1)


class NeuralNetwork:
    def __init__(self, input_data, output_data):
        self.input = input_data
        self.output = output_data
        self.weights1 = [[random_weight() for _ in range(4)] for _ in range(len(input_data[0]))]  
        self.weights2 = [random_weight() for _ in range(4)]  
        self.layer1 = [0] * 4
        self.output_layer = 0

    def feedforward(self):
        
        for i in range(4):
            self.layer1[i] = sigmoid(sum(self.input[j] * self.weights1[j][i] for j in range(len(self.input))))

        
        self.output_layer = sigmoid(sum(self.layer1[i] * self.weights2[i] for i in range(4)))

    def backprop(self):
        
        error = self.output - self.output_layer

        
        d_weights2 = [error * sigmoid_derivative(self.output_layer) * self.layer1[i] for i in range(4)]
        d_weights1 = [[error * sigmoid_derivative(self.output_layer) * self.weights2[i] * sigmoid_derivative(self.layer1[i]) * self.input[j]
                    for i in range(4)] for j in range(len(self.input))]

        
        for i in range(4):
            self.weights2[i] += d_weights2[i]
            for j in range(len(self.input)):
                self.weights1[j][i] += d_weights1[j][i]


input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [0, 1, 1, 0]


nn = NeuralNetwork(input_data, output_data)


for _ in range(10000):
    for i in range(len(input_data)):
        nn.input = input_data[i]  
        nn.output = output_data[i]
        nn.feedforward()
        nn.backprop()


for i in range(len(input_data)):
    nn.input = input_data[i]
    nn.feedforward()
    print(f"Input: {input_data[i]}, Predicted Output: {nn.output_layer}")
