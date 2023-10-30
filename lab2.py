import numpy as np


class Neuron:
    def __init__(self, num_inputs, learning_rate):
        self.weights = np.random.rand(num_inputs)
        self.learning_rate = learning_rate

    def print_weights(self):
        print(self.weights)

    def forward(self, inputs):
        output = np.dot(inputs, self.weights)
        return output

    def train(self, input_data, target_data, num_epochs):
        for epoch in range(num_epochs):
            for input_sample, target in zip(input_data, target_data):
                output = self.forward(input_sample)
                error = target - output
                self.weights += self.learning_rate * input_sample * error

##x1 + x2 + 2x3
input_data = np.array([[2, 1, 3], [3, 5, 4], [2, 5, 5]])
target_data = np.array([9, 16, 17])

learning_rate = 0.01
num_epochs = 501

neuron = Neuron(3, learning_rate)
print("Веса:")
neuron.print_weights()
neuron.train(input_data, target_data, num_epochs)
print("Новые веса:")
neuron.print_weights()
new_input = [2, 1, 4]
prediction = neuron.forward(new_input)
print("Результат:")
print(prediction)
