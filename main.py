import numpy as np
import matplotlib.pyplot as plt

def activation_function(x):
    return 1 / (1 + np.exp(-x))

def derivative_activation_function(x):
    fx = activation_function(x)
    return fx * (1 - fx)

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class FilmClassifier:
    def __init__(self):
        self.weight1 = np.random.normal()
        self.weight2 = np.random.normal()
        self.weight3 = np.random.normal()
        self.weight4 = np.random.normal()
        self.weight5 = np.random.normal()
        self.weight6 = np.random.normal()
        self.weight7 = np.random.normal()
        self.weight8 = np.random.normal()

        self.bias1 = np.random.normal()
        self.bias2 = np.random.normal()
        self.bias3 = np.random.normal()

    def predict(self, x):
        hidden1 = activation_function(self.weight1 * x[0] + self.weight2 * x[1] + self.bias1)
        hidden2 = activation_function(self.weight3 * x[0] + self.weight4 * x[1] + self.bias2)
        output = activation_function(self.weight5 * hidden1 + self.weight6 * hidden2 + self.bias3)
        return output

    def train(self, data, labels):
        learning_rate = 0.005
        epochs = 100

        for epoch in range(epochs):
            for x, label in zip(data, labels):
                input1 = self.weight1 * x[0] + self.weight2 * x[1] + self.bias1
                hidden1 = activation_function(input1)

                input2 = self.weight3 * x[0] + self.weight4 * x[1] + self.bias2
                hidden2 = activation_function(input2)

                input3 = self.weight5 * hidden1 + self.weight6 * hidden2 + self.bias3
                output = activation_function(input3)

                loss_derivative = -2 * (label - output)

                output_derivative_w5 = hidden1 * derivative_activation_function(input3)
                output_derivative_w6 = hidden2 * derivative_activation_function(input3)
                output_derivative_b3 = derivative_activation_function(input3)

                output_derivative_hidden1 = self.weight5 * derivative_activation_function(input3)
                output_derivative_hidden2 = self.weight6 * derivative_activation_function(input3)

                hidden1_derivative_w1 = x[0] * derivative_activation_function(input1)
                hidden1_derivative_w2 = x[1] * derivative_activation_function(input1)
                hidden1_derivative_b1 = derivative_activation_function(input1)

                hidden2_derivative_w3 = x[0] * derivative_activation_function(input2)
                hidden2_derivative_w4 = x[1] * derivative_activation_function(input2)
                hidden2_derivative_b2 = derivative_activation_function(input2)

                self.weight1 -= learning_rate * loss_derivative * output_derivative_hidden1 * hidden1_derivative_w1
                self.weight2 -= learning_rate * loss_derivative * output_derivative_hidden1 * hidden1_derivative_w2
                self.bias1 -= learning_rate * loss_derivative * output_derivative_hidden1 * hidden1_derivative_b1

                self.weight3 -= learning_rate * loss_derivative * output_derivative_hidden2 * hidden2_derivative_w3
                self.weight4 -= learning_rate * loss_derivative * output_derivative_hidden2 * hidden2_derivative_w4
                self.bias2 -= learning_rate * loss_derivative * output_derivative_hidden2 * hidden2_derivative_b2

                self.weight5 -= learning_rate * loss_derivative * output_derivative_w5
                self.weight6 -= learning_rate * loss_derivative * output_derivative_w6
                self.bias3 -= learning_rate * loss_derivative * output_derivative_b3

            if epoch % 10 == 0:
                predictions = np.apply_along_axis(self.predict, 1, data)
                loss = mean_squared_error(labels, predictions)
                print("Epoch %d loss: %.10f" % (epoch, loss))


cucmber_images = np.random.randn(1000, 2) + np.array([0, -5])  ##x [-3, 3] y [-2, -8]
tomato_images = np.random.randn(1000, 2) + np.array([10, 10])   ##x [6, 13] y [6, 13]

feature_set = np.vstack([cucmber_images, tomato_images])

labels = np.array([0]*1000 + [1]*1000)

one_hot_labels = np.zeros((2000, 2))

data = (feature_set)


network = FilmClassifier()
network.train(feature_set, labels)

new_data = [-1, -2]

result = network.predict(new_data)
print("Tomato: {:.5f}".format(result))
print("Cucumber: {:.5f}".format(1 - result))
new_set = np.vstack([feature_set, new_data])
new_labels = np.array([0]*1000 + [1]*1000 + [2]*1)
plt.scatter(new_set[:,0], new_set[:,1], c=new_labels, cmap='plasma', s=100, alpha=0.5)
plt.show()
