import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def print_loss(features, labels, weights, num_steps):
    loss = cost_function(features, labels, weights, num_steps)
    for i in range(10):
        print("[" + str(i) + "]: " + str(loss[i]))
        print(weights[i])
    print("\n")

def linear_regression(feature, weight):
    bias = 1e-7
    lr = [0.0] * 10
    for i in range(10):
        lr[i] = np.dot(weight[i].transpose(), feature) + bias
    return lr

def cost_function(features, labels, weights, num_steps):
    loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    loss_n = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(num_steps):
        loss_n[labels[i]] += 1
        lr = linear_regression(features[i], weights)
        loss[labels[i]] += np.log(1 + (np.e ** (-1 * lr[labels[i]] * np.max(lr))))
    loss = np.true_divide(loss, loss_n)
    return loss

def calc_gradient(features, labels, weights, num_steps):
    gradient = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gradient_n = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(num_steps):
        gradient_n[labels[i]] += 1
        lr = linear_regression(features[i], weights)
        gradient[labels[i]] += (lr[labels[i]] * np.max(lr)) / (1 + np.e ** (lr[labels[i]] * np.max(lr)))
    gradient = np.true_divide(gradient, gradient_n)
    gradient *= -1
    return gradient

def logistic_regression_PB(features, labels, num_steps, learning_rate):
    weights = [np.zeros(features.shape[1])] * 10
    for step in range(num_steps):
        gradient = calc_gradient(features, labels, weights, num_steps)
        avg_gradient = np.mean(gradient, axis=0)
        weights[labels[step]] = weights[labels[step]] - learning_rate * avg_gradient
        if step % 1000 == 0:
            print_loss(features, labels, weights, num_steps)
    return weights


training_data = unpickle("cifar-10-batches-py/data_batch_1")
meta_data = list(unpickle("cifar-10-batches-py/batches.meta")[b'label_names'])

# print(training_data.keys())
weights = logistic_regression_PB(training_data[b'data'], training_data[b'labels'], 5000, 0.1)
print(weights)