import numpy as np
import matplotlib.pyplot as plt
import load_dataset as ld
import scipy.sparse

def initialize_weighted_vector(dimensions=1, class_count=1):
    w = np.ones(shape=(dimensions, class_count))
    print("Initialized weights:",np.shape(w))
    return w

def create_onehot_label(Y):
    a = Y.shape[0]
    y_sparse = scipy.sparse.csr_matrix((np.ones(a), (Y, np.array(range(a)))))
    y_sparse = np.array(y_sparse.todense()).T
    print("ONE HOT LABELS:",np.shape(y_sparse))
    return y_sparse


def calculate_probability(z):
    py = (np.exp(z).T / (1 + np.sum(np.exp(z), axis=1))).T
    print("Probability:", np.shape(py))
    return py

def fit(w, x, y):
    m = x.shape[0]
    y_label = create_onehot_label(y)
    z = np.dot(x, w)
    print("Calculated WX:",np.shape(z))
    posterior_probability = calculate_probability(z)
    print("Calculated Z:", np.shape(posterior_probability))
    gradient = np.dot(x.T, (y_label - posterior_probability))
    print("Calculated Gradient:", np.shape(gradient))
    return gradient


def test_accuracy(w):
    test_label, test_img = ld.read(dataset="testing", path="MNIST/")
    test_img_vector = test_img.reshape(test_img.shape[0], -1) / 255.0
    print("Test image:", np.shape(test_img_vector))
    posterior_probability = calculate_probability(np.dot(test_img_vector, w))
    prediction = np.argmax(posterior_probability, axis=1)
    print("Prediction:",np.shape(prediction))
    correct_count = sum(prediction == test_label)
    total_count = float(len(test_label))
    accuracy = correct_count / total_count
    return accuracy


def init():
    train_label, train_img = ld.read(dataset = "training", path="MNIST/")

    #Converting the input images as 1D vectors and regularizing it
    train_img_vector = train_img.reshape(train_img.shape[0], -1) / 255.0
    #print(train_img_vector)
    #print(len(train_label))
    #print(train_img_vector.shape[1])

    w = initialize_weighted_vector(train_img_vector.shape[1], len(np.unique(train_label)))
    print("Weights", np.shape(w))
    print("Imageset", np.shape(train_img_vector))

    step = 1e-4
    #for every record, updating the value of weight (1 epoch only)
    accuracies = []
    epoch_count = 5
    for i in range(0, epoch_count):
        print("epoch:", i)
        grad = fit(w, train_img_vector, train_label)
        w = w + grad * step
        print(test_accuracy(w))
        accuracies.append(test_accuracy(w) * 100)

    plt.plot(accuracies)
    plt.legend(["Accuracy"])

    plt.show()

init()
