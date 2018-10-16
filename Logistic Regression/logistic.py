import load_dataset
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt


def prob_calculator(z):
    p = (np.exp(z).T / (1 + np.sum(np.exp(z), axis=1))).T
    #print("Probability:", np.shape(p))
    return p

def model(w, image, label):
    z = np.dot(image, w)
    #print("Calculated WX:",np.shape(z))
    prob = prob_calculator(z)
    #print("Calculated Z:", np.shape(prob))
    gradient = np.dot(image.T, (label - prob))
    #print("Calculated Gradient:", np.shape(gradient))
    return gradient


def test_accuracy(w,test_image, test_label):
    #print("Test image:", np.shape(test_image))
    prob = prob_calculator(np.dot(test_image, w))
    pred = np.argmax(prob, axis=1)
    count = 0
    #print("Prediction:",np.shape(prediction))
    for i in range(len(pred)):
        if (pred[i] == test_label[i]):
            count += 1
    total = float(len(test_label))
    accuracy = count / total
    return accuracy


def main():
    path = "/Users/samarth/Desktop/Fall 2018/CSE 575/Assignment 2/MNIST"
    train_labels, train_images = load_dataset.read("training", path)
    train_images = np.reshape(train_images, (60000, 784)) / 255.0

    #print(len(train_images))
    test_labels, test_images = load_dataset.read("testing", path)
    test_images = np.reshape(test_images, (10000, 784)) / 255.0

    no_classes = 10
    w = np.ones(shape=(train_images.shape[1], no_classes))
    #print("Initialized weights:", np.shape(w))
    #print("Weights", np.shape(w))
    #print("Imageset", np.shape(train_images))

    step = 1e-4
    accuracy_list = []
    iteration = 10

    size = train_labels.shape[0]
    labels_final = scipy.sparse.csr_matrix((np.ones(size), (train_labels, np.array(range(size)))))
    labels_final = np.array(labels_final.todense()).T

    for i in range(0, iteration):
        print("Iteration Number:", i)
        gradient = model(w, train_images, labels_final)
        w = w + gradient * step
        if i%5==0:
            step_accuracy = test_accuracy(w,test_images,test_labels)
            print("Accuracy calculated after "+ str(i) + " interactions", step_accuracy)
            accuracy_list.append(step_accuracy * 100)

    plt.plot(accuracy_list, color='g')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Logisitc Regression Graph")
    plt.show()


if __name__ == '__main__':
    main()
