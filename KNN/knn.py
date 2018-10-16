import numpy as np
from collections import defaultdict
from sortedcontainers import SortedList
from scipy import linalg as LA
import load_dataset
import pickle
import matplotlib.pyplot as plt


def euclidean_distance(train, test):
    dist = 0
    dist = dist + np.linalg.norm(train-test)
    #print(dist)
    return dist



def find_max(labels):

    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1

    # Finding the majority class.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key



def compute_distance_matrix(k, train_images, train_labels, test_image):


    distances = SortedList()
    #finallist = []

    for i in range(len(train_images)):
        #temp = euclidean_distance(test_image,train_images[i]), train_labels[i])
        dist = euclidean_distance(test_image,train_images[i])
        if len(distances) < k:
            distances.add((dist, train_labels[i]))
        elif dist < distances[-1][0]:
            del distances[-1]
            distances.add((dist, train_labels[i]))

    #print(distances)
    return distances


def test_accuracy(k_list, test_images, test_labels):
    with open('sorted_list_latest', 'rb') as fp:
        distances = pickle.load(fp)
    # print(distances)

    accuracy_list = []

    #total_correct = 0
    for k in k_list:
        # dist1 = distances[:k]
        # print(dist1)
        total_correct = 0
        for i, test_image in enumerate(test_images):
            dist1 = distances[i]
            k_labels = []
            for (dis, label) in dist1[:k]:
                k_labels.append(label)

            pred = find_max(k_labels)
            if pred == test_labels[i]:
                total_correct += 1
        accuracy = (total_correct / len(test_images)) * 100
        accuracy_list.append(accuracy)

    return accuracy, accuracy_list


def main():
    train_labels, train_images = load_dataset.read("training",
                                                   "/Users/samarth/Desktop/Fall 2018/CSE 575/Assignment 2/MNIST")
    train_images = np.reshape(train_images, (60000, 784)) / 255.0
    # train_images = getPCAData(train_images,50)
    # train_images = np.reshape(train_images, (60000, 784))

    print(len(train_images))
    test_labels, test_images = load_dataset.read("testing",
                                                 "/Users/samarth/Desktop/Fall 2018/CSE 575/Assignment 2/MNIST")
    test_images = np.reshape(test_images, (10000, 784)) / 255.0
    k_list = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    i = 0
    final = []
    for test_image in test_images:
        dist = compute_distance_matrix(100, train_images, train_labels, test_image)
        # print(dist)
        # print(pred)
        final.append(dist)

        print('Calculating the matrix for k = 100 for test image[' + str(i) + ']')
        i += 1

    with open('sorted_list_latest', 'wb') as fp:
        pickle.dump(final, fp)

    accuracy, accuracy_list = test_accuracy(k_list, test_images, test_labels)

    plt.plot(k_list, accuracy_list, color='g')
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("KNN Graph")
    plt.show()


main()