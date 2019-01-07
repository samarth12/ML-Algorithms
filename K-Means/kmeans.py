import csv
import numpy as np
import scipy.io
import random
import sys
from numpy import linalg as LA
from numpy import cov
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt


np.random.seed(10)

#Calculate distance between two data points
def get_dist(data1, data2):
    sum= 0
    for i in range(len(data1)):
        sum = sum + np.square(LA.norm(data1[i]-data2[i]))
    return sum
 
#Function to calulate the object function value
def object_value(data_points, clusters_list):
    start_loss = 0
    end = 1000
    while end != 0:
        curr_loss = 0.0
        cluster_centers = []
        
        for i in range(len(data_points)):
            start_dis = sys.maxsize
            cluster = 0
            for j in range(len(clusters_list)):
                dist = get_dist(data_points[i], clusters_list[j])
                if dist<start_dis:
                    start_dis = dist
                    cluster = j
            #Find the membership of each cluster and add that to the loss/object value
            curr_loss = curr_loss + start_dis
            cluster_centers.append(cluster)
        #If the loss converges/memberships stop changing stop and return the value    
        if (curr_loss - start_loss) ==0:
            return curr_loss

        start_loss = deepcopy(curr_loss)

        count_latest = [0] * len(clusters_list)
        centers_latest = [[0.0] * len(data_points[0]) for _ in range(len(clusters_list))]
        #Update the cluster centers and the memberships of all the clusters
        for i in range(len(cluster_centers)):
            centers_latest[cluster_centers[i]] = np.add(centers_latest[cluster_centers[i]],data_points[i])
            count_latest[cluster_centers[i]] = count_latest[cluster_centers[i]] + 1
          
        for i in range(len(clusters_list)):
            if count_latest[i] == 0:
                continue
            centers_latest[i] = [x / count_latest[i] for x in centers_latest[i]]     
        clusters_list = deepcopy(centers_latest)
    return start_loss


def PCA(data, components):
    data = np.array(data)
    mean = data.mean(axis=0)
    balanced_data = data - mean
    scalar = cov(balanced_data.T)
    
    e_vals, e_vecs = LA.eig(scalar)
    result = []
    for i in range(len(balanced_data)):
        temp = []
        for j in range(components):
            temp.append(np.dot(data[i], e_vecs[j]))
        result.append(temp)
    return result


def main():
   
   #Load the data
    data_set =[]
    with open('audioData.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            row = [float(i) for i in row]
            data_set.append(row)
    
    data_set = np.array(data_set)

    lossend_data = []
    numb =[]
    for k in range(2,11):
        cent= []
        #Randomly select center points from the given data
        center_indices = np.random.randint(0, data_set.shape[0], k)
        for i in range(len(center_indices)):
            cent.append(data_set[center_indices[i]])

        loss_data = object_value(data_set,cent)
        print("Object Value:",loss_data)
        lossend_data.append(loss_data)
        numb.append(k)
   


    plt.plot(numb,lossend_data)
    plt.xlabel("K Value")
    plt.ylabel("Object Value")
    plt.show()


main()


