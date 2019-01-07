import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import cov

np.random.seed(10)

#Function to initialize variables for gmm
def initialize(data,k):
    cov = []
    mean = []
    n, _ = data.shape
    #initialize the responsibilites equally
    prob= np.array([0.5, 0.5])

    #Split the data
    data_div = [data[i:i + int(np.floor(n / k))] for i in range(0, n, int(np.floor(n / k)))]
    #Initialize the covariances
    for i in range(k):
        cov.append(np.cov(data_div[i].T))
    cov = np.array(cov)
    #Initialize the means
    for i in range(k):
        mean.append(np.mean(data_div[i], axis=0))
    mean = np.array(mean)

    return cov, mean, prob


#Calculate the PDF of the data points for the 2 clusters respectively
def cal_pdf(data, mean, cov, x=2):
    diff = data - mean
    numer = np.dot(np.dot(diff.T,  np.linalg.inv(cov)), diff)
    p = np.exp(-0.5 * numer) / np.sqrt(np.linalg.det(cov) * np.power(2 * np.pi, x))
    return p


#Function to perform EM
def EM(data,cov,mean,prob,k):
    n,_ = data.shape
    log_likelihood = 0
    diff = 1000
    pdf = np.empty((n, k), np.float64)
    resp = np.empty((n, k), np.float64)
    likelihood = []
    while diff !=0:
        for i in range(n):
            row = data[i]
            for j in range(k):
                pdf[i][j] = cal_pdf(row, mean[j], cov[j], k)

        log = np.log(np.array([np.dot(prob.T, pdf[i]) for i in np.arange(n)]))
        updated_likelihood = np.dot(log.T, np.ones(n))

        likelihood.append(updated_likelihood)

        if abs(updated_likelihood - log_likelihood) <0.01:
            return resp

       
        log_likelihood = updated_likelihood
        #E step
        for i in range(n):
            denom = np.dot(prob.T, pdf[i])
            for j in range(k):
                numer = prob[j] * pdf[i][j]
                resp[i][j] = float(numer/denom)

        #M Step
        for i in range(k):
            resp_array = (resp.T)[i]
            resp_total = np.sum(resp_array, axis=0)
            mean[i] = np.dot(resp_array.T, data) / resp_total
            mean_data_diff = data - np.tile(mean[i], (n, 1))
            #Update covariance
            cov[i] = np.dot(np.multiply(resp_array.reshape(n, 1), mean_data_diff).T,mean_data_diff) / resp_total
            prob[i] = resp_total/n
    
    return resp

#Assign data points to clusters
def cluster_assignment(data,membership):
    k1 = []
    k2 = []
    n,_ = data.shape
    for i in range(n):
        if membership[i][0] > membership[i][1]:
            k1.append(data[i][:2])
        else:
            k2.append(data[i][:2])
    k1 = np.array(k1)
    k2 = np.array(k2)
    return k1, k2


def PCA(data, components):
    data = np.array(data)
    mean = data.mean(axis=0)
    balanced_data = data - mean
    #Get the covariance matrix
    scalar = cov(balanced_data.T)
    
    #Calculate eigen vectors
    e_vals, e_vecs = LA.eig(scalar)
    result = []
    for i in range(len(balanced_data)):
        temp = []
        for j in range(components):
            #Multiply each eigen vector with the corresponding data point
            temp.append(np.dot(data[i], e_vecs[j]))
        result.append(temp)
    return result

def main():
    data_set =[]
    #Number of clusters
    k=2
    with open('audioData.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            row = [float(i) for i in row]
            data_set.append(row)

    #Applying PCA to the data
    data_set = PCA(data_set,2)
    data_set = np.array(data_set)

    #Initialize covariance, mean and responsibilities
    cov, mean, prob= initialize(data_set,k)
    print("done")
    #Perform the E and M step
    members = EM(data_set,cov,mean, prob,k)
    print("done")

    #Assigning data points to cluster
    c1, c2 = cluster_assignment(data_set, members)


    #Plot
    f = plt.figure()
    scatter_plot = f.add_subplot(111)
    scatter_plot.scatter(c1.T[0], c1.T[1], c="red", label="First cluster")
    scatter_plot.scatter(c2.T[0], c2.T[1], c="green", label="Second cluster")
    plt.legend(loc='best')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

main()
