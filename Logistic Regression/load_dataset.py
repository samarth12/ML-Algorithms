import os
import struct
import numpy as np
#import matplotlib as mpl
#from matplotlib import pyplot


def read(dataset, path):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    #print("LABELS",lbl)
    #print(np.shape(lbl))
    #print(img)
    #print(np.shape(img))
    #img_res = np.reshape(img, (60000, 784))
    #print(np.shape(img_res))
    #print(np.shape(img_res))
    #for i in range(60000):
        #print(img_res[i])
        #print(len(img_res[i]))
    return (lbl, img)
"""
def show(image):

    Render a given numpy.uint8 2D array of pixel data.
    
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
"""
read("training", "/Users/samarth/Desktop/Fall 2018/CSE 575/Assignment 2/MNIST")