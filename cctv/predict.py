import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import load
import os
import tensorflow as tf
from tensorflow.python.framework import ops


def read_image(fname):
 image = np.array(ndimage.imread(fname, flatten=False))
 my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
 plt.imshow(my_image.reshape(64,64,3))
 plt.show()
 return my_image

count=1
error=0

for (dirpath, dirnames, filenames) in os.walk("test"):
 for dirname in dirnames:
     for (dirpath, dirnames, filenames) in os.walk( "test/" + str(dirname)):
         for filename in filenames:
            image=read_image("test/"+str(dirname)+"/"+str(filename))
            normalized_image=np.array(image/255.)



            a1 = tf.placeholder(tf.float32,[25,None], name = 'a1')
            a2 = tf.placeholder(tf.float32,[12,None], name = 'a2')
            a3 = tf.placeholder(tf.float32,[1,None], name = 'a3')
            init = tf.global_variables_initializer()


            with load('parameter.npz') as para:
             b1=np.array(para['b1'])
             b2=np.array(para['b2'])
             b3=np.array(para['b3'])
             W1=np.array(para['W1'])
             W2 =np.array(para['W2'])
             W3 =np.array(para['W3'])
            with tf.Session() as sess:
                sess.run(init)
                Z1 = np.dot(W1, normalized_image) + b1
                A1 = sess.run(tf.nn.relu(a1), feed_dict={a1: Z1})

                Z2 = np.dot(W2, A1) + b2
                A2 = sess.run(tf.nn.relu(a2), feed_dict={a2: Z2})

                Z3 = np.dot(W3, A2) + b3
                A3 = sess.run(tf.nn.sigmoid(a3), feed_dict={a3: Z3})

                if int(dirname) != np.round(Z3):

                    error = error + 1
                else:
                    pass

                print(np.round(Z3))
                print(count)
                count = count + 1

 calcerror = (error / count) * 100
 print(calcerror)








