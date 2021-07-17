import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def read_image(fname):
 image = np.array(ndimage.imread(fname, flatten=False))
 my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
 return my_image


def making_dataset(folder_name):                                   #give a folder and it will return x which contains already read files dataset
 xdataset=[]                                                       #and y which contains folderlabel
 ydataset=[]
 for (dirpath, dirnames, filenames) in os.walk(folder_name):
  for dirname in dirnames:
   print(dirname)
   for (dirpath, dirnames, filenames) in os.walk(folder_name+'/'+dirname):
    for filename in filenames:
     print(filename)
     image=read_image(folder_name+'/'+str(dirname)+'/'+str(filename))
     xdataset.append(image)
     ydataset.append(dirname)
 x=np.array(xdataset)
 y=np.array(ydataset)
 return x,y

x_train,y_train=making_dataset("x_train")
x_test,y_test=making_dataset("test")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
np.savez("dataset.npz",X_train=x_train, Y_train=y_train,X_test=x_test,Y_test=y_test)




