import numpy as np

import matplotlib.pyplot as plt

pcd1=np.loadtxt("point_set_with_outliers.txt",dtype='float')
pcd_x=pcd1[0,:]
pcd_y=pcd1[1,:]

class RANSAC:
    def __init__(self,pcd1,pcd_x,pcd_y,M_min,p,s,w):
        self._pcd1=pcd1
        self._pcd_x=pcd_x
        self._pcd_y=pcd_y
        self._M_min=M_min
        self._p=p
        self._s=s
        self._w=w

    def scatter(self):
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()

    def random_sample(self):
        =ram
