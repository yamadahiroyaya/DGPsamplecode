import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


pcd1=np.loadtxt("original_point_set.txt",dtype='float')
pcd1=np.loadtxt("point_set_with_outliers.txt",dtype='float')
pcd_x=pcd1[0,:]
pcd_y=pcd1[1,:]

class pcd_LSM:
    def __init__(self,pcd1,pcd_x,pcd_y):
        self._pcd1=pcd1
        self._pcd_x=pcd_x
        self._pcd_y=pcd_y

    def scatter(self):
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()

    def LSM(self):
        M=np.cov(self._pcd1,bias=True)
        mu,n=LA.eig(M)
        min_mu=np.min(mu)
        min_n=n[:,np.argmin(mu)]
        mean=[np.mean(self._pcd_x),np.mean(self._pcd_y)]
        
        xrange = np.arange(-2, 10, 1)
        yrange = np.arange(-2, 10, 1)
        x, y = np.meshgrid(xrange, yrange)

        z = min_n[0]*x+min_n[1]*y-np.dot(min_n,mean)
        # z = 1 の等高線を描く
        plt.contour(x, y, z, [0])
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()

        #print(np.dot(min_n,np.array[x,y]))
        #plt.contour(x,y,z,[0])
        #plt.show()

        
pcd_instance=pcd_LSM(pcd1,pcd_x,pcd_y)

pcd_instance.LSM()