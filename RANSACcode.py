import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.linalg as LA
import LSMcode

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
        self.random_sample()
        self.LSM()
        self.Error_Calculate()

    def scatter(self):
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()

    def random_sample(self):
        sample=random.sample(range(len(self._pcd_x)),k=self._s)
        sample_x=np.array(self._pcd_x[sample[:]])
        sample_y=np.array(self._pcd_y[sample[:]])
        self._sample=np.array([sample_x,sample_y])
       
    def LSM(self):
        M=np.cov(self.sample,bias=True)
        mu,n=LA.eig(M)
        min_mu=np.min(mu)
        min_n=n[:,np.argmin(mu)]
        mean=[np.mean(self._pcd_x),np.mean(self._pcd_y)]
        
        self._n=np.array([min_n[0],min_n[1]])
        self._c=np.array(-np.dot(min_n,mean))
    
    def Error_Calculate(self):
        e_k=np.asarray(np.dot(self._n,x_k)+self._c)



RANSAC_instance=RANSAC(pcd1,pcd_x,pcd_y,5,0.95,2,0.9)
RANSAC_instance.random_sample()
