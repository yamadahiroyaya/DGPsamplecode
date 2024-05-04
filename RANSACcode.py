import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.linalg as LA

pcd1=np.loadtxt("point_set_with_outliers.txt",dtype='float')
pcd_x=pcd1[0,:]
pcd_y=pcd1[1,:]

class RANSAC:
    def __init__(self,pcd1,pcd_x,pcd_y,M_min,p,s,w,threshold,Kmax):
        self._Kmax=Kmax
        self._pcd1=pcd1
        self._pcd_x=pcd_x
        self._pcd_y=pcd_y
        self._M_min=M_min
        self._p=p
        self._s=s
        self._w=w
        self.threshold=threshold
        

    def scatter(self):
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()

    def random_sample(self):
        sample=random.sample(range(len(self._pcd_x)),k=self._s)
        self.sample_x=np.array(self._pcd_x[sample[:]])
        self.sample_y=np.array(self._pcd_y[sample[:]])
        self._sample=np.array([self.sample_x,self.sample_y])
       
    def LSM(self):
        self.sample_x
        
        self._n=np.array([self.sample_y[0]-self.sample_y[1],self.sample_x[1]-self.sample_x[0]])#直線方程式から
        self._c=np.array(self.sample_y[0]*(self.sample_x[0]-self.sample_x[1])-self.sample_x[0]*(self.sample_y[0]-self.sample_y[1]))
    
    def Error_Calculate(self):
        x_k=np.asarray([self._pcd_x,self._pcd_y])
        e_k=np.asarray(np.dot(self._n,x_k)+self._c)
        count=sum([i<self.threshold for i in abs(e_k)])
        print(count)
        return count
    
    def Loss_Calculate(self):
        x_k=np.asarray([self._pcd_x,self._pcd_y])
        Loss_e_k=np.asarray((np.dot(self._n,x_k)+self._c))
        Loss_e_k=np.where(abs(Loss_e_k)<self.threshold,(Loss_e_k)**2,(self.threshold)**2)
        print(Loss_e_k)
        Loss=np.sum(Loss_e_k)
        return Loss
        
       
    
    def Simple_RANSAC(self):
        for num in range(self._Kmax):
            self.random_sample()
            self.LSM()
            count=self.Error_Calculate()
            if(count>self._M_min):
                break
        xrange = np.arange(-2, 10, 1)
        yrange = np.arange(-2, 10, 1)
        x, y = np.meshgrid(xrange, yrange)
        z = self._n[0]*x+self._n[1]*y+self._c
        plt.contour(x, y, z, [0])
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()

    def MSAC(self):
        Pre_Loss=float('inf')
        n=np.array([0,0])
        c=0
        for num in range(self._Kmax):
            self.random_sample()
            self.LSM()
            Current_Loss=self.Loss_Calculate()
            if(Current_Loss<Pre_Loss):
                Pre_Loss=Current_Loss
                n=self._n
                c=self._c
        
        xrange = np.arange(-2, 10, 1)
        yrange = np.arange(-2, 10, 1)
        x, y = np.meshgrid(xrange, yrange)
        z = n[0]*x+n[1]*y+c
        plt.contour(x, y, z, [0])
        plt.scatter(self._pcd_x,self._pcd_y)
        plt.show()
        




RANSAC_instance=RANSAC(pcd1,pcd_x,pcd_y,10,0.95,2,0.9,0.5,3)
#RANSAC_instance.Simple_RANSAC()
RANSAC_instance.MSAC()
