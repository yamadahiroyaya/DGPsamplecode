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

    def L1_estimator_IRLS(self):
        n_0=np.asarray([-1,1])
        c_0=0
        x_k=np.asarray([self._pcd_x,self._pcd_y])
        #print(x_k.shape)
        e_k=np.asarray([np.dot(n_0,x_k)+c_0])
        w_e_k=1/abs(e_k)
        print(w_e_k.shape)
        s=np.sum(w_e_k)
        #print(s)
        
        weighted_m=(np.sum(w_e_k*x_k,axis=1)/s).T#アダマール積
     
        weighted_M=np.zeros((2,2))
        
        dev_x=x_k[0]-weighted_m[0]
        dev_y=x_k[1]-weighted_m[1]
       
        for num in range(len(dev_x)):
            weighted_M+=w_e_k[0][num]*np.array([[dev_x[num],dev_y[num]]]).T@np.array([[dev_x[num],dev_y[num]]])
            #print(np.array([[dev_x[num],dev_y[num]]]).T@np.array([[dev_x[num],dev_y[num]]]))
        weighted_M/=s
        print(weighted_M)
        #weighted_mu,weighted_n=





        
pcd_instance=pcd_LSM(pcd1,pcd_x,pcd_y)

#pcd_instance.LSM()
pcd_instance.L1_estimator_IRLS()