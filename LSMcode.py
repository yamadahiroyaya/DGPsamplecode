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
        pcd=np.array([pcd_x[:],pcd_y[:]])
        print(self._pcd1)
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
        #plt.contour(x, y, z, [0])
        #plt.scatter(self._pcd_x,self._pcd_y)
        #plt.show()
        n_0=np.array([min_n[0],min_n[1]])
        c_0=np.array(-np.dot(min_n,mean))
        return n_0 ,c_0

    def L1_estimator_IRLS(self,n_0,c_0):
        x_k=np.asarray([self._pcd_x,self._pcd_y])
        for num in range(5):
            e_k=np.asarray(np.dot(n_0,x_k)+c_0)
            w_e_k=1/abs(e_k)
            s=np.sum(w_e_k)
            
            weighted_m=(np.sum(w_e_k*x_k,axis=1)/s).T#アダマール積
        
            weighted_M=np.zeros((2,2))
            
            dev_x=x_k[0]-weighted_m[0]
            dev_y=x_k[1]-weighted_m[1]
        


        
            for num in range(len(dev_x)):
                weighted_M+=w_e_k[num]*np.array([[dev_x[num],dev_y[num]]]).T@np.array([[dev_x[num],dev_y[num]]])
            weighted_M/=s#分散共分散行列の計算
            weight_mu,weight_n=LA.eig(weighted_M)
            weighted_mu=np.min(weight_mu)
            weighted_n=weight_n[:,np.argmin(weight_mu)]
            n_0=weighted_n
            c_0=-np.dot(weighted_n,weighted_m)
            xrange = np.arange(-2, 10, 1)
            yrange = np.arange(-2, 10, 1)
            x, y = np.meshgrid(xrange, yrange)
            z = n_0[0]*x+n_0[1]*y+c_0
            plt.contour(x, y, z, [0])
            plt.scatter(self._pcd_x,self._pcd_y)
            plt.show()




        
pcd_instance=pcd_LSM(pcd1,pcd_x,pcd_y)

n_0,c_0=pcd_instance.LSM()
pcd_instance.L1_estimator_IRLS(n_0,c_0)