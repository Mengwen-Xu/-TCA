import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

"""
求解核矩阵K
    ker:求解核函数的方法
    X1:源域数据的特征矩阵
    X2:目标域数据的特征矩阵
    gamma:当核函数方法选择rbf时，的参数
"""
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker=='primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        :param kernel_type:
        :param dim:
        :param lamb:
        :param gamma:
        """
        self.kernel_type = kernel_type  #选用核函数的类型
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        :param Xs: 源域的特征矩阵 （样本数x特征数）
        :param Xt: 目标域的特征矩阵 （样本数x特征数）
        :return: 经过TCA变换后的Xs_new,Xt_new
        """
        X = np.hstack((Xs.T, Xt.T))     #X.T是转置的意思；hstack则是将数据的相同维度数据放在一起
        X = X/np.linalg.norm(X, axis=0)  #求范数默认为l2范数即平方和开方，按列向量处理，这里相当于
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        #构造MMD矩阵 L
        e = np.vstack((1 / ns*np.ones((ns, 1)), -1 / nt*np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        #构造中心矩阵H
        H = np.eye(n) - 1 / n*np.ones((n, n))
        #构造核函数矩阵K
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n

        #注意核函数K就是后边的X特征，只不过用核函数的形式表示了
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye)#XMX_T+lamb*I
        b = np.linalg.multi_dot([K, H, K.T])#XHX_T

        w, V = scipy.linalg.eig(a, b)  #这里求解的是广义特征值，特征向量
        ind = np.argsort(w)#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到ind
        A = V[:, ind[:self.dim]]#取前dim个特征向量得到变换矩阵A，按照特征向量的大小排列好,
        Z = np.dot(A.T, K)#将数据特征*映射A
        Z /= np.linalg.norm(Z, axis=0)#单位向量话
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T#得到源域特征和目标域特征
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt , then make p r edi c tion s on ta rg e t using 1NN
        param Xs : ns ∗ n_feature , source feature
        param Ys : ns ∗ 1 , source label
        param Xt : nt ∗ n_feature , target feature
        param Yt : nt ∗ 1 , target label
        return : Accuracy and predicted_labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)#经过TCA映射
        clf = KNeighborsClassifier(n_neighbors=1) #k近邻分类器，无监督学习
        clf.fit(Xs_new, Ys.ravel())#训练源域数据
        # 然后直接用于目标域的测试
        y_pred = clf.predict(Xs_new)
        acc = sklearn.metrics.accuracy_score(Ys, y_pred)
        print(acc)
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred
# dslr_SURF_L10.mat -> webcam_SURF_L10.mat
# acc: 0.688135593220339
if __name__ == '__main__':
    domains = ['amazon_SURF_L10.mat', 'Caltech10_SURF_L10.mat', 'dslr_SURF_L10.mat', 'webcam_SURF_L10.mat']
    src = 'SURF/' + "Caltech10_SURF_L10.mat"
    tar = 'SURF/' + "amazon_SURF_L10.mat"
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    # 获取数据
    Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']

    tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
    print(src, "->", tar)
    print("acc:", acc)
    #k近邻设置为3
    # 0.6714158504007124
    # SURF / Caltech10_SURF_L10.mat -> SURF / amazon_SURF_L10.mat
    # acc: 0.40292275574112735
    # for i in range(4):
    #     for j in range(4):
    #         if i != j:
    #             src= 'SURF/' + domains[i]
    #             tar = 'SURF/' + domains[j]
    #             src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    #             # 获取数据
    #             Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']
    #
    #             tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    #             acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
    #             print(domains[i], "->", domains[j])
    #             print("acc:",acc)


# 一个分类器的模型，先对源数据进行训练，然后直接用于目标域的测试