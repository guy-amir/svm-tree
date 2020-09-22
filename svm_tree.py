import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

def svm_tree_init(X,y,depth=3):
    Xy_n = (X,y)
    nt = svt(Xy_n,depth)
    return nt

class svt:
    def __init__(self,Xy_n,depth):
        self.Xy_n = Xy_n
        self.depth=depth
        self.nodes = []
        self.root = node(self,None,Xy_n,1,0)
        self.sort()
        
    def sort(self):
        a = [node.number for node in self.nodes]
        sortkey = list(np.argsort(a))
        self.nodes = [self.nodes[int(i)] for i in sortkey]

    def output_weights(self):
        weights = [node.weights() for node in self.nodes]
        return weights
        
    def predict(self,X):
        svms = [node.svm for node in self.nodes]
        splits = []
        
        for node in self.nodes:
            # if not node.dummy:
            splits.append(ALT_split_testset(X,node.svm))
            # else:
            #     svm = node.parent.svm
            #     while (svm == None):
            #         p_node = node.parent
            #         svm = node.parent.svm
            #     splits.append(ALT_split_testset(X,svm))
        a = np.array(splits)
        y_pred = np.zeros(np.shape(a)[1])
        for ii in range(np.shape(a)[1]):
            b = a[:,ii]
            j=1
            for kk in range(self.depth):
                if b[j-1]:
                    j=2*j+1
                else:
                    j=2*j
            y_pred[ii] = self.nodes[j-1].value
        return y_pred

    def accuracy(self,X,y):
        y_pred = self.predict(X)
        y_pred_int = (y_pred>0.5)
        y_pred_int.astype(int)
        return accuracy_score(y, y_pred_int)

    def plot(self):
        
        fig, sub = plt.subplots(1,1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        X = self.Xy_n[0]
        y = self.Xy_n[1]
        ax = sub

        X0, X1 = X[:,0], X[:,1]
        xx, yy = make_meshgrid(X0, X1)
        
        for node in self.nodes:
            
            w = node.weights()
            # # ###########################################################################
            a = -w[0] / w[1]
            XX = np.linspace(-5, 5)
            YY = a * XX - w[2] / w[1]
            ax.plot(XX, YY, "-")
            # # ###########################################################################
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        plt.show()

    def plot_contours(self, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """

        fig, sub = plt.subplots(1,1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        X = self.Xy_n[0]
        y = self.Xy_n[1]
        ax = sub

        X0, X1 = X[:,0], X[:,1]
        xx, yy = make_meshgrid(X0, X1)

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, **params)

        for node in self.nodes:
            
            w = node.weights()
            # # ###########################################################################
            a = -w[0] / w[1]
            XX = np.linspace(-5, 5)
            YY = a * XX - w[2] / w[1]
            ax.plot(XX, YY, "-")
            # # ###########################################################################
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        plt.show()


class node:
    def __init__(self,tree,parent,Xy_n,node_number,level,dummy = False):
        
        if parent==None:
            self.root = True
        else:
            self.root = False
            
        self.Xy_n = Xy_n

        self.number = node_number
        self.level = level
        self.tree = tree
        self.tree.nodes.append(self)
        self.parent = parent
        self.dummy = dummy
        
        
        #define node value
        self.value = np.mean(Xy_n[1])
        if np.isnan(self.value):
            self.dummy = True
            self.svm = self.parent.svm
        #parameterize svm
        if len(Xy_n[1]) and np.std(Xy_n[1]):
            self.svm = find_proper_svm(Xy_n)
        else:
            self.svm = self.parent.svm
            self.dummy = True

        # 
        if level<tree.depth:
            self.leaf = False
            if self.dummy == False:
                Xy_n_l,  Xy_n_r = ALT_split_dataset(Xy_n,self.svm)
                self.l = node(tree,self,Xy_n_l,2*node_number,level+1)
                self.r = node(tree,self,Xy_n_r,2*node_number+1,level+1)
            else:
                self.l = node(tree,self,Xy_n,2*node_number,level+1)
                self.r = node(tree,self,Xy_n,2*node_number+1,level+1)

        else:
            self.leaf=True
        # else:
        #     if level<tree.depth:
        #         self.svm = self.parent.svm
        #         self.l = node(tree,self,Xy_n,2*node_number,level+1,False)
        #         self.r = node(tree,self,Xy_n,2*node_number+1,level+1,False)
        #         self.leaf=False
        #     else:
        #         self.leaf=True

    def weights(self):
        return np.array([self.svm.coef_[0][0],self.svm.coef_[0][1],self.svm.intercept_[0]])



def ALT_split_dataset(linearly_seperable, svm):

    X = linearly_seperable[0]
    y = linearly_seperable[1]

    predictions = svm.predict(X)

    y_left = y[predictions==0]
    X_left = X[predictions==0]
    y_right = y[predictions==1]
    X_right = X[predictions==1]
    linearly_seperable_left = (X_left,y_left)
    linearly_seperable_right = (X_right,y_right)
    return linearly_seperable_left,  linearly_seperable_right

def split_dataset(X,y,w):
    a = -w[0] / w[1]
    Xsvm = a * X[:,0] - w[2] / w[1]
    div = np.zeros(len(y))
    div[X[:,1]>Xsvm] = 1
    div = np.transpose(np.reshape(div,(-1,1)))
    sep = div[0]
    y_left = y[sep==0]
    X_left = X[sep==0]
    y_right = y[sep==1]
    X_right = X[sep==1]
    return X_left, y_left,  X_right,  y_right


def split_testset(X,w):
    a = -w[0] / w[1]
    Xsvm = a * X[:,0] - w[2] / w[1]
    div = np.zeros(len(Xsvm))
    div[X[:,1]>Xsvm] = 1
    div = np.transpose(np.reshape(div,(-1,1)))
    sep = div[0]
    return sep

def ALT_split_testset(X,svm):
    predictions = svm.predict(X)
    return predictions

def params_to_coef(X,y):
    svm = find_proper_svm(X,y)
    w = np.array([svm.coef_[0][0],svm.coef_[0][1],svm.intercept_[0]])
    return w

def ALT_params_to_coef(svm):
    w = np.array([svm.coef_[0][0],svm.coef_[0][1],svm.intercept_[0]])
    return w    

def find_proper_svm(Xy):
    X = Xy[0]
    y = Xy[1]
    Xs,ys,var = find_nn(X,y)
    combined_score = []
    clf = LinearSVC(random_state=0, tol=1e-5)
    for i in range(len(ys)):
        if np.std(ys[i]):
            clf.fit(Xs[i], ys[i])
            combined_score.append(clf.score(Xs[i], ys[i])*var[i])
            am = np.argmax(combined_score)
            return clf.fit(Xs[am], ys[am])
        else:
            return clf.fit(X, y)

##V
def find_nn(X,y,n_neighbors=50,n_most_different=50):
    if len(y)<n_neighbors:
        n_neighbors = len(y)
    if len(y)<n_most_different:
        n_most_different = len(y)
        
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    ys = y[indices]
    Xs = X[indices]
    most_different_nn = np.argsort(-np.var(ys,1))[:n_most_different]
    most_different_var = -np.sort(-np.var(ys,1))[:n_most_different]
    return Xs[most_different_nn],ys[most_different_nn],most_different_var

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    margin = 0.1
    x_min, x_max = x.min() - margin, x.max() + margin
    y_min, y_max = y.min() - margin, y.max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy