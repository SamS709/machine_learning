import matplotlib.pyplot as plt
from scipy.ndimage import label

from regression import *
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline

m=100
X = 6*np.random.rand(m,1)-3
Y = 0.5*X**2+X+2+np.random.randn(m,1)

def training_curve(X,Y,deg):
    polynomial_regression = make_pipeline(PolynomialFeatures(degree=deg,include_bias=False),LinearRegression())
    train_sizes,train_scores,valid_scores = learning_curve(polynomial_regression,X,Y,train_sizes=np.linspace(0.01,1.0,40),cv=5,scoring='neg_root_mean_squared_error')
    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scores.mean(axis=1)
    plt.plot(train_sizes,train_errors,'r',label='Entrainement')
    plt.plot(train_sizes,valid_errors,'r',label='Validation')
    plt.legend()
    plt.grid()

if __name__=='__main__':
    training_curve(X,Y,2)
    plt.ylim(0,2.5)
    plt.show()


