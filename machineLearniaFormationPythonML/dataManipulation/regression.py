import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression # s'appuie sur une méthode des moindres carrés
from sklearn.linear_model import SGDRegressor # s'appuie sur une descente de gradient stochastique
from sklearn.preprocessing import PolynomialFeatures
# équation normale pour obtention des coeffs directeurs d'une droite


def reg_lin_square_root(X,Y,deg):
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly_features.fit_transform(X) #contient X et X carré tq Y = 0.5*Xpoly[1]+Xpoly[0]+rand+2
    lin_reg = LinearRegression()  # moindrescarrés
    lin_reg.fit(X_poly, Y)
    xmin = X.min()
    xmax = X.max()
    x = np.linspace(xmin, xmax, 1000)
    ylin = float(lin_reg.intercept_)
    for i in range(lin_reg.coef_.shape[1]):
        ylin += lin_reg.coef_[0, i] * np.power(x, i + 1)
    plt.plot(X, Y, '+')
    plt.plot(x, ylin)

def reg_lin_stochastic(X,Y,deg):
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly_features.fit_transform(X)  # contient X et X carré tq Y = 0.5*Xpoly[1]+Xpoly[0]+rand+2
    sdg_reg = SGDRegressor()
    sdg_reg.fit(X_poly, Y.ravel())
    xmin = X.min()
    xmax = X.max()
    x = np.linspace(xmin, xmax, 1000)
    ylin = float(sdg_reg.intercept_)
    for i in range(sdg_reg.coef_.shape[0]):
        ylin += sdg_reg.coef_[i] * np.power(x, i + 1)
    plt.plot(X, Y, '+')
    plt.plot(x, ylin)

if __name__=='__main__':
    np.random.seed(42)
    m = 100# n obs
    X = 2*np.random.rand(m,1)
    X = np.concatenate((X,np.array([[1] for i in range(m)])),axis = 1)
    X = X + np.random.randn(m,1)
    theta = np.array([[3],[4]])
    Y = X.dot(theta)
    A = np.linalg.inv((X.T).dot(X))
    theta0 = (A.dot(X.T)).dot(Y) #solution analytique des moindres carrés
    print(theta0)
    plt.figure()
    plt.plot(X[:,0],Y,'+')
    plt.plot()
    plt.show()


    X = 2*np.random.rand(m,1)
    Y = 4+3*X+np.random.randn(m,1)
    lin_reg = LinearRegression()
    lin_reg.fit(X,Y)
    print(lin_reg.intercept_,lin_reg.coef_)
    plt.figure()
    plt.plot(X,Y,'+')
    plt.show()

    #Régression d'une fonction polynomiale
    ax = plt.figure()



    X = 6*np.random.rand(m,1)-3
    Y = 0.5*X**2+X+2+np.random.randn(m,1)
    reg_lin_square_root(X,Y,40)
    reg_lin_stochastic(X,Y,2)
    plt.ylim((Y.min(),Y.max()))
    plt.show()


