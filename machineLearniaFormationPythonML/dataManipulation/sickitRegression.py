import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # s'appuie sur une descente de gradient stochastique
from sklearn.svm import SVR # s'appuie sur une descente de gradient stochastique
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(0)
m = 100
X = np.linspace(0,10,m).reshape(m,1)
y = np.power(X,2) + np.random.randn(m,1)

#linéaire
lin_reg = LinearRegression()
lin_reg.fit(X,y)
score_lin=lin_reg.score(X,y)
predictions_lin = lin_reg.predict(X)

# support vector machine
svm = SVR(C=100)
svm.fit(X, y.ravel())
score_sgd=svm.score(X, y.ravel())
predictions_sgd = svm.predict(X)

#polynomial²
poly = PolynomialFeatures(degree=2)
Xpoly = poly.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(Xpoly,y)
predictions_poly = lin_reg2.predict(Xpoly)

#KNeighbors
kneighbors = KNeighborsRegressor(n_neighbors=5)
kneighbors.fit(X,y)
predictions_kneighbors = kneighbors.predict(X)


plt.plot(X,predictions_lin,c='r')
plt.plot(X,predictions_sgd,'y')
plt.plot(X,predictions_poly,c='g')
plt.plot(X,predictions_kneighbors,c='purple')

plt.scatter(X,y)
plt.show()