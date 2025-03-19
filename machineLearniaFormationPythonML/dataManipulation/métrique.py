import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression


y = np.array([1,2,6,5000])
y_pred = np.array([4,2,3,3])

#MSE si on veut pénaliser les erreurs quand elles sont de plus en plus grandes
#MAE si on veut pénaliser une erreur linéairement
#Median AE pour repérer si il n'y a pas une valeur aberrante : fait la median des erreurs

print('MAE : ',mean_absolute_error(y,y_pred))
print('MSE : ',mean_squared_error(y,y_pred))
print('RMSE : ',np.sqrt(mean_squared_error(y,y_pred)))
print('MedianAE : ',median_absolute_error(y,y_pred))

boston = load_boston()
X = boston.data
y = boston.target
model = LinearRegression()
model.fit(X,y)
# le score est le ceoff de détermination R2 = 1-[MSE(y-ypred)/(y-mean(y))]
print(model.score(X,y))









