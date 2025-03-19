import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

titanic = sns.load_dataset('titanic')
titanic = titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0,inplace=True)
titanic['sex'].replace(['male','female'],[0,1],inplace=True)
print(titanic.head())

y = titanic['survived']
X = titanic.drop('survived',axis=1)

classfier = KNeighborsClassifier(n_neighbors=5)
classfier.fit(X,y)
print(classfier.score(X,y))

Xpred = np.array([[3,0,21]])
ypred=classfier.predict_proba(Xpred)
print(ypred) #80% de chance de mort et 20% de chance de survie selon le modèle


