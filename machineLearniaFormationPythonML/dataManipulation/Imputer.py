import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ENLEVER OU MODIFIER DES VALEURS NON RENSEIGNEES

X = np.array([[10,3],
              [0,4],
              [5,3],
              [np.nan,3]])

# but: modifier ou enlever np.nan
# Plusieurs stratégies de modification: strategy=
# mean  //  median  //  most_frequent  //  constant

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

X = imputer.fit_transform(X)

print(X)

# Si il manque une donnée, KNNImputer va la remplacer par celle qu'elle aurait probablement eu au vu des autres éléments du dataset
# exemple: si il manque l'age d'un personne, KNN va l'estimer en fonction de ses autres data en les comparant avec les autres passagers

X =  np.array([[10,3],
              [0,4],
              [5,2],
              [np.nan,3]])
imputer = KNNImputer(n_neighbors=1)
X = imputer.fit_transform(X)
print(X)

#Pour savoir où il manque des données dans le dataset: MissingIndicator

X =  np.array([[10,3],
              [0,4],
              [5,3],
              [np.nan,3]])

Xmiss = MissingIndicator().fit_transform(X)
print(Xmiss)


# Avoir des infos sur ce qui nous manque tout en transformant nos données en parrallèle:

pipeline = make_union(SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=99),MissingIndicator())
Xnew = pipeline.fit_transform(X)
print(Xnew)

# Optimiser les params de KNNImputer avec GridSearchCV
titanic = sns.load_dataset('titanic')
X = titanic[['pclass','age']]
print(X[X==np.nan])
y = titanic['survived']
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
model = make_pipeline(KNNImputer(missing_values=np.nan),SGDClassifier())
param_grid = {'knnimputer__n_neighbors':np.arange(1,20)}
grid = GridSearchCV(model,param_grid=param_grid,cv=5)
grid.fit(Xtrain,ytrain)
model = grid.best_estimator_



