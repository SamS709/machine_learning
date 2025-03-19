from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel, RFE, RFECV
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import  load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
y = iris.target
X = iris.data

## VARIANCE

# Calculer la variance d'une variable pour savoir si son choix est pertinent:
# Une variable qui ne varie pas n'a pas ou peu d'interet: c'est une cste

#On voir par exemple ici que sepal wisth ne varie que très peu
plt.plot(X)
plt.legend(iris.feature_names)

print(iris.feature_names)
print(X.var(axis=0)) # sepal width a une variance de 0.18

print()
selector = VarianceThreshold(threshold=0.2) #On selectionne les variables de variance supérieure à 0.2
selector.fit_transform(X)
print('Les variables dont la variance est >=0.2 sont la 3è et la dernière:',selector.get_support())
print("Il s'agit des variables: ",np.array(iris.feature_names)[selector.get_support()==True]) # On ne prend que les features qui ont une variance >=0.2


# TEST DE DEPENDANCE DE y en fonction de chaque donnée de X

# TEST CHI2
print()
print('chi2 :', chi2(X,y)) #renvoie 2 tableaux : le premier contient le score chi2 de chacune des variables de X. Plus il est élevé, plus y dépend de cette variable
# Ici, y dépend le plus de la troisième variable contenue dans X
selector = SelectKBest(chi2,k=2) # selector qui renvoie les deux variables de X ayant le meilleur score chi2
selector.fit_transform(X,y)
print('les 2 variables ayant le meilleur score chi 2 sont:',np.array(iris.feature_names)[selector.get_support()==True])


# On détermine la dépendance de y en fonction des variables de X en simulant un estimateur paramétré. Si les paramètres (ex : poids de reseau de neurones) sont élevés => variable fortmement corrélée à y sinon l'inverse
selector = SelectFromModel(SGDClassifier(random_state=0),threshold = 'mean') #On enlève les variables dont les paramètres sont inférieurs à la moyenne des paramètres
selector.fit_transform(X,y)
print()
print('SelectFromModel:')
print('Les variables sélectionnées à partir du model SGDClassifier sont la 3è et la dernière:',selector.get_support())
print('Les coeffs du Classifier sont :\n',selector.estimator_.coef_)
print('La moyenne des coeffs pour chaque variable est:',selector.estimator_.coef_.mean(axis=0))
print('La moyenne de tous les coeffs est:',selector.estimator_.coef_.mean())
print('On retrouve bien que les variables sélectionnées sont les deux dernières qui ont leur param >= la moyenne')

#Si on veut garder au minimum N variables, on élimine k var ayant les poids les plus faibles à chaque itération de la simulation d'un modèle tant qu'il y a plus de N variables
# RFECV est comme RFE avec de la cross validation

k = 1
N = 2
selector = RFECV(SGDClassifier(random_state=0),step = k,min_features_to_select=N,cv=4)
selector.fit_transform(X,y)
print("Le rank de chque variables est :",selector.ranking_)
print("Le score obtenu pour notre SGDClassifier après l'élimination de k variant de 1 à 3 variables éliminées est:",selector.grid_scores_)

plt.show()
