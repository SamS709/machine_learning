import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import Stack
from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# But : utiliser les résultats de plusieurs models pour faire une prédiction

X,y = make_moons(n_samples=500,noise=0.3,random_state=0)
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=0)



#VOTING

#On prend comme model final celui qui prend en compte l'avis de chacun des models initiaux sous forme de vote

model1 = SGDClassifier(random_state=0)
model2 = DecisionTreeClassifier(random_state=0)
model3 = KNeighborsClassifier(n_neighbors=2)
# permet d'indiquer si le choix se fait en fonction de la proba déterminée par chaque classe(soft) ou directement sur les prédictions(hard)
model4 = VotingClassifier([('SGD',model1),('Tree',model2),('KNeighboors',model3)],voting='hard')

models = [model1,model2,model3,model4]

for model in models:
    model.fit(Xtrain,ytrain)
    print(model.__class__.__name__,model.score(Xtest,ytest))



# BAGGING

# Utiliser en situation d'overfitting des models de base
# On tire des données aléatoirement dans le trainset et on entrène chqcun des modèles sur ceux-ci
# Avec BaggingClassifier on choisit l'estimateur
model = BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=100) #n_estimators est le nombre d'estimateurs (models) entraînés sur chacun des n_estimators datasets sélectionnés aléatoirement
model.fit(Xtrain,ytrain)
print(model.__class__.__name__, model.score(Xtest, ytest))
# Avec random forest, l'estimateur est un arbre de décision
model = RandomForestClassifier(n_estimators=100) #n_estimators est le nombre d'estimateurs (models) entraînés sur chacun des n_estimators datasets sélectionnés aléatoirement
model.fit(Xtrain,ytrain)
print(model.__class__.__name__, model.score(Xtest, ytest))



# BOOSTING

# Utiliser en situation d'underfitting pour améliorer à la suite les models
# On entraîne une suite de models à détecter les erreurs du model précédent
#Adaboost
model = AdaBoostClassifier(n_estimators=100) # n_estimators est le nb d'estimateurs entraînés à la suite
model.fit(Xtrain,ytrain)
print(model.__class__.__name__, model.score(Xtest, ytest))
#Gradienboost
model = GradientBoostingClassifier(n_estimators=100) # n_estimators est le nb d'estimateurs entraînés à la suite
model.fit(Xtrain,ytrain)
print(model.__class__.__name__, model.score(Xtest, ytest))



#STACKING

#Utile si chaque model de base a passé du temps à apprendre et est donc assez performant
# On entraîne un model à prédire qui à juste parmi qqes models déjà entraînés
final_model = KNeighborsClassifier()
model = StackingClassifier([('SGD',model1),('Tree',model2),('KNeighboors',model3)],final_model)
model.fit(Xtrain,ytrain)
print(model.__class__.__name__, model.score(Xtest, ytest))
