import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, LeaveOneOut,ShuffleSplit, StratifiedKFold, GroupKFold



iris = load_iris()

X = iris.data
y = iris.target

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=5)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(Xtrain,ytrain)
print("score d'entrainement = ",model.score(Xtrain,ytrain))
print("score de test = ",model.score(Xtest,ytest))

#Cross Validation : validation croisée sur des jeux de données de validation =! de ceux de test
#ils permettent de choisir les hyperparamètres optimaux

crossVal=cross_val_score(KNeighborsClassifier(3),Xtrain,ytrain,cv=5,scoring='accuracy')
#cv=( indique le nombre de split du Xtrain pour la cross validation
print(crossVal.mean())

#observer l'influence d'un hyperparamètre sur le jeu de validation:
model = KNeighborsClassifier()
k = np.arange(1,50)
train_score,val_score = validation_curve(estimator=model,X=Xtrain,y=ytrain,param_name='n_neighbors',param_range=k,cv=5)
# l'hyperparamètre à faire varier est ici n_neighbors qui varie dans k
plt.plot(k,val_score.mean(axis=1),label='Validation')
plt.plot(k,train_score.mean(axis=1),label='Train')
plt.legend()

#Trouver la meilleure combinaison d'hyperparamètres
param_grid = {'n_neighbors':np.arange(1,20),'metric':['euclidean','manhattan']}
#les hyperparamètres à faire varier sont n_neighbors de 1 à 20 et metric entre euclidian et manhattan
grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(Xtrain,ytrain)
model = grid.best_estimator_
print('meilleur score = ',grid.best_score_,'pour les paramètres : ',grid.best_params_)
M = confusion_matrix(ytest,model.predict(Xtest))
print(M) # 2 fleurs de la classe 2 ont été rangées dans la classe 3. Les autres sont bien classées

#Courbe d'apprentiddage: influence du nombre de données d'entrainement:
N,train_score,val_score=learning_curve(model,Xtrain,ytrain,train_sizes=np.linspace(0.1,1,20),cv=5)
#trainsizes est la taille des lots d'entraineents utilisés pour chaque apprentissage :
#on observe l'influence de la taille du lot sur la le score de validation
plt.figure()
plt.plot(N,val_score.mean(axis=1),label='Validation')
plt.plot(N,train_score.mean(axis=1),label='Train')
plt.legend()


#Choix de la cross validation

#On tire une donnée pour la validation à chaque fois
cv = LeaveOneOut()
val_score=cross_val_score(KNeighborsClassifier(),Xtrain,ytrain,cv=cv)
print('cv = LeaveOneOut:',val_score)

#On fixe le nombre de données tirées à N
N=5
cv = KFold(N)
val_score=cross_val_score(KNeighborsClassifier(),Xtrain,ytrain,cv=cv)
print('cv = Kfold:',val_score)

#On répète N fois l'opération: mélanger puis diviser le set en un TrainSet et validationSet ( de taille p)
N = 4
p = 0.2
cv = ShuffleSplit(N,test_size=p)
val_score=cross_val_score(KNeighborsClassifier(),Xtrain,ytrain,cv=cv)
print('cv = Shuffle:',val_score)

# PB: il est possible d'avoir une proportion faible d'une des catégories dans l'un des sets
# SOLUTION: StratifiedKafold assure la présence de toutes les catégories dans tous les trainset et validationSet (divise en N groupes)
N = 4
cv = StratifiedKFold(N)
val_score=cross_val_score(KNeighborsClassifier(),Xtrain,ytrain,cv=cv)
print('cv = Stratified:',val_score)

#GroupKfold permet d'assurer la présence ded data ayant des caractéristiques cmmunes dans chacun des set
N = 4
cv = GroupKFold(N).get_n_splits(Xtrain,ytrain,groups=X[:,0])
val_score=cross_val_score(KNeighborsClassifier(),Xtrain,ytrain,cv=cv)
print('cv = GroupKFold:',val_score)

plt.show()


