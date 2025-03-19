import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, LabelBinarizer, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

#On fit le transformer sur le trainSet et on l'applique au testSet avant de passer au test du modèle
# 1) transformer.fit(Xtrain)
# 2) Xtrain_scaled = transformer.transform(Xtrain)
# 3) Xtest_scaled = transformer.transform(Xtest)
# 1) + 2) <=> transformer.fit_transform(X)




# ENCODAGE

y = np.array(['chien','chat','chat','oiseau','chien'])
X = np.array([['chat','poils'],
              ['chien','poils'],
              ['chat','poils'],
              ['oiseau','plumes']])

#Pour transformer des str en chiffres entiers distincts
#Pas fou car donne des valeurs plus grandes à certains str
# Ca fausse l'entrainement car neurone pas sensible à 1>2>3..
encoder = LabelEncoder()
y_transformed = encoder.fit_transform(y)
print('y ordinal : \n',y_transformed)
y_init = encoder.inverse_transform(y_transformed) # pour revenir au y initial
print(y_init)

# Sur un tableau X à plusieurs dimensions
encoder = OrdinalEncoder()
X_transformed = encoder.fit_transform(X)
print('X Ordinal : \n',X_transformed)

#MIEUX : encodage One Hot
# on créé autant de dimensions nécessaires pour distinguer chien de chat
encoder = LabelBinarizer()
y_transformed = encoder.fit_transform(y)
print('y One Hot : \n',y_transformed)

# Sur un tableau X à plusieurs dimensions
encoder = OneHotEncoder(sparse=True)
X_transformed = encoder.fit_transform(X)
print('X oneHot : \n',X_transformed)
# Prendre moins de place: mettre sparse = True pour avoir un encodage moins spacieux des matrices transformées qui sont creuses




#NORMALISATION

X = np.array([[70],
              [80],
              [120]])

iris = load_iris()
X1 = load_iris().data
y1 = load_iris().target
Xtrain,Xtest,ytrain,ytest = train_test_split(X1,y1)

# MinMax => les valeurs de X sont dans [0,1]:  Xscaled = (X-Xmin)/(Xmax-Xmin)
scalerMinMax = MinMaxScaler()
X_scaled = scalerMinMax.fit_transform(X)
print('Scale MinMax : \n',X_scaled)

# StandardScaler: chaque variable a une moyenne nulle et un écart-type (ET) de 1: Xscaled = (X-mean(X))/ET(X)
scalerStandard = StandardScaler()
X_scaled = scalerStandard.fit_transform(X)
print('Scale StandardScaler : \n',X_scaled)

#Pour faire face aux valeurs aberrantes: soustraire la mediane plutôt que la moyenne
#IQR = écart entre premier et troisème quartile de donnée
# RobustScaler => valeurs aberrantes négligées grâce à la médiane: Csvaled = (X-médiane)/IQR
scalerRobust = RobustScaler()
X_scaled = scalerRobust.fit_transform(X)
print('Scale RobustScaler : \n',X_scaled)

X1_minmax = scalerMinMax.fit_transform(X1)
X1_Standard = scalerStandard.fit_transform(X1)
X1_Robust = scalerRobust.fit_transform(X1)
plt.scatter(X1[:,2],X1[:,3],label='initial')
plt.scatter(X1_minmax[:, 2], X1_minmax[:, 3], label='MinMax')
plt.scatter(X1_Standard[:, 2], X1_Standard[:, 3], label='Standard')
plt.scatter(X1_Robust[:, 2], X1_Robust[:, 3], label='Robust')
plt.legend()


#AUTRES TRANSFORMERS

#POLYNOMIALFEATURES
# transforme X en X,X^2,...,X^N    Voir fichier sickitRegression.py

#VOIR AUTRES transformeurs dans la video de machine learnia


# PIPELINE
#permet de regrouper le transformer et le modèle en 1 objet

model = make_pipeline(StandardScaler(),SGDClassifier())
model.fit(Xtrain,ytrain)
print('score=',model.score(Xtest,ytest))



plt.show()




