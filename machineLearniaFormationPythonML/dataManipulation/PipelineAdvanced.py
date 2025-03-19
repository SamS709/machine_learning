import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.linear_model import SGDClassifier
from sklearn.compose import make_column_transformer, make_column_selector
import seaborn as sns

titanic = sns.load_dataset('titanic')
print(titanic.head())

# PB: on ne peut pas appliquer un scaler direct à des données de type str
# SOL : on applique des pipeline distinctes selon que ce sont des nombres ou pas

y = titanic['survived']
X = titanic.drop(['survived'],axis=1)

#On sépare X en deux catégories
numerical_features = ['pclass','age','fare']
str_features = ['sex','deck','alone']
#Autre solution de séparation:
numerical_features = make_column_selector(dtype_include=np.number) # on prend TOUTES les features numériques de X
str_features = make_column_selector(dtype_exclude=np.number) # on prend TOUTES les features str de X

#On traite les donnée float et str de manière différente
numerical_pipeline = make_pipeline(SimpleImputer(),StandardScaler()) #simpleImputer retire les valeurs son-renseignées
str_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder()) #strategy = remplacer les valeurs manquantes par les plus fréq

# On transforme nos données selon leur type
preprocessor = make_column_transformer((numerical_pipeline,numerical_features),(str_pipeline,str_features))

# On utilise notre transformer avant d'appliquer notre modele de regression
model = make_pipeline(preprocessor,SGDClassifier())

model.fit(X,y)
print('score =',model.score(X,y))


# PIPELINE Parallèles avec make_union

numerical_features = X[['age','fare']]
# On applique deux transformer distincts sur les features sélectionnées
# Ici, on a deux features auxquelles on applique StandardScaler puis on applique aux même features de base un Binarizer (en parallèle)
# On part de 4 features et on se retrouve avec 4

imputer = SimpleImputer()
numerical_features = imputer.fit_transform(numerical_features) # on enlève les valeurs non renseignées d'âge
pipeline = make_union(StandardScaler(),Binarizer())
X_parall = pipeline.fit_transform(numerical_features)
print(X_parall)