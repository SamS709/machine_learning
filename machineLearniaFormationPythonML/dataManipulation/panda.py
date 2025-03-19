
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel('titanic3.xls')

print('shape =',data.shape)
print('head =',data.head())
print('columns =',data.columns)

#on élimine des colonnes :
#inplace = True permet de faire la modification directement sur data
data.drop([ 'name', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], inplace=True,axis = 1)
#Equivalent à data = data.drop([ 'name', 'sibsp', 'parch', 'ticket','fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest', axis = 1])

print('columns après supression =',data.columns)
print('head après suppression =\n', data.head())

#stats de base sur le dataset avec describe
print('stats =\n',data.describe()) #on remarque que le nombre d'âges renseignés est plus faible que les autres infos : NaN

#Supprimer les lignes NaN:
data.dropna(axis=0,inplace=True)
print(data.shape)

#compter le nombre de personne dans chaque classe
print('nombre de passager par classe\n',data['pclass'].value_counts())

#Graphique
data['age'].hist()
#plt.show()

print('Moyennes par sexe et par classe\n',data.groupby(['sex','pclass']).mean())

#structure d'une série : un index et les âges à droite
print("Structure d'une série\n",data['age'])

#pour avoir les dix premières colonnes de âges:
print("10 premières colles de age\n",data['age'][0:10])

#selectionner les infos des personnes de moins de 18 ans
print("Infos des personnes de moins de 18 ans\n",data[data['age']<18])

#raisonner comme avec un tableau numpy avec iloc:
data.iloc[0:2,0:2]