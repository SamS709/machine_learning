import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np

X,y = make_blobs(n_samples=100, centers=3,random_state=0)

# K-means clusturing : Trouver les centroîdes d'un cluster de points

# Nombre de centroïdes connnus
plt.figure("Clustering et centroïdes des clusters")
model = KMeans(n_clusters=3)
model.fit(X)
print('chaque point de X appartient à la classe 0,1 ou 2:',model.predict(X))
print(X)
plt.scatter(X[:,0],X[:,1],c=model.predict(X))
print('Les coordonnées des clusters sont:\n',model.cluster_centers_)
clutersCenters = model.cluster_centers_
plt.scatter(clutersCenters[:,0],clutersCenters[:,1],c='r')
print("La somme des distances entre le cluster et les points d'un centroîde est :", model.inertia_)

#Nombre de centrïdes inconnus : Elbow methode (méthode du coude) : donne le nb optimal de clusters
inertia = []
k_range = range(1,20)
for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertia.append(model.inertia_)
plt.figure("Zone de COUDE : Trouver le nb de cluster optimaux")
# Au coude,on trouve le nb de cluster optimal:
# Avant, l'intertia est grande car le nombre de cluster est trop faible : la somme des distance des points associés à chaque centroïde est grande
# Après, on ajoute des clusters inutiles : on overfit : chaque point tend à avoir un centroïde et on est inccapable de généraliser
plt.plot(k_range,inertia)

# IsolationForest : repérer une anomalie dans un dataset
plt.figure("Détection d'anomalies")
X,y = make_blobs(50,centers=1,cluster_std=0.1, random_state=0)
X[-1,:]= np.array([2.25,5])
model = IsolationForest(contamination=0.01) # contamination est le nb d'erreurs qu'on estime avoir ds le dataset
model.fit(X)
prediction = model.predict(X)
print("La détection d'anomalie donne:", prediction)
plt.scatter(X[:,0],X[:,1],c=prediction)

# Détecter les anomalies du dataset digit:
plt.figure("Détection d'anomalies digit")
digits = load_digits()
images = digits.images
X = digits.data
y = digits.target
model = IsolationForest(contamination=0.02)
model.fit(X)
contaminated = model.predict(X)==-1
first_contaminated = images[contaminated][0] # on remarque que le chiffre est complexe à déchiffrer même pour un humain
plt.imshow(first_contaminated)

#Réduction de la dimension
# ATTENTION : il faut standardiser nos données avec StandardScaler pour qu'elles soient centrées de variance 1
plt.figure('Reduction de la dimension : de 64 à 2 variabes')
model = PCA(n_components=64) # On réduit le nb de variables de X à 2 alors qu'il en comprend 64 (son nb de pixels)
Xreduced = model.fit_transform(X)
plt.scatter(Xreduced[:,0],Xreduced[:,1],c=y) # on voit que le nb de variables n'est pas suffisant
# Le but est de conserver entre 95 et 99% de la variance de nos données
Xreduced = model.fit_transform(X)
var = model.explained_variance_ratio_ #pourcentage de la variance conservée par chacune des variables
cumsum = np.cumsum(var)
print("Le pourcentage de variance apporté par la première variable de X est ",var[0])
print("Le tableau des sommes cumulées des indices précédentq de la variance est :",cumsum)
proportion = 0.95
print("A partir de la ",np.argmax(cumsum>proportion)," ème variable, la variance de nos données est inférieure à", proportion*100,"% de sa valeur initiale")
#On trouve que l'on a besoin de conserver que 28 variables pour conserver une variance de 95% de l'initiale
plt.figure('Reduction de la dimension variance = 95%')
model = PCA(n_components=28)
Xreduced = model.fit_transform(X)
Xrecovered = model.inverse_transform(Xreduced) #On visualise les images obtenues apres compression,à partir de la variance choisie
plt.imshow(Xrecovered[0].reshape((8,8)))

#Conserver x pourcent de la variance : on fait la même chose que précédemment mais en plus rapide
x = 0.95
model = PCA(n_components=x)
model.fit_transform(X)
print("le nb de variables nécessaires pour obtenir",x*100,"% de la variance initiale est",model.n_components_)

plt.show()