import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv')

print(iris.head())

sns.pairplot(iris,hue='species')

titanic = sns.load_dataset('titanic')

titanic.drop(['alone','alive','who','adult_male','embark_town','class'], inplace=True,axis = 1)

titanic.dropna(axis=0,inplace=True)
print(titanic.head())
plt.figure()
sns.catplot(x='pclass',y='age',data=titanic,hue='sex')
plt.figure()
sns.boxplot(x='pclass',y='age',data=titanic,hue='sex')
plt.figure()
sns.distplot(titanic['fare'])
plt.figure()
sns.heatmap(titanic.corr())
plt.show()