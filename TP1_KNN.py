from sklearn.datasets import fetch_openml
import sklearn.neighbors as neighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')   # suppression warnings

# recuperation du dataset
mnist = fetch_openml('mnist_784')

# selection de certaines images
sample = np.random.randint(70000, size=5000)   # 5000 nombres au hasard
data_sample = mnist.data[sample]               # on récupère les images
target_sample = mnist.target[sample]           # on récupère les réponses

# entrainement du classifieur avec 80% et k=10
xtrain,xtest,ytrain,ytest = train_test_split(data_sample, target_sample, train_size=0.8)
clf = neighbors.KNeighborsClassifier(10)
clf.fit(xtrain, ytrain)

# Test de la valeur de la 4eme image
images = mnist.data.reshape((-1, 28, 28))
print("Valeur predite pour l'image : ", clf.predict([mnist.data[3]]))
plt.imshow(images[3], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

print("Score sur l'echantillon de test : ", clf.score(xtest,ytest))
print("Score sur les donnees d'apprentissage : ", clf.score(xtrain,ytrain))
print("**************************************************\n")

# Test sur 2 à 15 voisins
print("Boucle de test de 2 à 15 voisins : ")
for i in range(2, 16):
    clf = neighbors.KNeighborsClassifier(i)
    clf.fit(xtrain, ytrain)
    print(i, " : ", clf.score(xtest, ytest))
print("**************************************************\n")

# Utilisation de KFold
print("KFold : ")
kf = KFold(10, shuffle=True)
for train, test in kf.split(data_sample):
    clf.fit(data_sample[train], target_sample[train])
    print("Precision : ", clf.score(data_sample[test], target_sample[test]))
print("**************************************************\n")

# Variations pourcentage données d'apprentissage
print("On fait varier les pourcentages de données d'apprentissage 10-90% : ")
for i in range(10, 100, 10):
    xtrain, xtest, ytrain, ytest = train_test_split(data_sample, target_sample, train_size=i/100)
    clf = neighbors.KNeighborsClassifier(10)
    clf.fit(xtrain, ytrain)
    print(i,"% précision : ", clf.score(xtest,ytest))
print("**************************************************\n")

# Variation taille données d'apprentissage
print("On fait varier la tailles des données d'apprentissage : ")
xtrain, xtest, ytrain, ytest = train_test_split(data_sample, target_sample, train_size=0.8)
taille = [100,1000,5000,10000]
for i in taille:
    sample = np.random.randint(70000, size=i)
    data_sample = mnist.data[sample]
    target_sample = mnist.target[sample]

    clf = neighbors.KNeighborsClassifier(10)
    clf.fit(xtrain, ytrain)
    print(i, "échantillons précision : ", clf.score(xtest,ytest))
print("**************************************************\n")

# Variation distance p
print("On fait varier les types de distances (p) qui peut prendre la valeur 1 ou 2 : ")
sample = np.random.randint(70000, size=5000)
data_sample = mnist.data[sample]
target_sample = mnist.target[sample]
p_val = [1, 2]

for i in p_val:
    clf = neighbors.KNeighborsClassifier(10, p=i)
    clf.fit(xtrain, ytrain)
    print("p = ", i, " précision : ", clf.score(xtest, ytest))
print("**************************************************\n")

# Variation n_jobs
print("On fait varier n_jobs qui peut prendre la valeur 1 ou -1 : ")
begin = time.process_time()
n_job = [-1, 1]

for i in n_job:
    clf = neighbors.KNeighborsClassifier(10, n_jobs=i)
    clf.fit(xtrain, ytrain)
    print("n_job = ", i, " |précision : ", clf.score(xtest, ytest))
    print("Temps écoulé : ", time.process_time()-begin, "\n")
    begin = time.process_time()
print("**************************************************\n")