from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
import warnings
import time
import sklearn.neighbors as neighbors
import numpy as np

warnings.simplefilter('ignore')
mnist = fetch_openml('mnist_784')

sample = np.random.randint(70000, size=10000)
data = mnist.data[sample]
target = mnist.target[sample]

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

clf = svm.SVC(kernel='linear')
clf.fit(xtrain, ytrain)
predicted = clf.predict([mnist.data[3]])
print("**************************************************\n")
print("Valeur de précision, temps et erreur pour les différentes fonctions kernel")
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
temps = []
precision = []
erreur = []

for i in kernel:
    begin = time.process_time()
    clf = svm.SVC(kernel=i)
    clf.fit(xtrain, ytrain)
    erreur.append(metrics.zero_one_loss(ytest, clf.predict(xtest)))
    score = clf.score(xtest, ytest)
    precision.append(score)
    print(i, " précision : ", score)
    t = time.process_time() - begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision, erreur]
# y_label = ['Temps', 'Precision', 'Erreur']
# for i in range(0, 3):
#     plt.plot(nb_couches, y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.show()
print("**************************************************\n")

print("Valeur de précision, temps et erreur pour les différentes fonctions kernel")
cost = [0.1, 0.4, 0.6, 0.8, 1.0]
temps = []
precision = []
erreur = []
erreur_train = []

for i in cost:
    begin = time.process_time()
    clf = svm.SVC(C=i)
    clf.fit(xtrain, ytrain)
    erreur.append(metrics.zero_one_loss(ytest, clf.predict(xtest)))
    erreur_train.append(metrics.zero_one_loss(ytrain, clf.predict(xtrain)))
    score = clf.score(xtest, ytest)
    precision.append(score)
    print(i, " cost précision : ", score)
    t = time.process_time() - begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision, erreur, erreur_train]
# y_label = ['Temps', 'Precision', 'Erreur test', 'Erreur train']
# for i in range(0, 4):
#     plt.plot(nb_couches, y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.xlabel("Cost")
#     plt.show()
print("**************************************************\n")

print("Matrice de confusion pour les paramètres c=0.6 et la fonction kernel 'poly")
clf = svm.SVC(C=0.6,kernel='poly')
clf.fit(xtrain,ytrain)
cm = confusion_matrix(ytest, clf.predict(xtest))
print(cm)




