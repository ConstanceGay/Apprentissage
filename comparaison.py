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

sample = np.random.randint(70000, size=10000)
data_sample = mnist.data[sample]               # on récupère les images
target_sample = mnist.target[sample]           # on récupère les réponses

xtrain, xtest, ytrain, ytest = train_test_split(data_sample, target_sample, train_size=0.8)

L_vingt = []
for i in range(60, 20, -2):
    L_vingt.append(i)

temps = []
precision = []
erreur = []

# KNN
begin = time.process_time()
clf_KNN = neighbors.KNeighborsClassifier(10)
clf_KNN.fit(xtrain, ytrain)
# ERREUR
erreur.append(metrics.zero_one_loss(ytest, clf_KNN.predict(xtest)))
# PRECISION
score = clf_KNN.score(xtest, ytest)
precision.append(score)
print(" KNN précision : ", score)
# TEMPS
t = time.process_time()-begin
temps.append(t)
print("Temps écoulé KNN : ", t)
cm = confusion_matrix(ytest, clf_KNN.predict(xtest))
print("Matrice de confusion\n", cm, "\n")


# ANN
begin = time.process_time()
clf_ANN = nn.MLPClassifier(hidden_layer_sizes=L_vingt)
clf_ANN.fit(xtrain, ytrain)
# ERREUR
erreur.append(metrics.zero_one_loss(ytest, clf_ANN.predict(xtest)))
# PRECISION
score = clf_ANN.score(xtest, ytest)
precision.append(score)
print(" ANN précision : ", score)
# TEMPS
t = time.process_time()-begin
temps.append(t)
print("Temps écoulé ANN : ", t)
cm = confusion_matrix(ytest, clf_ANN.predict(xtest))
print("Matrice de confusion:\n", cm, "\n")


# SVM
begin = time.process_time()
clf_SVM = svm.SVC(kernel='poly', C=0.6)
clf_SVM.fit(xtrain, ytrain)
# ERREUR
erreur.append(metrics.zero_one_loss(ytest, clf_SVM.predict(xtest)))
# PRECISION
score = clf_SVM.score(xtest, ytest)
precision.append(score)
print(" SVM précision : ", score)
# TEMPS
t = time.process_time()-begin
temps.append(t)
print("Temps écoulé SVM : ", t, "\n")
cm = confusion_matrix(ytest, clf_SVM.predict(xtest))
print("Matrice de confusion:\n", cm, "\n")


algo = ['KNN', 'ANN', 'SVM']
y_axis = [temps, precision, erreur]
y_label = ['Temps', 'Precision', 'Erreur']
for i in range(0, 3):
    plt.plot(algo, y_axis[i])
    plt.ylabel(y_label[i])
    plt.show()
