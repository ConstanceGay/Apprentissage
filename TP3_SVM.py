from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import warnings
import time

warnings.simplefilter('ignore')
mnist = fetch_openml('mnist_784')

data = mnist.data
target = mnist.target

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

clf = svm.SVC(kernel='linear')
clf.fit(xtrain, ytrain)
predicted = clf.predict([mnist.data[3]])
print("**************************************************\n")

kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

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

y_axis = [temps, precision, erreur]
y_label = ['Temps', 'Precision', 'Erreur']
for i in range(0, 3):
    plt.plot(nb_couches, y_axis[i])
    plt.ylabel(y_label[i])
    # plt.xlabel("Nombre de couches")
    plt.show()
print("**************************************************\n")

cost = [0.1, 0.4, 0.6, 0.8, 1.0]

temps = []
precision = []
erreur = []
erreur_train = []

for i in cost:
    begin = time.process_time()
    clf = svm.SVC(c=i)
    clf.fit(xtrain, ytrain)

    erreur.append(metrics.zero_one_loss(ytest, clf.predict(xtest)))
    erreur_train.append(metrics.zero_one_loss(ytrain, clf.predict(xtrain)))

    score = clf.score(xtest, ytest)
    precision.append(score)
    print(i, " cost précision : ", score)

    t = time.process_time() - begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

y_axis = [temps, precision, erreur, erreur_train]
y_label = ['Temps', 'Precision', 'Erreur test', 'Erreur train']
for i in range(0, 4):
    plt.plot(nb_couches, y_axis[i])
    plt.ylabel(y_label[i])
    plt.xlabel("Cost")
    plt.show()

cm = confusion_matrix(ytest, clf.predict(xtest))
