from typing import List
from sklearn.datasets import fetch_openml
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time

# import des données
mnist = fetch_openml('mnist_784') 

# 49 000 premiere données pour le train
train_data = mnist.data[:49000]
train_target = mnist.target[:49000]

# les donnees d'après pour le test
test_data = mnist.data[49000:70000]
test_target = mnist.target[49000:70000]

# Entrainement du classifieur
clf = nn.MLPClassifier(hidden_layer_sizes=50)
clf.fit(train_data, train_target)
print("Précision avec clf.score : ", clf.score(test_data, test_target))
print("Précision avec metrics.precision_score : ", metrics.precision_score(test_target,
                                                                           clf.predict(test_data), average='micro'))
print("**************************************************\n")

images = mnist.data.reshape((-1, 28, 28))
print("Valeur prédite pour l'image suivante : ", clf.predict([mnist.data[3]]))
plt.imshow(images[3], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
print("**************************************************\n")

# Variation nombre de couches avec nb neurones constant
print(" On fait varier le nombre de couches 2,10,20,50 et 100 avec chacun 50 neurones")
nb_couches: List[int] = [2, 10, 20, 50, 100]
temps = []
precision = []
erreur = []

for i in nb_couches:
    begin = time.process_time()
    couche = [50] * i
    clf = nn.MLPClassifier(hidden_layer_sizes=couche)
    clf.fit(train_data, train_target)
    score = clf.score(test_data, test_target)
    precision.append(score)
    print(i, " couches : ", score)
    error = metrics.zero_one_loss(test_target, clf.predict(test_data))
    erreur.append(error)
    print("Erreur : ", error)
    t = time.process_time() - begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision,erreur]
# y_label = ['Temps','Precision','Erreur']
# for i in range (0,3) :
#     plt.plot(activation,y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.show()
print("**************************************************\n")

# 2 modèles de : 50 et 20 couches cachées & de taille commençant à 60, décrementant respectivement de 1 et 2
print("2 modèles de : 50 et 20 couches cachées & de taille commençant à 60, décrementant respectivement de 1 et 2")

temps = []
precision = []
erreur = []
nb_couches = [20, 50]

L_cinquante = []
for i in range(60, 10, -1):
    L_cinquante.append(i)
L_vingt = []
for i in range(60, 20, -2):
    L_vingt.append(i)

begin = time.process_time()
clf = nn.MLPClassifier(hidden_layer_sizes=L_vingt)
clf.fit(train_data, train_target)
erreur.append(metrics.zero_one_loss(test_target, clf.predict(test_data)))
score = clf.score(test_data, test_target)
precision.append(score)
print("20 couches precision : ", score)
t = time.process_time() - begin
temps.append(t)
print("Temps écoulé : ", t, "\n")

begin = time.process_time()
clf = nn.MLPClassifier(hidden_layer_sizes=L_cinquante)
clf.fit(train_data, train_target)
erreur.append(metrics.zero_one_loss(test_target, clf.predict(test_data)))
score = clf.score(test_data, test_target)
precision.append(score)
print("50 couches precision : ", score)
t = time.process_time() - begin
temps.append(t)
print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision,erreur]
# for i in range (0,3) :
#     plt.plot(nb_couches,y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.xlabel("Nombre de couches")
#     plt.show()
print("**************************************************\n")

# algorithmes d’optimisation disponibles : L-BFGS, SGD et Adam
print("Algorithmes d’optimisation disponibles : L-BFGS, SGD et Adam")
algo = ['adam', 'lbfgs', 'sgd']
temps = []
precision = []
erreur = []

for i in algo:
    begin = time.process_time()
    clf = nn.MLPClassifier(hidden_layer_sizes=50, solver=i)
    clf.fit(train_data, train_target)
    erreur.append(metrics.zero_one_loss(test_target, clf.predict(test_data)))
    score = clf.score(test_data, test_target)
    precision.append(score)
    print("Précision ", i, " : ", score)
    t = time.process_time()-begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision,erreur]
# for i in range (0,3) :
#     plt.plot(algo,y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.show()
print("**************************************************\n")

# Varier les fonctions d’activation {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
print("On fait varier les fonctions d'activation : ")
activation = ['identity', 'logistic', 'tanh', 'relu']
temps = []
precision = []
erreur = []

for i in activation:
    begin = time.process_time()
    clf = nn.MLPClassifier(hidden_layer_sizes=50, activation=i)
    clf.fit(train_data, train_target)
    erreur.append(metrics.zero_one_loss(test_target, clf.predict(test_data)))
    score = clf.score(test_data, test_target)
    precision.append(score)
    print("Précision ", i, " : ", score)
    t = time.process_time()-begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision,erreur]
# for i in range (0,3) :
#     plt.plot(activation,y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.show()
print("**************************************************\n")

# Changer la valeur de la régularisation L2 (paramètre α)
print("On fait varier la valeur de régularisation alpha")
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
temps = []
precision = []
erreur = []

for i in alpha:
    begin = time.process_time()
    clf = nn.MLPClassifier(hidden_layer_sizes=50, alpha=i)
    clf.fit(train_data, train_target)
    erreur.append(metrics.zero_one_loss(test_target, clf.predict(test_data)))
    score = clf.score(test_data, test_target)
    precision.append(score)
    print("alpha = ", i, " précision : ", score)
    t = time.process_time()-begin
    temps.append(t)
    print("Temps écoulé : ", t, "\n")

# y_axis = [temps, precision,erreur]
# for i in range (0,3) :
#     plt.plot(alpha,y_axis[i])
#     plt.ylabel(y_label[i])
#     plt.xlabel("alpha")
#     plt.show()
print("**************************************************\n")

print("Modèle avec les paramètres qui nous paraissent les plus appropriés")
L_vingt = []
for i in range(60, 20, -2):
    L_vingt.append(i)

begin = time.process_time()
clf = nn.MLPClassifier(hidden_layer_sizes=L_vingt, solver='adam', activation='relu', alpha=1.0)
clf.fit(train_data, train_target)
print("20 couches : ", clf.score(test_data, test_target))
print("Temps écoulé : ", time.process_time() - begin, "\n")
print("**************************************************\n")
