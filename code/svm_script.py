#Importation des bibliothèques

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

######################################Définition des fonctions de svm_source.py
def rand_gauss(n=100, mu=[1, 1], sigmas=[0.1, 0.1]):
    """ Sample n points from a Gaussian variable with center mu,
    and std deviation sigma
    """
    d = len(mu)
    res = np.random.randn(n, d)
    return np.array(res * sigmas + mu)


def rand_bi_gauss(n1=100, n2=100, mu1=[1, 1], mu2=[-1, -1], sigmas1=[0.1, 0.1],
                  sigmas2=[0.1, 0.1]):
    """ Sample n1 and n2 points from two Gaussian variables centered in mu1,
    mu2, with respective std deviations sigma1 and sigma2
    """
    ex1 = rand_gauss(n1, mu1, sigmas1)
    ex2 = rand_gauss(n2, mu2, sigmas2)
    y = np.hstack([np.ones(n1), -1 * np.ones(n2)])
    X = np.vstack([ex1, ex2])
    ind = np.random.permutation(n1 + n2)
    return X[ind, :], y[ind]

###############################################################################
#           Displaying labeled data
###############################################################################

symlist = ['o', 's', 'D', 'x', '+', '*', 'p', 'v', '-', '^']


def plot_2d(data, y=None, w=None, alpha_choice=1):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""

    k = np.unique(y).shape[0]
    color_blind_list = sns.color_palette("colorblind", k)
    sns.set_palette(color_blind_list)
    if y is None:
        labs = [""]
        idxbyclass = [range(data.shape[0])]
    else:
        labs = np.unique(y)
        idxbyclass = [np.where(y == labs[i])[0] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.scatter(data[idxbyclass[i], 0], data[idxbyclass[i], 1],
                    color=color_blind_list[i], s=80, marker=symlist[i])
    plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    mx = np.min(data[:, 0])
    maxx = np.max(data[:, 0])
    if w is not None:
        plt.plot([mx, maxx], [mx * -w[1] / w[2] - w[0] / w[2],
                              maxx * -w[1] / w[2] - w[0] / w[2]],
                 "g", alpha=alpha_choice)

###############################################################################
#           Displaying tools for the Frontiere
###############################################################################


def frontiere(f, X, y, w=None, step=50, alpha_choice=1, colorbar=True,
              samples=True):
    """ trace la frontiere pour la fonction de decision f"""
    # construct cmap

    min_tot0 = np.min(X[:, 0])
    min_tot1 = np.min(X[:, 1])

    max_tot0 = np.max(X[:, 0])
    max_tot1 = np.max(X[:, 1])
    delta0 = (max_tot0 - min_tot0)
    delta1 = (max_tot1 - min_tot1)
    xx, yy = np.meshgrid(np.arange(min_tot0, max_tot0, delta0 / step),
                         np.arange(min_tot1, max_tot1, delta1 / step))
    z = np.array([f(vec) for vec in np.c_[xx.ravel(), yy.ravel()]])
    z = z.reshape(xx.shape)
    labels = np.unique(z)
    color_blind_list = sns.color_palette("colorblind", labels.shape[0])
    sns.set_palette(color_blind_list)
    my_cmap = ListedColormap(color_blind_list)
    plt.imshow(z, origin='lower', interpolation="mitchell", alpha=0.80,
               cmap=my_cmap, extent=[min_tot0, max_tot0, min_tot1, max_tot1])
    if colorbar is True:
        ax = plt.gca()
        cbar = plt.colorbar(ticks=labels)
        cbar.ax.set_yticklabels(labels)

    labels = np.unique(y)
    k = np.unique(y).shape[0]
    color_blind_list = sns.color_palette("colorblind", k)
    sns.set_palette(color_blind_list)
    ax = plt.gca()
    if samples is True:
        for i, label in enumerate(y):
            label_num = np.where(labels == label)[0][0]
            plt.scatter(X[i, 0], X[i, 1], color=color_blind_list[label_num],
                        s=80, marker=symlist[label_num])
    plt.xlim([min_tot0, max_tot0])
    plt.ylim([min_tot1, max_tot1])
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    if w is not None:
        plt.plot([min_tot0, max_tot0],
                 [min_tot0 * -w[1] / w[2] - w[0] / w[2],
                  max_tot0 * -w[1] / w[2] - w[0] / w[2]],
                 "k", alpha=alpha_choice)


def plot_gallery(images, titles, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90,
                       hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i])
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


def title(y_pred, y_test, names):
    pred_name = names[int(y_pred)].rsplit(' ', 1)[-1]
    true_name = names[int(y_test)].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

#%%
########################################################QUESTION 1

# Fixer la graine
np.random.seed(42)
iris = datasets.load_iris()

X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]


# mélange les données
X, y = shuffle(X, y, random_state=42)

# Divise les données en un ensemble d'entraînement et un ensemble de test avec 50 % 
# des données pour chaque ensemble
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

# Utilise GridSearchCV pour tester plusieurs valeurs du paramètre C et sélectionner le meilleur modèle
clf_linear = GridSearchCV(SVC(), parameters, n_jobs=-1)
clf_linear.fit(X_train, y_train)

print(clf_linear.best_params_)
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#%%
########################################################QUESTION 2


Cs = list(np.logspace(-3, 3, 10))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]# Définit les degrés à tester (1, 2, 3)

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(SVC(), parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

print(clf_poly.best_params_)# Affiche les meilleurs paramètres trouvés
print('Generalization score for polynomial kernel: %s, %s' %# Score sur l'entraînement et le test
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


# Affiche les résultats

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()
#plt.savefig('iris_front.svg')
#%%
########################################################QUESTION 3 (voir code svm_gui.py)
#%%


########################################################QUESTION 4

# Chargement des données
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)

# data_home='.'

# Introspection des tableaux d'images pour connaître leurs dimensions (pour l'affichage)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# Le label à prédire est l'identité de la personne
target_names = lfw_people.target_names.tolist()

# Sélection d'un duo de personnes à classifier
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)


#%%
####################################################################
# Extraction des caractéristiques

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Séparation des données en un ensemble d'entraînement et un ensemble de test (50% pour le test)
X_train, X_test, y_train, y_test, images_train, images_test = \
    train_test_split(X, y, images, test_size=0.5, random_state=10)
indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

# affiche un extrait des données
plot_gallery(images_train, [names[i] for i in y_train], n_row=3)
plt.show()


np.random.seed(42)
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
train_scores = []
test_errors = []


for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    
    # Stocker le score d'apprentissage
    train_scores.append(clf.score(X_train, y_train))
    
    # Calculer l'erreur de prédiction sur le jeu de test
    test_errors.append(1 - clf.score(X_train, y_train))

# Trouver le meilleur C en fonction des scores d'apprentissage
ind = np.argmax(train_scores)
print("Best C: {}".format(Cs[ind]))

# Tracer les courbes des scores d'apprentissage
plt.figure()
plt.plot(Cs, train_scores, label="Scores d'apprentissage")
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.legend()
plt.tight_layout()
#plt.savefig("./plot/err_train.png")
plt.show()

# Tracer les erreurs de prédiction dans un graphique séparé
plt.figure()
plt.plot(Cs, test_errors, label="Erreurs de prédiction", linestyle='--')
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Erreurs de prédiction")
plt.xscale("log")
plt.legend()
plt.tight_layout()
#plt.savefig("./plot/err_test.png")
plt.show()

print("Best score: {}".format(np.max(train_scores)))

#%% Prédiction sur le jeu de test
print("Predicting the people names on the testing set")
t0 = time()

# Entraîner le modèle avec le meilleur C trouvé
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)

# Prédire les étiquettes pour les images de X_test
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

# Le niveau de chance est la précision obtenue en prédisant toujours la classe majoritaire
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))

# Précision finale du modèle
print("Accuracy : %s" % clf.score(X_test, y_test))




# Visualisation des prédictions

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# visualisation des coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

#%%
########################################################QUESTION 5

np.random.seed(42)

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X,y)# Exécute le SVM sans variables de nuisance

print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, )
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy,y)# Exécute le SVM avec variables de nuisance

#%%



########################################################QUESTION 6

np.random.seed(42)
# Réduction de dimension avec PCA
print("Score après réduction de dimension")
n_components = 150 # Nombre de composantes, peut être ajusté150

# Appliquer la PCA
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
X_noisy_pca = pca.fit_transform(X_noisy)

# Passer les données réduites à la fonction run_svm_cv
run_svm_cv(X_noisy_pca, y)

#graphique de la variance expliqué
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.arange(1, n_components + 1), explained_variance)
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.title('Variance expliquée par PCA')
plt.show()



#graphique final
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import svm

np.random.seed(42)
# Liste pour stocker les scores en fonction du nombre de composantes
scores_train = []
scores_test = []
n_components_list = np.arange(50, 151, 10)  # Tester les valeurs de 50 à 150 avec un pas de 10

def run_svm_cv_return_score(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    # Retourner les scores d'entraînement et de test
    return _clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)

# Boucle sur différents nombres de composantes PCA
for n_components in n_components_list:
    print(f"Nombre de composantes: {n_components}")
    
    # Appliquer PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    X_noisy_pca = pca.fit_transform(X_noisy)
    
    # Obtenir les scores
    score_train, score_test = run_svm_cv_return_score(X_noisy_pca, y)
    
    # Stocker les scores
    scores_train.append(score_train)
    scores_test.append(score_test)

# Calculer les moyennes des scores
mean_score_train = np.mean(scores_train)
mean_score_test = np.mean(scores_test)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(n_components_list, scores_train, label='Score d\'entraînement', marker='o')
plt.plot(n_components_list, scores_test, label='Score de test', marker='s')

# Ajouter les lignes pointillées pour les moyennes
plt.axhline(y=mean_score_train, color='blue', linestyle='--', label='Moyenne entraînement')
plt.axhline(y=mean_score_test, color='orange', linestyle='--', label='Moyenne test')

plt.xlabel('Nombre de composantes principales (PCA)')
plt.ylabel('Score (Accuracy)')
plt.title('Impact du nombre de composantes PCA sur les performances du SVM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
