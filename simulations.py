"""
Simple simulation code to reproduce figures on bias and variance of
leave-one-out cross-validation.
"""

import pandas
import numpy as np
from scipy import ndimage

from joblib import Parallel, delayed, Memory
from sklearn.cross_validation import (LabelShuffleSplit, LeaveOneOut,
        cross_val_score)
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from matplotlib import pyplot as plt
import seaborn

def mk_data(n_samples=200, random_state=0, separability=1,
            noise_corr=2, dim=100):
    rng = np.random.RandomState(random_state)
    y = rng.random_integers(0, 1, size=n_samples)
    noise = rng.normal(size=(n_samples, dim))
    if not noise_corr is None and noise_corr > 0:
        noise = ndimage.gaussian_filter1d(noise, noise_corr, axis=0)
    noise = noise / noise.std(axis=0)
    # We need to decrease univariate separability as dimension increases
    centers = 4. / dim * np.ones((2, dim))
    centers[0] *= -1
    X = separability * centers[y] + noise
    return X, y


###############################################################################
# Code to run the cross-validations


def sample_and_cross_val_clf(train_size=200, noise_corr=2, dim=3, sep=.5,
                             random_state=0):
    clf = LinearSVC(penalty='l2', fit_intercept=True)

    n_samples = train_size + 10000
    X, y = mk_data(n_samples=n_samples,
                   separability=sep, random_state=random_state,
                   noise_corr=noise_corr, dim=dim)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    validation_score = accuracy_score(
                            y_test,
                            clf.fit(X_train, y_train).predict(X_test))

    scores = dict()
    scores['loo'] = (np.mean(cross_val_score(clf, X_train, y_train,
                            cv=LeaveOneOut(train_size)))
                        - validation_score)

    # Create 10 blocks of evenly-spaced labels for LabelShuffleSplit
    labels = np.arange(train_size) // (train_size // 10)

    scores['3 splits'] = (np.mean(cross_val_score(clf, X_train, y_train,
                    cv=LabelShuffleSplit(labels, n_iter=3, random_state=0)))
                - validation_score)
    scores['10 splits'] = (np.mean(cross_val_score(clf, X_train, y_train,
                    cv=LabelShuffleSplit(labels, n_iter=10, random_state=0)))
                - validation_score)
    scores['200 splits'] = (np.mean(cross_val_score(clf, X_train, y_train,
                    cv=LabelShuffleSplit(labels, n_iter=200, random_state=0)))
                - validation_score)

    return scores


###############################################################################
# Run the simulations

N_JOBS = 2
N_DRAWS = 30
mem = Memory(cachedir='cache')



results = pandas.DataFrame(
    columns=['loo', '3 splits', '10 splits', '200 splits'])

for sep in (1.25, 2.5, 5.):
    scores = Parallel(n_jobs=N_JOBS, verbose=10)(
                    delayed(mem.cache(sample_and_cross_val_clf))(
                            train_size=200,
                            noise_corr=1, dim=300, sep=sep,
                            random_state=i)
                    for i in range(N_DRAWS))
    results = results.append(scores)


seaborn.boxplot(results, orient='h')

plt.show()

