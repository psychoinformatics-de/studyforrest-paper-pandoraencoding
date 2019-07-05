from __future__ import division
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import multivariate_normal
from inspect import isclass
from sklearn.decomposition import PCA
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _num_samples
import warnings

class CustomSplitKFold(_BaseKFold):
    """K-Fold cross validation with custom splits
    Expects as splits a list of indices that indicate boundaries between runs"""
    def __init__(self, splits=None):
        if splits is None:
            raise ValueError('Need to provide a list of indices for splits')
        else:
            self.splits = splits
            self.n_splits = len(self.splits)

    def _iter_test_indices(self, X, y=None, groups=None):
        num_samples = _num_samples(X)
        indices = np.arange(num_samples)
        split_indices = np.array_split(indices, self.splits)
        for splits in split_indices:
            yield splits


def select_ridge_output(ridge, select_voxels):
    '''Returns a new Ridge object containing only coefs for select_voxels'''
    from copy import deepcopy
    ridge_copy = deepcopy(ridge)
    ridge_copy.coef_ = ridge.coef_[select_voxels]
    ridge_copy.intercept_ = ridge.intercept_[select_voxels]
    if ridge.alpha.size > 1:
        ridge_copy.alpha = ridge.alpha[select_voxels]
    return ridge_copy


def run_split_from_labels(labels, label_occ=3):
    '''Computes when a new run starts from labels
    Returns a list of numbers each number indicating the first sample in each run
    It assumes that in each run each label occurs label_occ times, thus the label_occ+1 time a label is found, a new run has started'''
    from collections import Counter
    counter = Counter()
    run_samples = []
    for i, label in enumerate(labels):
        if counter[label] == label_occ:
            run_samples.append(i)
            counter.clear()
        counter[label] += 1
    return run_samples


def compute_stability(fmri, labels):
    '''Computes stability score for each voxel
    Uses spearman correlation coefficient'''
    from scipy.stats import spearmanr
    from functools import reduce
    run_samples = run_split_from_labels(labels)
    runs_fmri = np.array_split(fmri, run_samples, axis=0)
    runs_labels = np.array_split(labels, run_samples)
    runs_fmri = [run_fmri[np.argsort(run_labels)][None]
                 for run_fmri, run_labels in zip(runs_fmri, runs_labels)]
    try:
        runs_fmri = np.concatenate(runs_fmri)
    except ValueError:
        warnings.warn('One of the runs has less stimuli than the others. Proceeding by aligning number of stimuli.',
                RuntimeWarning)
        shared_stimuli_in_all_runs = reduce(
                np.intersect1d, [np.unique(run_labels) for run_labels in runs_labels])
        runs_fmri = [run_fmri[:, np.isin(run_labels, shared_stimuli_in_all_runs)]
                     for run_labels, run_fmri in zip(runs_labels, runs_fmri)]
        runs_fmri = np.concatenate(runs_fmri)

    stability = np.array([spearmanr(voxel, axis=0)[0][np.triu_indices(len(run_samples))].mean()
                          for voxel in runs_fmri.T])
    return stability


def transform_select_voxels(X, select_voxels=None):
    if select_voxels is None:
        select_voxels = np.arange(X.shape[1])
    return X[:, select_voxels]

def inverse_transform_select_voxels(X_transformed, select_voxels=None):
    if select_voxels is None:
        select_voxels = np.arange(X_transformed.shape[1])
    X = np.zeros((X_transformed.shape[0], select_voxels.shape[0]))
    X[:, select_voxels] = X_transformed
    return X

def mve_score(mve, stimulus, fmri, labels, scoring='new'):
    '''For fitted MultiVoxelEncoding object mve.
    returns the decoding accuracy for stimuli,fmri'''
    catdist = np.zeros((25, 5))
    probdist = np.zeros((25, 25))
    for ilbl, lbl in enumerate(np.unique(labels)):
        lbl_given_r = np.array([
            np.sum(
                mve.score(stimulus[np.where(labels==alt_lbl)[0],:],
                fmri[np.where(labels==lbl)[0],:]))
            for alt_lbl in np.unique(labels)])
        probdist[ilbl,:] = lognormalize2(lbl_given_r)
        catdist[ilbl,:] = lognormalize2(logsumexp(np.reshape(lbl_given_r,(5,-1)), axis=1))
    catpbz = np.zeros(catdist.shape)
    maxes = np.argmax(catdist,axis=1)
    catpbz[np.arange(25),maxes] = 1
    pbz = np.zeros(probdist.shape)
    maxes = np.argmax(probdist,axis=1)
    pbz[np.arange(25),maxes] = 1

    return (np.sum(np.reshape(catpbz,(5,5,5)),1),pbz)

class MultiVoxelEncoding(BaseEstimator):
    def __init__(self, models=Ridge, n_components=0.95, cv=None, scoring=None):
        if cv is not None:
            if not isinstance(n_components,list):
                raise RuntimeError('n_components needs to be a'
                        'list for cross-validation')
            if not all(isinstance(n, int) for n in n_components):
                raise RuntimeError('all elements of n_components need '
                        'to be integers')

        self.models = models
        self.n_components = n_components
        self.cv = cv
        self.scoring = scoring
        self.pca = None
        self.pca_cov = None

    def _pca_transform(self, y):
        pca_y = self.pca.transform(y)
        return pca_y/np.linalg.norm(pca_y)

    def fit(self, X, y, labels=None):
        if isinstance(self.n_components, int):
            pca = PCA(n_components=self.n_components)
            y_pred = self.models.predict(X)
            self.pca = pca.fit(y_pred)
            pca_y_pred, pca_y = (self._pca_transform(pcav) for pcav in [y_pred, y])
            self.pca_cov = np.cov(pca_y - pca_y_pred, rowvar=0)
            if self.pca_cov.shape == ():
                self.pca_cov = self.pca_cov[None, None]
        else:
            if self.scoring is None:
                self.scoring = self.score
            scores_dict = {}
            kfold = StratifiedKFold(n_splits=self.cv)
            for value in self.n_components:
                scores = []
                for train, test in kfold.split(X, labels):
                    cv_mve = MultiVoxelEncoding(models=self.models, n_components=value, cv=None)
                    cv_mve.fit(X[train, :], y[train, :])
                    scores.append(
                            np.sum(np.diag(mve_score(
                                cv_mve, X[test, :], y[test,:], labels[test])[1]))/25.)
                scores_dict[value] = np.mean(scores)
            self.n_components = max(scores_dict, key=scores_dict.get)
            y_pred = self.models.predict(X)
            pca = PCA(n_components=self.n_components)
            pca.fit(y_pred)
            self.pca = pca
            pca_predy, pca_y = (self._pca_transform(pcav) for pcav in [y_pred,y])
            self.pca_cov = np.cov(pca_y - pca_predy,rowvar=0)

    def predict(self, X):
        if self.pca is None:
            raise RuntimeError('must call "fit" first')
        return self._pca_transform(self.models.predict(X))

    def score(self, X, y):
        if self.pca is None:
            raise RuntimeError('must call "fit" first')
        return np.array(
                [multivariate_normal.logpdf(row[:, 0], row[:, 1],
                                            self.pca_cov)
                 for row in np.dstack((self._pca_transform(y), self.predict(X)))])

def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)

def pdf_multi_normal(x, mean, cov):
    '''Multivariate normal pdf with three parts'''
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

def score_diffcat(y_true, y_pred, labels, ctype='pearson'):
    '''Compute the rankscore (Santoro et al., 2014) of the data'''
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    ranks = np.zeros((y_pred.shape[0],))
    for i in xrange(y_pred.shape[0]):
        correlations = [cfunc(y_pred[i,:],y_true[j,:])[0] for j in xrange(y_true.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(y_pred[i,:],y_true[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1])/(len(correlations)))
    return ranks

def binary_retrieval(y_true, y_pred, labels):
    '''Compute the binary retrieval accuracy (Mitchell et al., 2008) for the data'''
    import numpy as np
    def cosine_similarity(a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    match_acc = np.zeros((y_pred.shape[0],))
    for i in xrange(y_pred.shape[0]):
        match = 0.
        for j in xrange(y_pred.shape[0]):
            if labels[j] == labels[i]:
                continue
            score_true = cosine_similarity(y_pred[i,:],y_true[i,:]) + cosine_similarity(y_pred[j,:],y_true[j,:])
            score_false = cosine_similarity(y_pred[i,:],y_true[j,:]) + cosine_similarity(y_pred[j,:],y_true[i,:])
            if score_true > score_false:
                match += 1.
        match_acc[i] = match / np.sum(labels!=labels[i])
    return match_acc

def confmat_from_labels(mve, stimulus, fmri, labels):
    '''Returns a 25x25 confusion matrix'''
    probdist = np.zeros((25, 25))
    unique_labels = [genre + str(i) for genre in ['ambient', 'country', 'metal', 'rocknroll', 'symphonic'] for i in range(5)]
    for ilbl, lbl in enumerate(unique_labels):
        lbl_given_r = np.array([
            np.sum(mve.score(
                stimulus[np.where(labels==alt_lbl)[0],:],
                fmri[np.where(labels==lbl)[0],:])) if alt_lbl in np.unique(labels) else 0.
            for alt_lbl in unique_labels])
        probdist[ilbl,:] = lognormalize2(lbl_given_r)
    confmat = np.zeros((25, 25))
    for i in range(25):
        confmat[i, np.argmax(probdist[i])] = 1
    return confmat

