# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import sys
from os.path import join
import glob
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import joblib
from skimage.util.shape import view_as_windows
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import validation_functions as val

memory = joblib.Memory('~/joblib')

lqmfs_list = glob.glob(os.path.join('/data','forrest_gump','phase1','stimulus',
                                    'task002','features','*.lq_mfs'))

# cut out 500 ms 
feature_dict = {lqmfs_fn.split('/')[-1].split('.')[0]: np.genfromtxt(
    lqmfs_fn, delimiter=',')[:60] for lqmfs_fn in lqmfs_list}

feature_dict = {key.split('_')[0]+key[-1]: value for key, value in feature_dict.items()}

ft_freq = feature_dict.values()[0].shape[1]

@memory.cache
def make_X_y_matrices(subject, preprocessed_path='/data/mboos/pandora/fmri'):
    '''Creates matrices for X and y for a given subject and returns them (as well as the StandardScaler)'''
    fmri = joblib.load(join(preprocessed_path,'fmri_subj_{}.pkl'.format(subject)))
    stimulus_ts = prepare_stimulus(subject, preprocessed_path)
    indices_to_keep = compute_indices_to_keep(stimulus_ts)
    fmri = fmri[indices_to_keep]
    stimulus_ts = stimulus_ts[indices_to_keep]

    scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = scaler.fit_transform(stimulus_ts)
    X[np.isnan(X)] = 0
    y = y_scaler.fit_transform(fmri)
    return X, y, scaler, y_scaler

def lagged_stimulus_idx(subject, preprocessed_path='/data/mboos/pandora/fmri'):
    '''Prepares a lagged matrix of stimulus labels, np.nan indicate no stimulus present'''
    print(preprocessed_path)
    labels = joblib.load(join(preprocessed_path,'labels_subj_{}.pkl'.format(subject)))
    #find indices of 4th consecutive music categories and replace with 'rest'
    rolled_view = view_as_windows(labels, 4)
    indices_of_duplicates = np.where(np.apply_along_axis(lambda x : not (len(np.unique(x))>1 or 'rest' in x),
        1, rolled_view))[0]+3
    labels[indices_of_duplicates] = 'rest'
    rolled_view = view_as_windows(labels, 3)
    music_timeseries = np.full((labels.shape[0], 1), np.nan, dtype=object)
    for stim_name, lqmfs in feature_dict.items():
        is_arr_stim = (rolled_view==stim_name).all(axis=1)
        for stim_idx in np.where(is_arr_stim)[0]:
            music_timeseries[stim_idx:stim_idx+3] = stim_name

    #first add 3 rows of zeros in the beginning (since we are lagging by 3 additional rows)
    n_samples = music_timeseries.shape[0]
    music_timeseries = np.vstack([np.full((3, 1), np.nan), music_timeseries])
    lagged_ts = view_as_windows(music_timeseries, (4, music_timeseries.shape[-1]))
    lagged_ts = np.reshape(np.squeeze(lagged_ts)[:, :-1], (n_samples, -1))
    return lagged_ts

def prepare_stimulus(subject, preprocessed_path='/data/mboos/pandora/fmri'):
    '''Prepares a lagged matrix of LQMFS features'''
    labels = joblib.load(join(preprocessed_path,'labels_subj_{}.pkl'.format(subject)))
    #find indices of 4th consecutive music categories and replace with 'rest'
    rolled_view = view_as_windows(labels, 4)
    indices_of_duplicates = np.where(np.apply_along_axis(lambda x : not (len(np.unique(x))>1 or 'rest' in x),
        1, rolled_view))[0]+3
    labels[indices_of_duplicates] = 'rest'
    rolled_view = view_as_windows(labels, 3)
    music_timeseries = np.full((labels.shape[0], 20*ft_freq), np.nan)
    for stim_name, lqmfs in feature_dict.items():
        is_arr_stim = (rolled_view==stim_name).all(axis=1)
        for stim_idx in np.where(is_arr_stim)[0]:
            music_timeseries[stim_idx:stim_idx+3,:] = np.reshape(lqmfs, (3, -1))

    #first add 3 rows of zeros in the beginning (since we are lagging by 3 additional rows)
    n_samples = music_timeseries.shape[0]
    music_timeseries = np.vstack([np.full((3, 20*ft_freq), np.nan), music_timeseries])
    lagged_ts = view_as_windows(music_timeseries, (4, music_timeseries.shape[-1]))
    lagged_ts = np.reshape(np.squeeze(lagged_ts)[:, :-1], (n_samples, -1))
    return lagged_ts

def compute_indices_to_keep(lagged_ts, stim_prop=0.66):
    '''Computes which indices to remove from the (lagged) stimulus and fmri timeseries.
    Selects timepoints for removal where stimulus is present les than stim_prop proportion of the interval.
    '''
    proportion_nan = np.isnan(lagged_ts).sum(axis=1) / np.float(lagged_ts.shape[1])
    which_to_keep = np.where((1-proportion_nan)>=stim_prop)[0]
    return which_to_keep

def get_ridge_plus_predictions(X, y, subj, alphas=None, preprocessed_path='/data/mboos/pandora/fmri', **kwargs):
    '''8-fold CV with inner CV gridsearch over alphas for Ridge(**kwargs) of X, y.
    Returns concatenated test-set predictions and 8 RidgeCV objects trained on the training folds'''
    indices_to_keep = compute_indices_to_keep(prepare_stimulus(subj, preprocessed_path))
    labels = lagged_stimulus_idx(subj, preprocessed_path)[indices_to_keep,1]
    splits = val.run_split_from_labels(labels)
    kfold = val.CustomSplitKFold(splits)
    if alphas is None:
        alphas = [1000]
    ridges = []
    predictions = []
    ridges = []
    for train, test in kfold.split(X,y):
        ridges.append(ridge_gridsearch_per_target(X[train], y[train], alphas, **kwargs))
        predictions.append(ridges[-1].predict(X[test]))
    return np.vstack(predictions), ridges

def fit_subjects(subj, save_path='/data/mboos/pandora', preprocessed_path='/data/mboos/pandora/fmri', **kwargs):
    from sklearn.metrics import r2_score
    X, y, _, _ = make_X_y_matrices(subj, preprocessed_path)
    predictions, ridges = get_ridge_plus_predictions(X, y, subj, preprocessed_path=preprocessed_path, **kwargs)
    joblib.dump(ridges, join(save_path, 'models', 'ridge_{}.pkl'.format(subj)), compress=9)
    joblib.dump(predictions, join(save_path, 'predictions', 'predicted_ridge_{}.pkl'.format(subj)))
    scores = r2_score(y, predictions, multioutput='raw_values')
    scores[np.var(y, axis=0)==0.] = 0.
    joblib.dump(scores, join(save_path, 'predictions', 'scores', 'scores_ridge_{}.pkl'.format(subj)))

def scores_per_folds_for_subj(subj, preprocessed_path, scorer=None):
    '''Computes the scores of the models saved for this subject
    scorer accepts all sklearn estimators (with multioutput)
    scorer=None (default) uses r2_score'''
    from sklearn.metrics import r2_score
    if scorer is None:
        scorer = r2_score
    X, y, _, _ = make_X_y_matrices(subj, preprocessed_path)
    indices_to_keep = compute_indices_to_keep(prepare_stimulus(subj, preprocessed_path))
    labels = lagged_stimulus_idx(subj, preprocessed_path)[indices_to_keep,1]
    splits = val.run_split_from_labels(labels)
    kfold = val.CustomSplitKFold(splits)
    scores = []
    ridges = joblib.load('/data/mboos/pandora/models/ridge_{}.pkl'.format(subj))
    for i, (train, test) in enumerate(kfold.split(X,y)):
        scores.append(scorer(y[train], ridges[i].predict(X[train]), multioutput='raw_values'))
        scores[-1][np.var(y[test], axis=0)==0.] = 0.
    return scores

def ridge_gridsearch_per_target(X, y, alphas, n_splits=7, **kwargs):
    '''Does gridsearch for alphas on X, y with cv and returns refit Ridge with best alphas'''
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    cv_results = {'alphas': []}
    cv = KFold(n_splits=n_splits)
    for alpha in alphas:
        scores = []
        for train, test in cv.split(X, y):
            ridge = Ridge(alpha=alpha, **kwargs)
            scores.append(mean_squared_error(y[test], ridge.fit(X[train], y[train]).predict(X[test]),
                              multioutput='raw_values'))
        scores = np.vstack(scores).mean(axis=0)
        cv_results['alphas'].append(scores)
    cv_results['alphas'] = np.vstack(cv_results['alphas'])
    best_alphas = np.array(alphas)[np.argmin(cv_results['alphas'], axis=0)]
    return Ridge(alpha=best_alphas, **kwargs).fit(X, y)

if __name__=='__main__':
    if len(sys.argv) == 1:
        dataset_keys = ['7T', '3T']
    else:
        dataset_keys = sys.argv[1:]

    subjects = {'7T': range(1,20),
                '3T': range(1,19)}
    # folder with preprocessed fMRI data
    preprocessed_path = {'7T': '/data/mboos/pandora/fmri',
                         '3T': '/data/mboos/pandora/fmri_3T'}

    # folder where encoding models and predictions should be saved
    save_path = {'7T': '/data/mboos/pandora',
                 '3T': '/data/mboos/pandora/3T'}

    for dataset in dataset_keys:
        joblib.Parallel(n_jobs=4)(
                joblib.delayed(fit_subjects)(
                    subj, alphas=[1000, 10000.0, 100000.0, 1000000.0], n_splits=7,
                    preprocessed_path=preprocessed_path[dataset],
                    save_path=save_path[dataset])
                for subj in subjects[dataset])
