
# coding: utf-8

# In[56]:

import mvpa2.suite as mvpa
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import ElasticNetCV,BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from pandas import read_csv
from sklearn.externals import joblib
from encoding_helpers import *
from __future__ import division
from itertools import combinations
import pylab as plt



# In[345]:

Ns = [250,500,1000,2500,5000,10000]
T3 = True
matches_3t_n = []
for n in Ns:
    matches_3t_subjs = []
    for subj in xrange(1,20):
        matches_3t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            matches_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/stable_rank_'+pattern+str(n)+'_' + identifier+'.pkl'))
        matches_3t_subjs.append(matches_3t_tmp)
    matches_3t_n.append(matches_3t_subjs)
    
T3 = False
matches_7t_n = []
for n in Ns:
    matches_7t_subjs = []
    for subj in xrange(1,20):
        matches_7t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            matches_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/stable_rank_'+pattern+str(n)+'_' + identifier+'.pkl'))
        matches_7t_subjs.append(matches_7t_tmp)
    matches_7t_n.append(matches_7t_subjs)
 


# In[171]:

def melt_to_df(array_to_melt,colnames):
    '''Melt array into dataframe'''
    import pandas as pd
    assert(len(colnames)==len(array_to_melt.shape))
    shapes = [1] + list(array_to_melt.shape) + [1]
    melted_df = pd.DataFrame()
    #rvcn_list = list(reversed(list(enumerate(colnames))))
    melted_df['data'] = array_to_melt.flatten()
    for i,name in enumerate(colnames):
        melted_df[name] = np.tile(np.repeat(np.arange(shapes[i+1]),np.cumprod(shapes[i+2:])[-1]),np.cumprod(shapes[:i+1])[-1])
    return melted_df


# In[302]:

def prepare_for_plotting(to_melt_list,colnames,Ns):
    '''first 3T, second 7T in melt_list'''
    melted_3T = melt_to_df(to_melt_list[0],colnames)
    melted_7T = melt_to_df(to_melt_list[1],colnames)
    melted_3T['Field strength'] = ['3T']*melted_3T.shape[0]
    melted_7T['Field strength'] = ['7T']*melted_7T.shape[0]
    melted_df = melted_7T.append(melted_3T)
    melted_df = melted_df.replace({'voxelnr':{ x:n for x,n in enumerate(Ns)}})
    return melted_df


# In[468]:

def prob_to_acc_for_array(prob_mat):
    '''Applies prob_to_acc to the last two axes with 25x25 elements'''
    shapes = prob_mat.shape
    prob_mat = prob_mat[:].reshape((-1,25,25))
    prob_mat_red = np.zeros((prob_mat.shape[0]))
    for i in xrange(np.cumprod(prob_mat_red.shape)[-1]):
        prob_mat_red[i] = prob_to_acc(prob_mat[i])
    return prob_mat_red.reshape(shapes[:-2])


# In[346]:

Ns = [250,500,1000,2500,5000,10000]
T3 = True
bin_3t_n = []
for n in Ns:
    bin_3t_subjs = []
    for subj in xrange(1,20):
        bin_3t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            bin_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/stable_bin_'+pattern+str(n)+'_' + identifier+'.pkl'))
        bin_3t_subjs.append(bin_3t_tmp)
    bin_3t_n.append(bin_3t_subjs)
bin_3t_n = np.array(bin_3t_n)    
T3 = False
bin_7t_n = []
for n in Ns:
    bin_7t_subjs = []
    for subj in xrange(1,20):
        bin_7t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            bin_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/stable_bin_'+pattern+str(n)+'_' + identifier+'.pkl'))
        bin_7t_subjs.append(bin_7t_tmp)
    bin_7t_n.append(bin_7t_subjs)
bin_7t_n = np.array(bin_7t_n)


# In[347]:

Ns = [250,500,1000,2500,5000,10000]
T3 = True
bin_3t_corr_n = []
for n in Ns:
    bin_3t_subjs = []
    for subj in xrange(1,20):
        bin_3t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            bin_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_bin_'+pattern+str(n)+'_' + identifier+'.pkl'))
        bin_3t_subjs.append(bin_3t_tmp)
    bin_3t_corr_n.append(bin_3t_subjs)
bin_3t_corr_n = np.array(bin_3t_corr_n)    

T3 = False
bin_7t_corr_n = []
for n in Ns:
    bin_7t_subjs = []
    for subj in xrange(1,20):
        bin_7t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            bin_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_bin_'+pattern+str(n)+'_' + identifier+'.pkl'))
        bin_7t_subjs.append(bin_7t_tmp)
    bin_7t_corr_n.append(bin_7t_subjs)
bin_7t_corr_n = np.array(bin_7t_corr_n)


# In[348]:

Ns = [250,500,1000,2500,5000,10000]
T3 = True
match_3t_corr_n = []
for n in Ns:
    match_3t_subjs = []
    for subj in xrange(1,20):
        match_3t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            match_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_rank_'+pattern+str(n)+'_' + identifier+'.pkl'))
        match_3t_subjs.append(match_3t_tmp)
    match_3t_corr_n.append(match_3t_subjs)
match_3t_corr_n = np.array(match_3t_corr_n)    

T3 = False
match_7t_corr_n = []
for n in Ns:
    match_7t_subjs = []
    for subj in xrange(1,20):
        match_7t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            match_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_rank_'+pattern+str(n)+'_' + identifier+'.pkl'))
        match_7t_subjs.append(match_7t_tmp)
    match_7t_corr_n.append(match_7t_subjs)
match_7t_corr_n = np.array(match_7t_corr_n)


# In[396]:

Ns = [250,500,1000,2500,5000,10000]
T3 = True
probs_3t_n = []
for n in Ns:
    probs_3t_subjs = []
    for subj in xrange(1,20):
        probs_3t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            probs_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/stable_probs_'+pattern+str(n)+'_' + identifier+'.pkl'))
        probs_3t_subjs.append(probs_3t_tmp)
    probs_3t_n.append(probs_3t_subjs)
probs_3t_n = np.array(probs_3t_n)    
T3 = False
probs_7t_n = []
for n in Ns:
    probs_7t_subjs = []
    for subj in xrange(1,20):
        probs_7t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            probs_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/stable_probs_'+pattern+str(n)+'_' + identifier+'.pkl'))
        probs_7t_subjs.append(probs_7t_tmp)
    probs_7t_n.append(probs_7t_subjs)
probs_7t_n = np.array(probs_7t_n)


# In[696]:

Ns = [250,500,1000,2500,5000,10000]
T3 = True
probs_3t_corr_n = []
for n in Ns:
    probs_3t_subjs = []
    for subj in xrange(1,20):
        probs_3t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            probs_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_probs_'+pattern+str(n)+'_' + identifier+'.pkl'))
        probs_3t_subjs.append(probs_3t_tmp)
    probs_3t_corr_n.append(probs_3t_subjs)
probs_3t_corr_n = np.array(probs_3t_corr_n)    
T3 = False
probs_7t_corr_n = []
for n in Ns:
    probs_7t_subjs = []
    for subj in xrange(1,20):
        probs_7t_tmp = []
        for run in xrange(8):
            identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
            probs_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_probs_'+pattern+str(n)+'_' + identifier+'.pkl'))
        probs_7t_subjs.append(probs_7t_tmp)
    probs_7t_corr_n.append(probs_7t_subjs)
probs_7t_corr_n = np.array(probs_7t_corr_n)


# In[580]:

probs_3t_tmp = []
for run in xrange(8):
    identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[True],run,1)
    probs_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/corr_probs_'+pattern+str(n)+'_' + identifier+'.pkl'))


# In[494]:

T3 = True
dec_3t_n = []

for subj in xrange(1,20):
    probs_3t_tmp = []
    for run in xrange(8):
        identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
        probs_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/discdec_decoder_'+ identifier+'.pkl'))
    dec_3t_n.append(probs_3t_tmp)
dec_3t_n = np.array(dec_3t_n)
    
T3 = False
dec_7t_n = []
for subj in xrange(1,20):
    probs_7t_tmp = []
    for run in xrange(8):
        identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
        probs_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/discdec_decoder_' + identifier+'.pkl'))
    dec_7t_n.append(probs_7t_tmp)
dec_7t_n = np.array(dec_7t_n)


# In[538]:

T3 = True
dec_svc_3t_n = []

for subj in xrange(1,20):
    probs_3t_tmp = []
    for run in xrange(8):
        identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
        probs_3t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/discdec_svc_decoder_'+ identifier+'.pkl'))
    dec_svc_3t_n.append(probs_3t_tmp)
dec_svc_3t_n = np.array(dec_svc_3t_n)
    
T3 = False
dec_svc_7t_n = []
for subj in xrange(1,20):
    probs_7t_tmp = []
    for run in xrange(8):
        identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],run,subj)
        probs_7t_tmp.append(joblib.load('/home/mboos/SpeechEncoding/validation/discdec_svc_decoder_' + identifier+'.pkl'))
    dec_svc_7t_n.append(probs_7t_tmp)
dec_svc_7t_n = np.array(dec_svc_7t_n)


# In[471]:

corr_bin_melted = prepare_for_plotting([np.mean(bin_3t_corr_n,axis=-1),np.mean(bin_7t_corr_n,axis=-1)],colnames[:-1],Ns)
corr_match_melted = prepare_for_plotting([np.mean(match_3t_corr_n,axis=-1),np.mean(match_7t_corr_n,axis=-1)],colnames[:-1],Ns)
corr_probs_melted = prepare_for_plotting([prob_to_acc_for_array(probs_3t_corr_n),prob_to_acc_for_array(probs_7t_corr_n)],colnames[:-1],Ns)

stable_match_melted = prepare_for_plotting([np.mean(matches_3t_n,axis=-1),np.mean(matches_7t_n,axis=-1)],colnames[:-1],Ns)
stable_bin_melted = prepare_for_plotting([np.mean(bin_3t_n,axis=-1),np.mean(bin_7t_n,axis=-1)],colnames[:-1],Ns)
stable_probs_melted = prepare_for_plotting([prob_to_acc_for_array(probs_3t_n),prob_to_acc_for_array(probs_7t_n)],colnames[:-1],Ns)

corr_bin_melted['selection'] = corr_bin_melted.shape[0]*['r^2']
corr_match_melted['selection'] = corr_match_melted.shape[0]*['r^2']
corr_probs_melted['selection'] = corr_probs_melted.shape[0]*['r^2']
stable_bin_melted['selection'] = stable_bin_melted.shape[0]*['stability']
stable_match_melted['selection'] = stable_match_melted.shape[0]*['stability']
stable_probs_melted['selection'] = stable_probs_melted.shape[0]*['stability']

bin_melted = stable_bin_melted.append(corr_bin_melted)
match_melted = stable_match_melted.append(corr_match_melted)
probs_melted = stable_probs_melted.append(corr_probs_melted)


# In[384]:

plt.figure(figsize=(14,12))
sns.set_style(style='whitegrid')
g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=match_melted)
sns.despine(left=True)
for axes in g.axes.flat:
    axes.set_xlabel('Number of voxels')
g.axes.flat[0].set_title('Most stable voxels')
g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
g.axes.flat[0].set_ylabel('Matching rank score')
#plt.savefig('Nr_of_voxels_matching_score_selection.svg')


# In[378]:

plt.figure(figsize=(10,8))
sns.set_style(style='whitegrid')
g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=bin_melted)
sns.despine(left=True)
for axes in g.axes.flat:
    axes.set_xlabel('Number of voxels')
    
g.axes.flat[0].set_title('Most stable voxels')
g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
g.axes.flat[0].set_ylabel('Binary rank score')
plt.savefig('Nr_of_voxels_binary_score_selection.svg')


# In[543]:

plt.figure(figsize=(10,8))
sns.set_style(style='whitegrid')
g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=probs_melted)
sns.despine(left=True)
for i,axes in enumerate(g.axes.flat):
    axes.set_xlabel('Number of voxels')
    axes.plot([-0.5,5.5],2*[np.mean(discdec_7t_acc)],linestyle="--")
    axes.plot([-0.5,5.5],2*[np.mean(discdec_3t_acc)],linestyle="--")
g.axes.flat[0].set_title('Most stable voxels')
g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
g.axes.flat[0].set_ylabel('Decoding accuracy of music category')

plt.savefig('Nr_of_voxels_decoding_accuracy_selection.svg')

