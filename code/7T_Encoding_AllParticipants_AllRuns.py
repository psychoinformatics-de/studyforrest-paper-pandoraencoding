# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
from __future__ import division
import mvpa2.suite as mvpa
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV,BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from pandas import read_csv
from sklearn.externals import joblib
from encoding_helpers import *

from itertools import combinations

T3 = False


for subj in xrange(12,13):
    
    	
    	
    subj_preprocessed_path = os.path.join('/home','mboos','SpeechEncoding','PreProcessed','subj%02dnpp.gzipped.hdf5' % subj)
    	
    s1ds = mvpa.h5load(subj_preprocessed_path)
    	
    	# <codecell>
    	
    events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)
    	
    	# <markdowncell>
    	
    	# ## High- and Low-pass filtering of data
    
    	# <codecell>
    
    	#TODO
    	
    	# <markdowncell>
    	
    	# ## Load LCQFT (from essentia)
    	# can and should be changed to different stimulus representations later
    	
    	# <codecell>
    	
    lqmfs_list = glob.glob(os.path.join('/home','data','psyinf','forrest_gump','anondata','stimulus','task002','features','*.lq_mfs'))
    feature_dict = {lqmfs_fn.split('/')[-1].split('_')[0] + lqmfs_fn.split('/')[-1].split('_')[1][2]:np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list}
    ft_freq = feature_dict.values()[0].shape[1]
    	
    	
    	#create representation with rolling window of length 4
    rvstr_TS = rolling_window(s1ds.sa['targets'][::-1].copy(),4)
    	
    	#find indices of 4th consecutive music categories and replace with 'rest'
    s1ds.sa['targets'].value[(np.where(np.apply_along_axis(lambda x : len(np.unique(x)) == 1 and x[0] != 'rest',1,rvstr_TS)[::-1])[0]+3)] = 'rest'
    labelsTS = s1ds.sa['targets'].value.copy()
    	
    	# <codecell>
    	
    	#unroll audio features
    	#cut last 500ms
    featureTS = np.zeros((labelsTS.shape[0],20*ft_freq))
    featureTS[labelsTS!='rest',:] = np.reshape(np.vstack([feature_dict[ev['targets']][:60,:] for ev in events if ev['targets']!='rest']),(-1,ft_freq*20))
    	
    	# <codecell>
    	
    	#now lag the audiofeatures
    	
    	#first add 3 rows of zeros in the beginning (since we are lagging by 3 additional rows)
    tmp_ft = np.vstack((np.zeros((3,20*ft_freq)),featureTS))
    	
    	#lagging by stride tricks
    strides = (tmp_ft.strides[0],tmp_ft.strides[0],tmp_ft.strides[1])
    shape = (tmp_ft.shape[0]-3,4,tmp_ft.shape[1])
    lagged_TS = np.lib.stride_tricks.as_strided(tmp_ft[::-1,:].copy(),shape=shape,strides=strides)
    lagged_TS = lagged_TS[::-1,:,:]
    
    #kick out the audio features presented at the timepoint to which the index corresponds --> too recent for any BOLD
    lagged_TS = np.reshape(lagged_TS[:,1:,:],(lagged_TS.shape[0],-1))
    
    #ONLY KICK OUT NONZEROS AND MISCLASSIFIEDS
    nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
    #now only use these that have non-zero elements
    nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
    zero_prop = np.array([np.sum(lagged_TS[i,:]==0)/lagged_TS.shape[1] for i in xrange(lagged_TS.shape[0])])
    tla = np.logical_and(zero_prop >= 2/3,np.all(lagged_TS[:,:np.trunc((2/3)*lagged_TS.shape[1])]==0,axis=1))
    tla = np.logical_and(tla,nonz_eles)
    whita = np.logical_and(tla,s1ds.sa['targets'].value!='rest')
    nonz_eles = zero_prop < 2/3
    nonz_eles = np.logical_and(np.logical_not(whita),nonz_eles)
    
    
    #kick out timepoints
    voxelTS = s1ds.samples[nonz_eles]
    lagged_TS = lagged_TS[nonz_eles,:]
    
    #exclude_run = np.random.randint(0,8)
    scaler = StandardScaler()
    y_scaler = StandardScaler()
    #nkeep = 500
    #only for the nkeep most stable voxels
    
    #get the most stable voxels
    #voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in voxelTS[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles],:].T ]
    #keep the 500 most stable voxels
    #most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]
    X = scaler.fit_transform(lagged_TS)
    y = y_scaler.fit_transform(voxelTS)
    data = {'X':X,'y':y,'nonz_eles':nonz_eles,'yscaler':y_scaler,'X_scaler':scaler,'zero_prop':zero_prop,'chunks':s1ds.sa['chunks'].value}
    joblib.dump(data,os.path.join('/home','mboos','SpeechEncoding','run_data','on_data_ridge_lcmfs{0}_subj_{1}.pkl'.format({True : '_3T', False : '_7T'}[T3],subj)),compress=3)

    
    for exclude_run in xrange(4,8):
        identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],exclude_run,subj)
        X_val = X[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles],:]
        y_val = y[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles],:]
        X_train = X[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles],:]
        y_train = y[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles],:]    
        
    #    y = y_scaler.fit_transform(voxelTS[:,most_stable_vx_idx])
    #    X_val = X[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles],:]
    #    y_val = y[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles],:]
    #    X_train = X[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles],:]
    #    y_train = y[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles],:]
        
        # <markdowncell>
        
        # Prepare for fitting using condor
        
        # <codecell>
        
        #save original data
        #data = {'Xtr':X_train,'ytr':y_train,'Xval':X_val,'yval':y_val,'excl_run':exclude_run,'nonz_eles':nonz_eles,'yscaler':y_scaler,'X_scaler':scaler,'zero_prop':zero_prop}
        #joblib.dump(data,os.path.join('/home','mboos','SpeechEncoding','joblibdumps','input_bayridge_subj' + str(subj) + '_LC_MFS_7T.pkl'))
        #joblib.dump(data,os.path.join('/home','mboos','SpeechEncoding','run_data','data_' + identifier + '.pkl'),compress=3)
               
        
        # <codecell>
        
        #partition data and submit to condor
        #MUE_helper_br(X_train,y_train,'bayridge_lcmfs_subj_%d_7T' % subj)
        MUE_helper2(X_train,y_train,identifier,nsplit=11)
