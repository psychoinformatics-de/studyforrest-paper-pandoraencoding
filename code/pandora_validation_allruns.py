#Validation all participantsimport mvpa2.suite as mvpa
from __future__ import division
import os

def submitfile_string(name):
	submstring = '''Executable = /usr/bin/python
Universe = vanilla
initialdir = /home/mboos/SpeechEncoding
request_cpus = 1
request_memory = 8000
getenv = True
kill_sig = 2
when_to_transfer_output = ON_EXIT_OR_EVICT
environment = PYTHONPATH=/usr/lib/python2.7
Arguments = -- /home/data/scratch/mboos/starter/''' + name + ''' error  = /home/mboos/SpeechEncoding/CondorLogFiles/$(PROCESS).$(CLUSTER).err output = /home/mboos/SpeechEncoding/CondorLogFiles/$(PROCESS).$(CLUSTER).out log = /home/mboos/SpeechEncoding/CondorLogFiles/$(PROCESS).$(CLUSTER).log
queue'''
	return submstring

def score_cat_perm_labels_file_string(ds_path,input_name,model_name,output_name):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)
    
def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1]-1)/(pred_val.shape[0]-1))
    return ranks
    
s1ds = mvpa.h5load('/home/mboos/SpeechEncoding/PreProcessed/''' + ds_path + '''')
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)
lqmfs_list = glob.glob(os.path.join('/home','data','psyinf','forrest_gump','anondata','stimulus','task002','features','*.lq_mfs'))
feature_dict = {lqmfs_fn.split('/')[-1].split('_')[0] + lqmfs_fn.split('/')[-1].split('_')[1][2]:np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list}
ft_freq = feature_dict.values()[0].shape[1]
#create representation with rolling window of length 4
rvstr_TS = rolling_window(s1ds.sa['targets'][::-1].copy(),4)

#find indices of 4th consecutive music categories and replace with 'rest'
s1ds.sa['targets'].value[(np.where(np.apply_along_axis(lambda x : len(np.unique(x)) == 1 and x[0] != 'rest',1,rvstr_TS)[::-1])[0]+3)] = 'rest'
labelsTS = s1ds.sa['targets'].value.copy()
#unroll audio features
#cut last 500ms
featureTS = np.zeros((labelsTS.shape[0],20*ft_freq))
featureTS[labelsTS!='rest',:] = np.reshape(np.vstack([feature_dict[ev['targets']][:60,:] for ev in events if ev['targets']!='rest']),(-1,ft_freq*20))

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
#now only use these that have non-zero elements
nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
#kick out timepoints
voxelTS = s1ds.samples[nonz_eles]
data = joblib.load('/home/mboos/SpeechEncoding/joblibdumps/''' + input_name + '''')
exclude_run = data['excl_run']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/joblibdumps/'''+model_name+''''+str(i)+'.pkl')) for i in xrange(10)])
targets_val = s1ds.sa['targets'][np.logical_and(s1ds.sa['chunks'].value==exclude_run,nonz_eles)]
categ = "rest"

#get the most stable voxels
voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = 500
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

for i,cat in enumerate(targets_val):
    if cat != 'rest':
        if categ == 'rest' or categ != cat[:-1]:
            categ = cat[:-1]
        targets_val[i] = categ
    else:
        targets_val[i] = categ

joblib.dump(score_diffcat(data['Xval'],data['yval'][:,most_stable_vx_idx][np.random.permutation(data['yval'].shape[0]),:],models[most_stable_vx_idx],targets_val),'/home/mboos/SpeechEncoding/joblibdumps/permuted_stable_'''+output_name + '''')
joblib.dump(score_diffcat(data['Xval'],data['yval'][np.random.permutation(data['yval'].shape[0]),:],models,targets_val),'/home/mboos/SpeechEncoding/joblibdumps/permuted'''+output_name + '''')'''
	return ff_string

def score_cat_n_file_string(ds_path,input_name,model_name,output_name,vxl_nr):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1]-1)/(pred_val.shape[0]-1))
    return ranks

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

s1ds = mvpa.h5load('/home/mboos/SpeechEncoding/PreProcessed/''' + ds_path + '''')
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)
lqmfs_list = glob.glob(os.path.join('/home','data','psyinf','forrest_gump','anondata','stimulus','task002','features','*.lq_mfs'))
feature_dict = {lqmfs_fn.split('/')[-1].split('_')[0] + lqmfs_fn.split('/')[-1].split('_')[1][2]:np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list}
ft_freq = feature_dict.values()[0].shape[1]
#create representation with rolling window of length 4
rvstr_TS = rolling_window(s1ds.sa['targets'][::-1].copy(),4)

#find indices of 4th consecutive music categories and replace with 'rest'
s1ds.sa['targets'].value[(np.where(np.apply_along_axis(lambda x : len(np.unique(x)) == 1 and x[0] != 'rest',1,rvstr_TS)[::-1])[0]+3)] = 'rest'
labelsTS = s1ds.sa['targets'].value.copy()
#unroll audio features
#cut last 500ms
featureTS = np.zeros((labelsTS.shape[0],20*ft_freq))
featureTS[labelsTS!='rest',:] = np.reshape(np.vstack([feature_dict[ev['targets']][:60,:] for ev in events if ev['targets']!='rest']),(-1,ft_freq*20))

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
#now only use these that have non-zero elements
nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
#kick out timepoints
voxelTS = s1ds.samples[nonz_eles]
data = joblib.load('/home/mboos/SpeechEncoding/joblibdumps/''' + input_name + '''')
exclude_run = data['excl_run']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/joblibdumps/'''+model_name+''''+str(i)+'.pkl')) for i in xrange(10)])
targets_val = s1ds.sa['targets'][np.logical_and(s1ds.sa['chunks'].value==exclude_run,nonz_eles)]
categ = "rest"

#get the most stable voxels
voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the vxl_nr most stable voxels
nkeep = ''' + str(vxl_nr) + '''
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

for i,cat in enumerate(targets_val):
    if cat != 'rest':
        if categ == 'rest' or categ != cat[:-1]:
            categ = cat[:-1]
        targets_val[i] = categ
    else:
        targets_val[i] = categ

joblib.dump(score_diffcat(data['Xval'],data['yval'][:,most_stable_vx_idx],models[most_stable_vx_idx],targets_val),'/home/mboos/SpeechEncoding/joblibdumps/'''+output_name + '''')'''
	return ff_string

def score_cat_bin_n_file_string(ds_path,input_name,model_name,output_name,vxl_nr,run):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1]-1)/(pred_val.shape[0]-1))
    return ranks

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

s1ds = mvpa.h5load('/home/mboos/SpeechEncoding/PreProcessed/''' + ds_path + '''')
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)

data = joblib.load('/home/mboos/SpeechEncoding/run_data/''' + input_name + '''')
exclude_run = ''' + run + '''
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/'''+model_name+''''+str(i)+'.pkl')) for i in xrange(10)])
targets_val = s1ds.sa['targets'][np.logical_and(s1ds.sa['chunks'].value==exclude_run,nonz_eles)]
categ = "rest"

#get the most stable voxels
voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the vxl_nr most stable voxels
nkeep = ''' + str(vxl_nr) + '''
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

for i,cat in enumerate(targets_val):
    if cat != 'rest':
        if categ == 'rest' or categ != cat[:-1]:
            categ = cat[:-1]
        targets_val[i] = categ
    else:
        targets_val[i] = categ

joblib.dump(score_diffcat(data['Xval'],data['yval'][:,most_stable_vx_idx],models[most_stable_vx_idx],targets_val),'/home/mboos/SpeechEncoding/joblibdumps/'''+output_name + '''')'''
	return ff_string

def score_cat_file_string(ds_path,input_name,model_name,output_name):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1])/(len(correlations)))
    return ranks

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

s1ds = mvpa.h5load('/home/mboos/SpeechEncoding/PreProcessed/''' + ds_path + '''')
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)
lqmfs_list = glob.glob(os.path.join('/home','data','psyinf','forrest_gump','anondata','stimulus','task002','features','*.lq_mfs'))
feature_dict = {lqmfs_fn.split('/')[-1].split('_')[0] + lqmfs_fn.split('/')[-1].split('_')[1][2]:np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list}
ft_freq = feature_dict.values()[0].shape[1]
#create representation with rolling window of length 4
rvstr_TS = rolling_window(s1ds.sa['targets'][::-1].copy(),4)

#find indices of 4th consecutive music categories and replace with 'rest'
s1ds.sa['targets'].value[(np.where(np.apply_along_axis(lambda x : len(np.unique(x)) == 1 and x[0] != 'rest',1,rvstr_TS)[::-1])[0]+3)] = 'rest'
labelsTS = s1ds.sa['targets'].value.copy()
#unroll audio features
#cut last 500ms
featureTS = np.zeros((labelsTS.shape[0],20*ft_freq))
featureTS[labelsTS!='rest',:] = np.reshape(np.vstack([feature_dict[ev['targets']][:60,:] for ev in events if ev['targets']!='rest']),(-1,ft_freq*20))

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
#now only use these that have non-zero elements
nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
#kick out timepoints
voxelTS = s1ds.samples[nonz_eles]
data = joblib.load('/home/mboos/SpeechEncoding/joblibdumps/''' + input_name + '''')
exclude_run = data['excl_run']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/joblibdumps/'''+model_name+''''+str(i)+'.pkl')) for i in xrange(10)])
targets_val = s1ds.sa['targets'][np.logical_and(s1ds.sa['chunks'].value==exclude_run,nonz_eles)]
categ = "rest"

#get the most stable voxels
voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = 500
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

for i,cat in enumerate(targets_val):
    if cat != 'rest':
        if categ == 'rest' or categ != cat[:-1]:
            categ = cat[:-1]
        targets_val[i] = categ
    else:
        targets_val[i] = categ

joblib.dump(score_diffcat(data['Xval'],data['yval'][:,most_stable_vx_idx],models[most_stable_vx_idx],targets_val),'/home/mboos/SpeechEncoding/joblibdumps/stable_'''+output_name + '''')
joblib.dump(score_diffcat(data['Xval'],data['yval'],models,targets_val),'/home/mboos/SpeechEncoding/joblibdumps/'''+output_name + '''')'''
	return ff_string


def score_cat_nolab_file_string(ds_path,input_name,model_name,output_name):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1])/(len(correlations)))
    return ranks

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

s1ds = mvpa.h5load('/home/mboos/SpeechEncoding/PreProcessed/''' + ds_path + '''')
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)
lqmfs_list = glob.glob(os.path.join('/home','data','psyinf','forrest_gump','anondata','stimulus','task002','features','*.lq_mfs'))
feature_dict = {lqmfs_fn.split('/')[-1].split('_')[0] + lqmfs_fn.split('/')[-1].split('_')[1][2]:np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list}
ft_freq = feature_dict.values()[0].shape[1]
#create representation with rolling window of length 4
rvstr_TS = rolling_window(s1ds.sa['targets'][::-1].copy(),4)

#find indices of 4th consecutive music categories and replace with 'rest'
s1ds.sa['targets'].value[(np.where(np.apply_along_axis(lambda x : len(np.unique(x)) == 1 and x[0] != 'rest',1,rvstr_TS)[::-1])[0]+3)] = 'rest'
labelsTS = s1ds.sa['targets'].value.copy()
#unroll audio features
#cut last 500ms
featureTS = np.zeros((labelsTS.shape[0],20*ft_freq))
featureTS[labelsTS!='rest',:] = np.reshape(np.vstack([feature_dict[ev['targets']][:60,:] for ev in events if ev['targets']!='rest']),(-1,ft_freq*20))

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
#now only use these that have non-zero elements
nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
#kick out timepoints
voxelTS = s1ds.samples[nonz_eles]
data = joblib.load('/home/mboos/SpeechEncoding/joblibdumps/''' + input_name + '''')
exclude_run = data['excl_run']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/joblibdumps/'''+model_name+''''+str(i)+'.pkl')) for i in xrange(10)])

#get the most stable voxels
voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = 500
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]
influence_val = (s1ds.sa['targets'][nonz_eles])
categ = "rest"
for i,cat in enumerate(influence_val):
	if cat != 'rest':
		if categ == 'rest' or categ != cat:
			categ = cat
		influence_val[i] = categ
	else:
		influence_val[i] = categ
valset_lbls = influence_val[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles]]
joblib.dump(score_diffcat(data['Xval'],data['yval'][:,most_stable_vx_idx],models[most_stable_vx_idx],valset_lbls),'/home/mboos/SpeechEncoding/joblibdumps/stable_'''+output_name + '''')
'''
	return ff_string

def score_cat_probs2_file_string(ds_path,input_name,model_name,output_name):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations
from sklearn.decomposition import PCA


def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)


def pdf_multi_normal(x,mean,cov):
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

n_comp = 5

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

s1ds = mvpa.h5load('/home/mboos/SpeechEncoding/PreProcessed/''' + ds_path + '''')
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)
lqmfs_list = glob.glob(os.path.join('/home','data','psyinf','forrest_gump','anondata','stimulus','task002','features','*.lq_mfs'))
feature_dict = {lqmfs_fn.split('/')[-1].split('_')[0] + lqmfs_fn.split('/')[-1].split('_')[1][2]:np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list}
ft_freq = feature_dict.values()[0].shape[1]
#create representation with rolling window of length 4
rvstr_TS = rolling_window(s1ds.sa['targets'][::-1].copy(),4)

#find indices of 4th consecutive music categories and replace with 'rest'
s1ds.sa['targets'].value[(np.where(np.apply_along_axis(lambda x : len(np.unique(x)) == 1 and x[0] != 'rest',1,rvstr_TS)[::-1])[0]+3)] = 'rest'
labelsTS = s1ds.sa['targets'].value.copy()
#unroll audio features
#cut last 500ms
featureTS = np.zeros((labelsTS.shape[0],20*ft_freq))
featureTS[labelsTS!='rest',:] = np.reshape(np.vstack([feature_dict[ev['targets']][:60,:] for ev in events if ev['targets']!='rest']),(-1,ft_freq*20))

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
#now only use these that have non-zero elements
nonz_eles = np.logical_not(np.all(lagged_TS==0,axis=1))
#kick out timepoints
voxelTS = s1ds.samples[nonz_eles]
data = joblib.load('/home/mboos/SpeechEncoding/joblibdumps/''' + input_name + '''')
exclude_run = data['excl_run']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/joblibdumps/'''+model_name+''''+str(i)+'.pkl')) for i in xrange(10)])

influence_val = (s1ds.sa['targets'][nonz_eles])
categ = "rest"
for i,cat in enumerate(influence_val):
	if cat != 'rest':
		if categ == 'rest' or categ != cat:
			categ = cat
		influence_val[i] = categ
	else:
		influence_val[i] = categ
valset_lbls = influence_val[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles]]

voxel_corr_vals = [ r.score(data['Xtr'],data['ytr'][:,j]) for j,r in enumerate(models) ]
nkeep = 500 #np.trunc(data['ytr'].shape[1]*0.1)
best_fit_vx_idx = np.argsort(voxel_corr_vals)[-nkeep:]
pca = PCA(n_components=n_comp)
pred_tr = np.hstack([r.predict(data['Xtr'])[:,None] for r in models])
pred_val = np.hstack([r.predict(data['Xval'])[:,None] for r in models])
pca.fit(pred_tr)
pca_tr = pca.transform(pred_tr)
pca_ytr = pca.transform(data['ytr'])
pca_val = pca.transform(pred_val)
pca_yval = pca.transform(data['yval'])

pca_tr,pca_ytr,pca_val,pca_yval = (pcav/np.linalg.norm(pcav) for pcav in [pca_tr,pca_ytr,pca_val,pca_yval])

pca_cov = np.cov(pca_ytr - pca_tr,rowvar=0)
probdist = np.zeros((25,25))
for ilbl,lbl in enumerate(np.unique(valset_lbls)):
    lbl_given_r = np.array([ np.sum([np.log(pdf_multi_normal(response,stimulus,pca_cov)) for stimulus,response in [(pca_val[alt_lbl_row,:],pca_yval[lbl_row,:]) for alt_lbl_row,lbl_row in zip(np.where(valset_lbls==alt_lbl)[0],np.where(valset_lbls==lbl)[0])]]) for alt_lbl in np.unique(valset_lbls)])
    probdist[ilbl,:] = lognormalize2(lbl_given_r)

joblib.dump(probdist,'/home/mboos/SpeechEncoding/joblibdumps/'''+output_name + '''')
'''
	return ff_string
 
def score_cat_bin_nrvx_file_string(ds_path,data_fn,identifier,output_prefix,nvx,vxl_corr=False):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import sys
sys.path.append('/home/mboos/SpeechEncoding')
import os
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations
from sklearn.decomposition import PCA
from encoding_helpers import *

def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)


def pdf_multi_normal(x,mean,cov):
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1])/(len(correlations)))
    return ranks

def binary_retrieval(X_val,y_val,models,labels):
    import numpy as np
    def cosine_similarity(a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])
    match_acc = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        match = 0
        for j in xrange(pred_val.shape[0]):
            if labels[j] == labels[i]:
                continue
            score_true = cosine_similarity(pred_val[i,:],y_val[i,:]) + cosine_similarity(pred_val[j,:],y_val[j,:])
            score_false = cosine_similarity(pred_val[i,:],y_val[j,:]) + cosine_similarity(pred_val[j,:],y_val[i,:])
            if score_true > score_false:
                match += 1
        match_acc[i] = match / np.sum(labels!=labels[i])
    return match_acc
    
n_comp = 0.975

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)


subj_preprocessed_path = '/home/mboos/SpeechEncoding/PreProcessed/'''+ds_path+''''
s1ds = mvpa.h5load(subj_preprocessed_path)
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)

corr_stable = ''' + str(vxl_corr) + '''

data = joblib.load('/home/mboos/SpeechEncoding/run_data/'''+data_fn+'''.pkl')
exclude_run = int("'''+identifier+'''".split('_')[-3])
nonz_eles = data['nonz_eles']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''_'+str(i)+'.pkl')) for i in xrange(10)])

data['Xtr'] = data['X'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['ytr'] = data['y'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['Xval'] = data['X'][data['chunks'][data['nonz_eles']]==exclude_run,:]
data['yval'] = data['y'][data['chunks'][data['nonz_eles']]==exclude_run,:]


influence_val = (s1ds.sa['targets'][nonz_eles])
categ = "rest"
for i,cat in enumerate(influence_val):
	if cat != 'rest':
		if categ == 'rest' or categ != cat:
			categ = cat
		influence_val[i] = categ
	else:
		influence_val[i] = categ
valset_lbls = influence_val[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles]]

if not corr_stable:
    voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,3*25)),axis=1)[0]) for voxel in data['ytr'].T ]
else:
    voxel_corr = np.concatenate([joblib.load('/home/mboos/SpeechEncoding/validation/vxlval/vxl_r2s_'''+identifier+'''_'+str(i)+'.pkl') for i in xrange(10)])

#[(zero_prop == 0)[np.logical_and(s1ds.sa['chunks'].value!=exclude_run,nonz_eles)],:]
#voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = ''' + str(nvx) + '''
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

joblib.dump(score_diffcat(data['Xval'],data['yval'][:,most_stable_vx_idx],models[most_stable_vx_idx],valset_lbls),'/home/mboos/SpeechEncoding/validation/'+{False:'stable',True:'corr'}[corr_stable]+'_rank_'''+output_prefix + '_' + identifier + ''''+'.pkl',compress=3)
joblib.dump(binary_retrieval(data['Xval'],data['yval'][:,most_stable_vx_idx],models[most_stable_vx_idx],valset_lbls),'/home/mboos/SpeechEncoding/validation/'+{False:'stable',True:'corr'}[corr_stable]+'_bin_'''+output_prefix + '_' + identifier + ''''+'.pkl',compress=3)
'''
	return ff_string

def probs_nrvx_file_string(ds_path,data_fn,identifier,output_prefix,nvx,vxl_corr=False):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import sys
sys.path.append('/home/mboos/SpeechEncoding')
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations
from sklearn.decomposition import PCA
from encoding_helpers import *

def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)


def pdf_multi_normal(x,mean,cov):
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1])/(len(correlations)))
    return ranks

def binary_retrieval(X_val,y_val,models,labels):
    import numpy as np
    def cosine_similarity(a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])
    match_acc = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        match = 0
        for j in xrange(pred_val.shape[0]):
            if labels[j] == labels[i]:
                continue
            score_true = cosine_similarity(pred_val[i,:],y_val[i,:]) + cosine_similarity(pred_val[j,:],y_val[j,:])
            score_false = cosine_similarity(pred_val[i,:],y_val[j,:]) + cosine_similarity(pred_val[j,:],y_val[i,:])
            if score_true > score_false:
                match += 1
        match_acc[i] = match / np.sum(labels!=labels[i])
    return match_acc
    
n_comp = 0.975

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)


subj_preprocessed_path = '/home/mboos/SpeechEncoding/PreProcessed/'''+ds_path+''''
s1ds = mvpa.h5load(subj_preprocessed_path)
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)

corr_stable = ''' + str(vxl_corr) + '''

data = joblib.load('/home/mboos/SpeechEncoding/run_data/'''+data_fn+'''.pkl')
exclude_run = int("'''+identifier+'''".split('_')[-3])
nonz_eles = data['nonz_eles']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''_'+str(i)+'.pkl')) for i in xrange(10)])

data['Xtr'] = data['X'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['ytr'] = data['y'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['Xval'] = data['X'][data['chunks'][data['nonz_eles']]==exclude_run,:]
data['yval'] = data['y'][data['chunks'][data['nonz_eles']]==exclude_run,:]


influence_val = (s1ds.sa['targets'][nonz_eles])
categ = "rest"
for i,cat in enumerate(influence_val):
	if cat != 'rest':
		if categ == 'rest' or categ != cat:
			categ = cat
		influence_val[i] = categ
	else:
		influence_val[i] = categ
valset_lbls = influence_val[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles]]
trainset_lbls = influence_val[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles]]

if not corr_stable:
    voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,3*25)),axis=1)[0]) for voxel in data['ytr'].T ]
else:
    voxel_corr = np.concatenate([joblib.load('/home/mboos/SpeechEncoding/validation/vxlval/vxl_r2s_'''+identifier+'''_'+str(i)+'.pkl') for i in xrange(10)])

#[(zero_prop == 0)[np.logical_and(s1ds.sa['chunks'].value!=exclude_run,nonz_eles)],:]
#voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = ''' + str(nvx) + '''
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

#n_components=[10,15,20,25,30]
mve = MultiVoxelEncoding(models = models[most_stable_vx_idx],cv=3,n_components=[5,8])
mve.fit(data['Xtr'],data['ytr'][:,most_stable_vx_idx],trainset_lbls)
#probdist = np.zeros((25,25))
#for ilbl,lbl in enumerate(np.unique(valset_lbls)):
#    lbl_given_r = np.array([ np.sum(mve.score(data['Xval'][np.where(valset_lbls==alt_lbl)[0],:],data['yval'][np.where(valset_lbls==lbl)[0],:][:,most_stable_vx_idx])) for alt_lbl in np.unique(valset_lbls)])
#    probdist[ilbl,:] = lognormalize2(lbl_given_r)

results = dict()
#results['probdist'] = mve_score(mve,data['Xval'],data['yval'],valset_lbls)
results['accuracy'] =  mve_score(mve,data['Xval'],data['yval'][:,most_stable_vx_idx],valset_lbls)
results['n_eigen'] = mve.scores_    

joblib.dump(results,'/home/mboos/SpeechEncoding/validation/'+{False:'stable',True:'corr'}[corr_stable]+'_probs_'''+output_prefix + '_' + identifier + ''''+'.pkl',compress=3)
'''
	return ff_string


def disc_decoder_file_string(ds_path,data_fn,identifier,output_prefix,nvx,vxl_corr=False):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import sys
sys.path.append('/home/mboos/SpeechEncoding')
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations
from sklearn.decomposition import PCA
from encoding_helpers import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)


def pdf_multi_normal(x,mean,cov):
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

def score_diffcat(X_val,y_val,models,labels,ctype='pearson'):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr,rankdata
        
    if ctype == 'pearson':
        cfunc = pearsonr
    else:
         cfunc = spearmanr
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])    
    ranks = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        correlations = [cfunc(pred_val[i,:],y_val[j,:])[0] for j in xrange(y_val.shape[0]) if labels[j]!=labels[i]]
        correlations.append(cfunc(pred_val[i,:],y_val[i,:])[0])
        ranks[i] = ((rankdata(correlations)[-1])/(len(correlations)))
    return ranks

def binary_retrieval(X_val,y_val,models,labels):
    import numpy as np
    def cosine_similarity(a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    pred_val = np.hstack([r.predict(X_val)[:,None] for r in models])
    match_acc = np.zeros((pred_val.shape[0],))
    for i in xrange(pred_val.shape[0]):
        match = 0
        for j in xrange(pred_val.shape[0]):
            if labels[j] == labels[i]:
                continue
            score_true = cosine_similarity(pred_val[i,:],y_val[i,:]) + cosine_similarity(pred_val[j,:],y_val[j,:])
            score_false = cosine_similarity(pred_val[i,:],y_val[j,:]) + cosine_similarity(pred_val[j,:],y_val[i,:])
            if score_true > score_false:
                match += 1
        match_acc[i] = match / np.sum(labels!=labels[i])
    return match_acc
    
n_comp = 0.975

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)


subj_preprocessed_path = '/home/mboos/SpeechEncoding/PreProcessed/'''+ds_path+''''
s1ds = mvpa.h5load(subj_preprocessed_path)
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)


data = joblib.load('/home/mboos/SpeechEncoding/run_data/'''+data_fn+'''.pkl')
exclude_run = int("'''+identifier+'''".split('_')[-3])
nonz_eles = data['nonz_eles']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''_'+str(i)+'.pkl')) for i in xrange(10)])

data['Xtr'] = data['X'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['ytr'] = data['y'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['Xval'] = data['X'][data['chunks'][data['nonz_eles']]==exclude_run,:]
data['yval'] = data['y'][data['chunks'][data['nonz_eles']]==exclude_run,:]


influence_val = (s1ds.sa['targets'][nonz_eles])
categ = "rest"
for i,cat in enumerate(influence_val):
	if cat != 'rest':
		if categ == 'rest' or categ != cat:
			categ = cat
		influence_val[i] = categ
	else:
		influence_val[i] = categ
valset_lbls = influence_val[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles]]

valset_lbls = [val[:-1] for val in valset_lbls]
valset_lbls = np.array(valset_lbls)
influence_val = np.array([infl_val[:-1] for infl_val in influence_val])
train_val = influence_val[(s1ds.sa['chunks'].value!=exclude_run)[nonz_eles]]

voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,3*25)),axis=1)[0]) for voxel in data['ytr'].T ]

#voxel_corr = np.concatenate([joblib.load('/home/mboos/SpeechEncoding/validation/vxlval/vxl_r2s_'''+identifier+'''_'+str(i)+'.pkl') for i in xrange(10)])

#[(zero_prop == 0)[np.logical_and(s1ds.sa['chunks'].value!=exclude_run,nonz_eles)],:]
#voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = ''' + str(nvx) + '''
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

#lrcv = LogisticRegressionCV(Cs=10)
#lrcv.fit(data['ytr'],train_val)
svc = LinearSVC()
svc.fit(data['ytr'][:,most_stable_vx_idx],train_val)

results = dict()
results['accuracy'] = svc.score(data['yval'][:,most_stable_vx_idx],valset_lbls)
results['confusion matrix'] = confusion_matrix(valset_lbls,svc.predict(data['yval'][:,most_stable_vx_idx]))
joblib.dump(results,'/home/mboos/SpeechEncoding/validation/discdec_svc_'''+output_prefix + '_' + identifier + ''''+'.pkl',compress=3)
'''
	return ff_string


def score_cat_probs_file_string(ds_path,data_fn,identifier,output_prefix):
	ff_string = '''from __future__ import division
import numpy as np
import scipy as sp
import sys
sys.path.append('/home/mboos/SpeechEncoding')
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations
from sklearn.decomposition import PCA
from encoding_helpers import *

def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)


def pdf_multi_normal(x,mean,cov):
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

n_comp = 0.975

def rolling_window(a,window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

#identifier = 'ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],exclude_run,subj)


subj_preprocessed_path = '/home/mboos/SpeechEncoding/PreProcessed/'''+ds_path+''''
s1ds = mvpa.h5load(subj_preprocessed_path)
events = mvpa.find_events(targets=s1ds.sa.targets,chunks=s1ds.sa.chunks)



data = joblib.load('/home/mboos/SpeechEncoding/run_data/data_'''+data_fn+'''.pkl')
exclude_run = int("'''+identifier+'''".split('_')[-3])
nonz_eles = data['nonz_eles']
models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''_'+str(i)+'.pkl')) for i in xrange(10)])

data['Xtr'] = data['X'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['ytr'] = data['y'][data['chunks'][data['nonz_eles']]!=exclude_run,:]
data['Xval'] = data['X'][data['chunks'][data['nonz_eles']]==exclude_run,:]
data['yval'] = data['y'][data['chunks'][data['nonz_eles']]==exclude_run,:]


influence_val = (s1ds.sa['targets'][nonz_eles])
categ = "rest"
for i,cat in enumerate(influence_val):
	if cat != 'rest':
		if categ == 'rest' or categ != cat:
			categ = cat
		influence_val[i] = categ
	else:
		influence_val[i] = categ
valset_lbls = influence_val[(s1ds.sa['chunks'].value==exclude_run)[nonz_eles]]

voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,3*25)),axis=1)[0]) for voxel in data['ytr'].T ]

#[(zero_prop == 0)[np.logical_and(s1ds.sa['chunks'].value!=exclude_run,nonz_eles)],:]
#voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels
nkeep = 500
most_stable_vx_idx = np.argsort(voxel_corr)[-nkeep:]

mve = MultiVoxelEncoding(n_components=n_comp,models=models[most_stable_vx_idx])


mve.fit(data['Xtr'],data['ytr'][:,most_stable_vx_idx])

probdist = np.zeros((25,25))
for ilbl,lbl in enumerate(np.unique(valset_lbls)):
    lbl_given_r = np.array([ np.sum(mve.score(data['Xval'][np.where(valset_lbls==alt_lbl)[0],:],data['yval'][np.where(valset_lbls==lbl)[0],:][:,most_stable_vx_idx])) for alt_lbl in np.unique(valset_lbls)])
    probdist[ilbl,:] = lognormalize2(lbl_given_r)

joblib.dump(probdist,'/home/mboos/SpeechEncoding/validation/'''+output_prefix+'''_'''+identifier+'''.pkl',compress=3)
'''
	return ff_string

#using most stable voxels (False) or correlation based voxel selection (True)
corrval = True

#field strength
T3 = False

#number of voxels to use
N_list = [10000,5000,2500,1000,500,250]

gen_postfix = 'on_nr_vx'

#validation strategies
func_to_validate = [probs_nrvx_file_string,disc_decoder_file_string,score_cat_bin_nrvx_file_string]


#N_list = [250]

#else withouth _pv_

for validation_func in func_to_validate:
    for T3 in [True,False]:
        for corrval in [True,False]:
            for subj in xrange(1,20):
                for exclude_run in xrange(8):
                    identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],exclude_run,subj)
                    data_fn = 'on_data_ridge_lcmfs{0}_subj_{1}'.format({True : '_3T', False : '_7T'}[T3],subj)
                    #postfix = 'probs_ar_pv_lcmfs{0}_run_{1}_subj_{2}'.format({True : '_3T', False : '_7T'}[T3],exclude_run,subj)
                    
                    if T3:
                        subj_preprocessed_path = 'subj%02dpp_3T.gzipped.hdf5' % subj
                    else:
                        subj_preprocessed_path = 'subj%02dpp.gzipped.hdf5' % subj
                        
                    for n in N_list:
                        #postfix = 'd_dc_nr' + str(n)
                        postfix = gen_postfix + str(n)
                        with open('/home/data/scratch/mboos/starter/'+'validator_'+postfix+'_'+identifier+'.py','w') as fh:
                            fh.write(validation_func(subj_preprocessed_path,data_fn,identifier,postfix,n,corrval))
                        os.system('python '+'/home/data/scratch/mboos/starter/validator_'+postfix+'_'+identifier+'.py')

        #with open('/home/data/scratch/mboos/starter/'+'validator_'+postfix+'_'+identifier+'.py','w') as fh:
        #    fh.write(score_cat_probs_file_string(subj_preprocessed_path,data_fn,identifier,postfix))
        #with open('/home/data/scratch/mboos/starter/'+'committer_'+postfix+'_'+identifier+'.sh','w') as fh:
        #    fh.write(submitfile_string('validator_'+postfix+'_'+identifier+'.py'))
        #os.system('condor_submit '+'/home/data/scratch/mboos/starter/committer_'+postfix+'_'+identifier+'.sh')

