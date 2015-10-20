#Validation all participantsimport mvpa2.suite as mvpa
from __future__ import division
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from pandas import read_csv
from sklearn.externals import joblib
from itertools import combinations

def vxl_sort_file_string(ds_path,data_fn,identifier,output_prefix):
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


data = joblib.load('/home/mboos/SpeechEncoding/run_data/data_'''+data_fn+'''.pkl')

#models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''_'+str(i)+'.pkl')) for i in xrange(10)])

voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(8,3*25)),axis=1)[0]) for voxel in data['y'].T ]

#[(zero_prop == 0)[np.logical_and(s1ds.sa['chunks'].value!=exclude_run,nonz_eles)],:]
#voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels

joblib.dump(voxel_corr,'/home/mboos/SpeechEncoding/validation/vxlval/'''+output_prefix+'''_'''+identifier+'''.pkl',compress=3)
'''
	return ff_string

def vxl_r2_sort_file_string(ds_path,data_fn,identifier,output_prefix):
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


data = joblib.load('/home/mboos/SpeechEncoding/run_data/data_'''+data_fn+'''.pkl')

models = np.concatenate([np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''_'+str(i)+'.pkl')) for i in xrange(10)])

voxel_r2 = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(8,3*25)),axis=1)[0]) for voxel in data['y'].T ]

#[(zero_prop == 0)[np.logical_and(s1ds.sa['chunks'].value!=exclude_run,nonz_eles)],:]
#voxel_corr = [np.mean(sp.stats.spearmanr(np.reshape(voxel,(7,125)),axis=1)[0]) for voxel in data['ytr'].T ]
#keep the 500 most stable voxels

joblib.dump(voxel_corr,'/home/mboos/SpeechEncoding/validation/vxlval/'''+output_prefix+'''_'''+identifier+'''.pkl',compress=3)
'''
	return ff_string
 
def vxl_r2_file_string(identifier):
	ff_string = '''from __future__ import division
import numpy as np
from sklearn.externals import joblib

models = np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''.pkl'))
data = joblib.load('/home/data/scratch/mboos/input_'''+identifier+'''.pkl')

r2s = [m.score(data['Xtr'],data['ytr'][:,i]) for i,m in enumerate(models)]

joblib.dump(r2s,'/home/mboos/SpeechEncoding/validation/vxlval/vxl_r2s_'''+identifier+'''.pkl',compress=3)
'''
	return ff_string 
 

def vxl_pandora_r2_file_string(identifier,seg):
	ff_string = '''from __future__ import division
import numpy as np
from sklearn.externals import joblib
import os


FG_data = joblib.load(os.path.join('/home','mboos','SpeechEncoding','FG_data','FG_subj2_data.pkl'))

models = np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''.pkl'))

#data = joblib.load('/home/data/scratch/mboos/input_'''+identifier+'''.pkl')


data = joblib.load('/home/mboos/SpeechEncoding/run_data/data_ridge_lcmfs_7T_subj_2.pkl')

#splits = np.floor(np.linspace(0,,1001))


#nonz_eles = data['nonz_eles']

data['X'] = FG_data['Xscaler'].transform(data['X_scaler'].inverse_transform(data['X']))
data['y'] = FG_data['yscaler'].transform(data['yscaler'].inverse_transform(data['y']))



data['X'] = data['X'][data['zero_prop'][data['nonz_eles']]==0,:]
data['y'] = data['y'][data['zero_prop'][data['nonz_eles']]==0,:]



r2s = [m.score(data['X'],data['y'][:,i]) for i,m in zip([l for l in xrange('''+str(seg[0])+','+str(seg[1])+''')],models)]

joblib.dump(r2s,'/home/mboos/SpeechEncoding/validation/vxlval/vxl_nonz_pandoras_r2s_'''+identifier+'''.pkl',compress=3)
'''
	return ff_string 
 
def FG_pandora_runwise_file_string(identifier,seg,chunk):
	ff_string = '''from __future__ import division
import numpy as np
from sklearn.externals import joblib
import os


FG_data = joblib.load(os.path.join('/home','mboos','SpeechEncoding','FG_data','FG_subj2_data.pkl'))

models = np.array(joblib.load('/home/mboos/SpeechEncoding/models/output_'''+identifier+'''.pkl'))

#data = joblib.load('/home/data/scratch/mboos/input_'''+identifier+'''.pkl')


data = joblib.load('/home/mboos/SpeechEncoding/run_data/data_ridge_lcmfs_7T_subj_2.pkl')

#splits = np.floor(np.linspace(0,,1001))


#nonz_eles = data['nonz_eles']

data['X'] = FG_data['Xscaler'].transform(data['X_scaler'].inverse_transform(data['X']))
data['y'] = FG_data['yscaler'].transform(data['yscaler'].inverse_transform(data['y']))

data['X'] = data['X'][data['chunks'][data['nonz_eles']]!='''+chunk+''',:]
data['y'] = data['y'][data['chunks'][data['nonz_eles']]!='''+chunk+''',:]

r2s = [m.score(data['X'],data['y'][:,i]) for i,m in zip([l for l in xrange('''+str(seg[0])+','+str(seg[1])+''')],models)]

joblib.dump(r2s,'/home/mboos/SpeechEncoding/validation/vxlval/run_'''+chunk+'''_vxl_pandoras_r2s_'''+identifier+'''.pkl',compress=3)
'''
	return ff_string  
 

postfix = 'pandora_voxel_corr'

#T3 = False

for T3 in [True,False]:
    for subj in xrange(1,20):
        for chunk in xrange(8):
            for i in xrange(10):
                identifier = 'on_ridge_lcmfs{0}_run_{1}_subj_{2}_{3}'.format({True : '_3T', False : '_7T'}[T3],chunk,subj,i)
                with open('/home/data/scratch/mboos/starter/'+'validator_'+postfix+'_'+identifier+'.py','w') as fh:
                    fh.write(vxl_r2_file_string(identifier))
                os.system('python '+'/home/data/scratch/mboos/starter/validator_'+postfix+'_'+identifier+'.py')
