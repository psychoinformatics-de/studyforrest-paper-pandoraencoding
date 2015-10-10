from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import numpy as np
from inspect import isclass
from sklearn.mixture import log_multivariate_normal_density
from scipy.stats import multivariate_normal

def _pdf_multi_normal(x,mean,cov):
    import numpy as np
    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(cov),-0.5)
    dev = x - mean
    part3 = np.exp(-0.5*np.dot(np.dot(dev.T,np.linalg.inv(cov)),dev))
    return part1 * part2 * part3

def rolling_window(a,window):
    '''From Erik Rigtorp'''
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)


def lognormalize2(x):
    a = np.max(x) + np.logaddexp.reduce(x-np.max(x))
    return np.exp(x-a)

def mve_score(mve,X_val,y_val,valset_lbls,scoring='new'):
    '''For fitted MultiVoxelEncoding object mve, returns the decoding accuracy for X_val,y_val'''
    catdist = np.zeros((25,5))
    probdist = np.zeros((25,25))
    for ilbl,lbl in enumerate(np.unique(valset_lbls)):
        lbl_given_r = np.array([ np.sum(mve.score(X_val[np.where(valset_lbls==alt_lbl)[0],:],y_val[np.where(valset_lbls==lbl)[0],:])) for alt_lbl in np.unique(valset_lbls)])
        probdist[ilbl,:] = lognormalize2(lbl_given_r)
        catdist[ilbl,:] = lognormalize2(np.sum(np.reshape(lbl_given_r,(5,-1)),axis=1))
    
    catpbz = np.zeros(catdist.shape)
    maxes = np.argmax(catdist,axis=1)
    catpbz[np.arange(25),maxes] = 1
    
    if scoring=='new':    
        return np.sum(np.reshape(catpbz,(5,5,5)),1)
    elif scoring=='old':
        pbz = np.zeros(probdist.shape)
        maxes = np.argmax(probdist,axis=1)
        pbz[np.arange(25),maxes] = 1
    
        per_stimulus_acc = np.sum(np.diag(pbz))/pbz.shape[0]
        cat_probdist = np.reshape(pbz,(25,5,5))
        corr_cat = 0
        for i in xrange(25):
            corr_cat += np.sum(cat_probdist[i,np.floor(i / 5.0),:])
        per_category_acc = corr_cat / 25
        return (per_category_acc,per_stimulus_acc)
        
    

class MultiVoxelEncoding(BaseEstimator):
    def __init__(self,models=RidgeCV,n_components=0.95,cv=None,scoring=None):
        if isclass(models):
            if not issubclass(models,BaseEstimator):
                raise RuntimeError('models needs to be a sklearn Estimator class or array-like of instances of sklearn estimators')
        else:
            if isinstance(models,list):
                models = np.array(models)
            if not isinstance(models,np.ndarray):
                raise RuntimeError('models needs to be a sklearn Estimator class or array-like of instances of sklearn estimators')
            if not issubclass(type(models[0]),BaseEstimator):
                raise RuntimeError('models needs to be a sklearn Estimator class or array-like of instances of sklearn estimators')
        if cv is not None:
            if not isinstance(n_components,list):
                raise RuntimeError('n_components needs to be a list for cross-validation')
            if not all(isinstance(n,int) for n in n_components):
                raise RuntimeError('all elements of n_components need to be integers')

        
        self.models = models
        self.n_components = n_components
        self.fitted = False
        self.cv = cv
        self.scoring = scoring
        
    def _pca_transform(self,y):
        pca_y = self.pca.transform(y)
        return pca_y/np.linalg.norm(pca_y)
    
    def fit(self,X,y,labels=None):
        if not y.shape[1] == len(self.models):
            raise RuntimeError('Nr. of columns of y needs to be equal to length of models')
        if self.cv is None:
            pca = PCA(n_components=self.n_components)
            pred_y = np.hstack([r.predict(X)[:,None] for r in self.models])
            pca.fit(pred_y)
            self.pca = pca
            #pca_predy = pca.transform(pred_y)
            #pca_y = pca.transform(y)
            #pca_predy,pca_y = (pcav/np.linalg.norm(pcav) for pcav in [pca_predy,pca_y])
            pca_predy,pca_y = (self._pca_transform(pcav) for pcav in [pred_y,y])
            self.pca_cov = np.cov(pca_y - pca_predy,rowvar=0)
            self.fitted = True
        else:
            #from sklearn.cross_validation import KFold
            if self.scoring is None:
                self.scoring = self.score
            self.scores_ = {}
            #kfold = KFold(n=y.shape[0],n_folds=self.cv)
            for value in self.n_components:
                scores = []
                indices = np.repeat(np.arange(7),75)
                splits = np.random.permutation(np.arange(7))[:-1].reshape(3,2)
                for i in xrange(3):
                    cv_mve = MultiVoxelEncoding(models=self.models,n_components=value)
                    booli = np.logical_or(indices==splits[i][0],indices==splits[i][1])
                    cv_mve.fit(X[~booli,:],y[~booli,:])
                    if labels != None:
                        labels = np.array(labels)
                        scores.append(np.sum(np.diag(mve_score(cv_mve,X[booli,:],y[booli,:],labels[booli])))/25)
#                    else:
#                        scores.append(cv_mve.score(X[test,:],y[test,:]))
                self.scores_[value] = np.mean(scores)
            self.n_components = max(self.scores_,key=self.scores_.get)
            pca = PCA(n_components=self.n_components)
            pred_y = np.hstack([r.predict(X)[:,None] for r in self.models])
            pca.fit(pred_y)
            self.pca = pca
            #pca_predy = pca.transform(pred_y)
            #pca_y = pca.transform(y)
            #pca_predy,pca_y = (pcav/np.linalg.norm(pcav) for pcav in [pca_predy,pca_y])
            pca_predy,pca_y = (self._pca_transform(pcav) for pcav in [pred_y,y])
            self.pca_cov = np.cov(pca_y - pca_predy,rowvar=0)
            self.fitted = True
                
            
    
    
    def predict(self,X):
        if not self.fitted:
            raise RuntimeError('must call "fit" first')
        return self._pca_transform(np.hstack([r.predict(X)[:,None] for r in self.models]))
    
    def score(self,X,y):
        if not self.fitted:
            raise RuntimeError('must call "fit" first')
        return np.array([log_multivariate_normal_density(row[:,0][None,:],row[:,1][None,:],self.pca_cov[None,:,:],covariance_type='full') for row in np.dstack((self._pca_transform(y),self.predict(X)))])
        #return np.array([multivariate_normal.logpdf(row[:,0][None,:],row[:,1][None,:],cov=self.pca_cov[None,:,:]) for row in np.dstack((self._pca_transform(y),self.predict(X)))])


    
    


def rolling_window(a,window):
    '''From Erik Rigtorp
    a			-		Numpy array
    window		-		length of window'''
    import numpy as np
    shape = a.shape[:-1] + (a.shape[0]+1-window,window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)
    
def submitfile_string(name):
	submstring = '''Executable = /usr/bin/python
Universe = vanilla
initialdir = /home/mboos/SpeechEncoding
request_cpus = 5
request_memory = 18000
getenv = True
kill_sig = 2
when_to_transfer_output = ON_EXIT_OR_EVICT
environment = PYTHONPATH=/usr/lib/python2.7
Arguments = -- /home/data/scratch/mboos/''' + name + ''' 
error  = /home/mboos/SpeechEncoding/CondorLogFiles/$(PROCESS).$(CLUSTER).err
output = /home/mboos/SpeechEncoding/CondorLogFiles/$(PROCESS).$(CLUSTER).out 
log = /home/mboos/SpeechEncoding/CondorLogFiles/$(PROCESS).$(CLUSTER).log
queue'''
	return submstring


def fitfile_string(input_fn,output_fn):
	ff_string = '''from sklearn.linear_model import RidgeCV
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
from sklearn.externals.joblib import Parallel,delayed,dump,load

def _rfit(self,args):
	return self.fit(*args)

data = load('/home/data/scratch/mboos/''' + input_fn + '''')
X_train = data['Xtr']
y_train = data['ytr']
ridges = Parallel(n_jobs=5)(delayed(_rfit)(RidgeCV(alphas=[10000.0,100000.0,1000000.0]),(X_train,y_train[:,i])) for i in xrange(y_train.shape[1]))
dump(ridges,'/home/mboos/SpeechEncoding/models/''' + output_fn + '''',compress=3)'''
	return ff_string


def MUE_helper2(X_train,y_train,postfix,nsplit=11):
	'''Creates nsplit different condor jobs with n_jobs'''
	from sklearn.externals.joblib import dump
	import numpy as np
	import os
	splits = np.floor(np.linspace(0,y_train.shape[1],nsplit))
	for i in xrange(nsplit-1):
		dump({'Xtr':X_train,'ytr':y_train[:,splits[i]:splits[i+1]]},'/home/data/scratch/mboos/input_'+postfix+'_'+str(i)+'.pkl')
		with open('/home/data/scratch/mboos/starter/'+'fitter_'+postfix+'_'+str(i)+'.py','w') as fh:
			fh.write(fitfile_string('input_'+postfix+'_'+str(i)+'.pkl','output_'+postfix+'_'+str(i)+'.pkl'))
		os.system('python '+'/home/data/scratch/mboos/starter/committer_'+postfix+'_'+str(i)+'.sh')

