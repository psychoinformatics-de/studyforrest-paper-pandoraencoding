# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import mvpa2.suite as mvpa
import numpy as np
import scipy as sp
import os
import nibabel as nb
import mvpa2.suite as mvpa
import glob
from scipy.io import loadmat
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from pandas import read_csv

for subj in xrange(1,20):
	#alternative for 3T pandora
	datapath = os.path.join('/home','data','psyinf','pandora_dartmouth','data')
	#boldlist = sorted(glob.glob(os.path.join(datapath,'task002*')))
	flavor = 'moco_to_subjbold3Tp2'

	mask_fname = os.path.join('/home','mboos','SpeechEncoding','temporal_lobe_mask_3T_subj' + str(subj) + 'bold.nii.gz')
	model = 1
	task = 1
	dhandle = mvpa.OpenFMRIDataset(datapath)

	T3 = True

	# <codecell>

	#load and save all datasets
	run_datasets = []

	#cat_nrs = np.array(['country0','symphonic0','rocknroll0','ambient0','ambient1','metal0','ambient2','symphonic1','country1','metal1','metal2','rocknroll1','metal3','symphonic2','symphonic3','ambient3','rocknroll2','rocknroll3','country2','rocknroll4','symphonic4','metal4','country3','country4','ambient4'])
	cat_nrs = np.array(['ambient3', 'ambient4', 'ambient1', 'ambient0', 'ambient2',
	'rocknroll4', 'rocknroll3', 'rocknroll2', 'rocknroll0',
	'rocknroll1', 'metal0', 'metal4', 'metal3', 'metal1', 'metal2',
	'symphonic1', 'symphonic3', 'symphonic2', 'symphonic0',
	'symphonic4', 'country3', 'country0', 'country1', 'country2',
	'country4'])
	nmbrs = []
	seqs = [[6,2,0,3,7,5,4,1],[1,5,0,7,3,2,4,6]]


	if T3:
		with open('/home/data/psyinf/pandora_dartmouth/stimuli_labels.txt') as fh:
			for i,ln in enumerate(fh.readlines()):
				nmbrs.append(ln)
			nmbrs = np.array([int(n.strip('\n')) for n in nmbrs])
			run_info = np.reshape(cat_nrs[nmbrs-1],(8,25))
	else:
		run_info = [read_csv(os.path.join('/home','mboos','SpeechEncoding','run'+str(i)+'.csv')) for i in xrange(8)]
		run_info = [map(lambda x : x.split('_')[0]+x.split('_')[1][2],ri['stim']) for ri in run_info]
		run_info = np.array(run_info)[seqs[subj-1]]



	for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
		run_events = dhandle.get_bold_run_model(model,subj,run_id)

	#add unique genre description
		for i in xrange(len(run_events)):
			run_events[i]['condition'] = run_info[run_id-1][i]

		run_ds = dhandle.get_bold_run_dataset(subj,task,run_id,chunks=run_id-1,mask=mask_fname,flavor=flavor)
		run_ds.sa['targets'] = mvpa.events2sample_attr(run_events,run_ds.sa.time_coords,noinfolabel='rest')
		run_datasets.append(run_ds)

	s1ds = mvpa.vstack(run_datasets)

	mvpa.poly_detrend(s1ds,polyord=1,chunks_attr='chunks')
     #zscore now not vs rest
	mvpa.zscore(s1ds)

	s1ds.save(os.path.join('/home','mboos','SpeechEncoding','PreProcessed','subj%02dnpp_3T.gzipped.hdf5' % subj),compression=9)


